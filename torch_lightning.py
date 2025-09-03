#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import pytorch_lightning as pl
import torchvision

#토치 라이트닝
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

#데이터 로더용
from mmcv import Config
from mmcv.parallel import scatter
import math
import copy

#데이터 모듈
from bevformer_datamodule import BEVFormerDataModule

#layer 정의
from layers.bevformer_head_pl import BEVFormerHeadPL
from layers.resnet50 import ResNet50Backbone, SingleScaleFPN
from layers.grid_mask import GridMask
from layers.utils import bbox3d2result

import numpy as np
from PIL import Image, ImageDraw

import random

# 클래스별 색상 맵핑 (NuScenes Detection 기준)
class_colors = {
    "car": (255, 0, 0),             # 빨강
    "truck": (255, 165, 0),         # 주황
    "bus": (255, 255, 0),           # 노랑
    "trailer": (0, 255, 0),         # 초록
    "construction_vehicle": (0, 128, 0),
    "pedestrian": (0, 0, 255),      # 파랑
    "motorcycle": (128, 0, 128),    # 보라
    "bicycle": (0, 255, 255),       # 청록
    "traffic_cone": (255, 20, 147), # 핑크
    "barrier": (128, 128, 128),     # 회색
}

def project_and_draw_boxes(img_np, boxes_3d, lidar2img, is_gt=True,
                           score_thr=0.3, scores=None, max_dist=50.0):
    """
    img_np: (H,W,3) numpy
    boxes_3d: BaseInstance3DBoxes (GT or Pred)
    lidar2img: (4,4) matrix
    is_gt: True면 GT, False면 Pred
    score_thr: pred에만 적용할 confidence threshold
    scores: pred box들의 confidence 점수 (tensor)
    max_dist: 최대 표시 거리 (m)
    """
    img_pil = Image.fromarray(img_np).convert("RGB")
    draw = ImageDraw.Draw(img_pil)

    if boxes_3d is None or len(boxes_3d) == 0:
        return np.array(img_pil)

    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]

    corners_3d = boxes_3d.corners  # (N,8,3)
    num_boxes = corners_3d.shape[0]
    corners_3d_hom = torch.cat(
        [corners_3d, torch.ones(num_boxes,8,1,device=corners_3d.device)], dim=-1
    )
    lidar2img_tensor = torch.tensor(lidar2img, device=corners_3d.device, dtype=torch.float32)
    corners_2d_hom = torch.einsum("ij,bkj->bki", lidar2img_tensor, corners_3d_hom)
    corners_2d = corners_2d_hom[...,:2] / corners_2d_hom[...,2:3]

    h, w = img_np.shape[:2]

    for i in range(num_boxes):
        # --- Score 필터 (Pred만 적용) ---
        if not is_gt and scores is not None:
            if scores[i].item() < score_thr:
                continue

        # --- 거리 필터 (Pred만 적용) ---
        center = boxes_3d.tensor[i, :3]
        dist = torch.norm(center).item()
        if not is_gt and dist > max_dist:
            continue
        
        # --- FOV 필터 (화면 안에 일부라도 보이는 경우만) ---
        # FOV 필터 (화면 안 & 깊이가 양수인 경우만)
        box_corners = corners_2d[i]
        depths = corners_2d_hom[i, :, 2]   # 각 코너의 z (depth)
        box_corners = corners_2d[i]
        in_front = torch.any(depths > 0)
        in_fov = torch.any(
            (box_corners[:,0] >= 0) & (box_corners[:,0] < w) &
            (box_corners[:,1] >= 0) & (box_corners[:,1] < h)
        )
        if not (in_fov and in_front):
            continue

        # --- 색상 (GT=초록, Pred=빨강) ---
        color = "green" if is_gt else "red"

        # --- 박스 그리기 ---
        for start, end in edges:
            p1 = tuple(box_corners[start].int().tolist())
            p2 = tuple(box_corners[end].int().tolist())
            draw.line([p1,p2], fill=color, width=2)

    return np.array(img_pil)



class LitBevFormer(pl.LightningModule):
    def __init__(self, cfg_path='projects/configs/bevformer/bevformer_tiny.py'):
        super().__init__()
        self.cfg = Config.fromfile(cfg_path)
        self.save_hyperparameters('cfg_path')

        # Build model components
        self.img_backbone = ResNet50Backbone(pretrained=True)#build_backbone(self.cfg.model.img_backbone)
        self.img_neck = SingleScaleFPN()#build_neck(self.cfg.model.img_neck)
                # Manually merge the correct part of train_cfg into the head's config
        pts_bbox_head_cfg = self.cfg.model.pts_bbox_head
        pts_train_cfg = self.cfg.model.get('train_cfg', {}).get('pts')
        if pts_train_cfg:
            pts_bbox_head_cfg['train_cfg'] = pts_train_cfg
        pts_bbox_head_cfg.pop('type', None)  # 'type' 키 제거
        self.pts_bbox_head = BEVFormerHeadPL(**pts_bbox_head_cfg)#build_head(pts_bbox_head_cfg)
        self.threshold = 0.2

        # Initialize weights
        #self.img_backbone.init_weights()
        #self.img_neck.init_weights()
        self.pts_bbox_head.init_weights()

        # Other components from BEVFormer wrapper
        self.use_grid_mask = self.cfg.model.get('use_grid_mask', False)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        
        # Temporal modeling state
        self.video_test_mode = self.cfg.model.get('video_test_mode', False)
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

    # --- Methods copied and adapted from BEVFormer class ---

    def extract_img_feat(self, img, img_metas, len_queue=None):
        B = img.size(0)
        if img is not None:
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped
    

    #@auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats

    def forward_pts_train(self, pts_feats, gt_bboxes_3d, gt_labels_3d, img_metas, gt_bboxes_ignore=None, prev_bev=None):
        outs = self.pts_bbox_head(pts_feats, img_metas, prev_bev)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses, outs

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        self.eval()
        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(img_feats, img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    # --- Lightning-specific methods ---

    def training_step(self, batch, batch_idx):
        data = scatter(batch, [self.device.index])[0]
        img = data['img']
        img_metas = data['img_metas']
        gt_bboxes_3d = data['gt_bboxes_3d']
        gt_labels_3d = data['gt_labels_3d']
        gt_bboxes_ignore = data.get('gt_bboxes_ignore', None)

        # Extract features and losses using the BEVFormer logic
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img_for_loss = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas_for_loss = [each[len_queue-1] for each in img_metas]
        if not img_metas_for_loss[0]['prev_bev_exists']:
            prev_bev = None
        
        img_feats = self.extract_feat(img=img_for_loss, img_metas=img_metas_for_loss)
        losses, outs = self.forward_pts_train(img_feats, gt_bboxes_3d, gt_labels_3d, img_metas_for_loss, gt_bboxes_ignore, prev_bev)

        # Log losses
        total_loss = 0
        for name, value in losses.items():
            if 'loss' in name:
                loss_mean = value.mean()
                self.log(f'train/{name}', loss_mean, on_step=True, on_epoch=False, prog_bar=False, logger=True)
                total_loss += loss_mean
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Visualize predictions (only every 10 steps)
        outs_detached = {k: v.detach() for k, v in outs.items() if isinstance(v, torch.Tensor)}
        bbox_list = self.pts_bbox_head.get_bboxes(outs_detached, img_metas_for_loss, rescale=False)
        if self.global_step > 0 and self.global_step % 10 == 0:
            mean = torch.tensor([123.675, 116.28, 103.53], device=img.device).view(3,1,1)
            std = torch.tensor([58.395, 57.12, 57.375], device=img.device).view(3,1,1)
            imgs_for_vis = img[0, -1]   # (N=6, C,H,W)

            # --- GT ---
            gt_boxes_3d_for_vis = gt_bboxes_3d[0]

            # --- Pred ---
            pred_bboxes_3d, pred_scores, pred_labels = bbox_list[0]
            pred_bboxes_3d.labels = pred_labels
            pred_bboxes_3d._meta = {"classes": self.cfg.class_names}

            gt_vis_list, pred_vis_list = [], []

            for cam_idx in range(imgs_for_vis.size(0)):
                # 디노멀라이즈
                img_denorm = imgs_for_vis[cam_idx] * std + mean
                img_np = torch.clamp(img_denorm,0,255).byte().cpu().permute(1,2,0).numpy()
                lidar2img = img_metas_for_loss[0]['lidar2img'][cam_idx]

                # --- GT (무조건 표시) ---
                gt_img = project_and_draw_boxes(img_np, gt_boxes_3d_for_vis, lidar2img, is_gt=True)
                gt_vis_list.append(torch.from_numpy(gt_img).permute(2,0,1))

                # --- Pred (score_thr, 거리, FOV 적용) ---
                pred_img = project_and_draw_boxes(img_np, pred_bboxes_3d, lidar2img,
                                                is_gt=False, scores=pred_scores,
                                                score_thr=self.threshold, max_dist=50.0)
                pred_vis_list.append(torch.from_numpy(pred_img).permute(2,0,1))

            # 각각 grid로 묶어서 TensorBoard에 기록
            grid_gt = torchvision.utils.make_grid(gt_vis_list, nrow=3)
            grid_pred = torchvision.utils.make_grid(pred_vis_list, nrow=3)

            self.logger.experiment.add_image("training/gt_multi_view", grid_gt, self.global_step)
            self.logger.experiment.add_image("training/pred_multi_view", grid_pred, self.global_step)

        return total_loss

    def validation_step(self, batch, batch_idx):
        data = scatter(batch, [self.device.index])[0]
        # Unwrap the list from MultiScaleFlipAug3D
        img = data['img'][0]
        img_metas = data['img_metas'][0]

        # Logic from BEVFormer.forward_test
        if img_metas[0]['scene_token'] != self.prev_frame_info['scene_token']:
            self.prev_frame_info['prev_bev'] = None
        self.prev_frame_info['scene_token'] = img_metas[0]['scene_token']

        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        tmp_pos = copy.deepcopy(img_metas[0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0]['can_bus'][-1] = 0
            img_metas[0]['can_bus'][:3] = 0

        # simple_test logic
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        outs = self.pts_bbox_head(img_feats, img_metas, prev_bev=self.prev_frame_info['prev_bev'])
        new_prev_bev = outs['bev_embed']
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=False)
        bbox_results = [bbox3d2result(bboxes, scores, labels) for bboxes, scores, labels in bbox_list]

        # Update temporal state
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev

        return bbox_results

    def draw_bev_predictions(self, img_tensor, pred_boxes, lidar2img, gt_boxes=None):
        # De-normalize image
        # cfg.img_norm_cfg is not available here, so we hardcode the values for now.
        # A better way would be to pass this from the main script.
        mean = torch.tensor([123.675, 116.28, 103.53], device=img_tensor.device)
        std = torch.tensor([58.395, 57.12, 57.375], device=img_tensor.device)
        img_denorm = img_tensor * std[:, None, None] + mean[:, None, None]
        img_denorm = torch.clamp(img_denorm, 0, 255)

        img_np = np.array(img_denorm.cpu().permute(1, 2, 0), dtype=np.uint8)
        h, w, _ = img_np.shape
        img_pil = Image.fromarray(img_np).convert("RGB")
        draw = ImageDraw.Draw(img_pil)

        # Draw Predicted Boxes (Red)
        if pred_boxes is not None and len(pred_boxes) > 0:
            corners_3d = pred_boxes.corners # (N, 8, 3)
            num_boxes = corners_3d.shape[0]
            corners_3d_hom = torch.cat([corners_3d, torch.ones(num_boxes, 8, 1, device=corners_3d.device)], dim=-1) # (N, 8, 4)

            # Project to 2D
            lidar2img_tensor = torch.tensor(lidar2img, device=corners_3d.device, dtype=torch.float32)
            corners_2d_hom = torch.einsum('ij, bkj -> bki', lidar2img_tensor, corners_3d_hom)
            corners_2d = corners_2d_hom[..., :2] / corners_2d_hom[..., 2:3]

            # Define connections for the box edges
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
                (4, 5), (5, 6), (6, 7), (7, 4),  # top face
                (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
            ]

            for i in range(num_boxes):
                box_corners = corners_2d[i] # (8, 2) 
                
                in_image = torch.all((box_corners[:, 0] >= 0) & (box_corners[:, 0] < w) & \
                                     (box_corners[:, 1] >= 0) & (box_corners[:, 1] < h))
                
                if in_image:
                    for start, end in edges:
                        p1 = tuple(box_corners[start].int().tolist())
                        p2 = tuple(box_corners[end].int().tolist())
                        draw.line([p1, p2], fill="red", width=2)

        # Draw Ground Truth Boxes (Green)
        if gt_boxes is not None and len(gt_boxes) > 0:
            corners_3d = gt_boxes.corners # (N, 8, 3)
            num_boxes = corners_3d.shape[0]
            corners_3d_hom = torch.cat([corners_3d, torch.ones(num_boxes, 8, 1, device=corners_3d.device)], dim=-1) # (N, 8, 4)

            # Project to 2D
            lidar2img_tensor = torch.tensor(lidar2img, device=corners_3d.device, dtype=torch.float32)
            corners_2d_hom = torch.einsum('ij, bkj -> bki', lidar2img_tensor, corners_3d_hom)
            corners_2d = corners_2d_hom[..., :2] / corners_2d_hom[..., 2:3]

            # Define connections for the box edges
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
                (4, 5), (5, 6), (6, 7), (7, 4),  # top face
                (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
            ]

            for i in range(num_boxes):
                box_corners = corners_2d[i] # (8, 2) 
                
                in_image = torch.all((box_corners[:, 0] >= 0) & (box_corners[:, 0] < w) & \
                                     (box_corners[:, 1] >= 0) & (box_corners[:, 1] < h))
                
                if in_image:
                    for start, end in edges:
                        p1 = tuple(box_corners[start].int().tolist())
                        p2 = tuple(box_corners[end].int().tolist())
                        draw.line([p1, p2], fill="green", width=2)

        # Convert back to tensor
        img_tensor_out = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float() / 255.0
        return img_tensor_out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.cfg.optimizer['lr'], 
            weight_decay=self.cfg.optimizer['weight_decay']
        )

        if 'lr_config' in self.cfg:
            lr_config = self.cfg.lr_config.copy()
            policy = lr_config.pop('policy')
            
            if policy.lower() == 'cosineannealing':
                if self.trainer.max_steps and self.trainer.max_steps > 0:
                    max_iters = self.trainer.max_steps
                else:
                    num_devices = self.trainer.world_size if self.trainer.world_size > 0 else 1
                    samples_per_gpu = self.cfg.data.samples_per_gpu
                    steps_per_epoch = len(self.trainer.datamodule.train_dataset) // (num_devices * samples_per_gpu)
                    max_iters = self.cfg.runner.max_epochs * steps_per_epoch

                warmup_iters = lr_config.get('warmup_iters', 0)
                warmup_ratio = lr_config.get('warmup_ratio', 1.0/3)
                min_lr_ratio = lr_config.get('min_lr_ratio', 1e-3)

                def lr_lambda_fn(current_step: int):
                    if current_step < warmup_iters:
                        return warmup_ratio + (1 - warmup_ratio) * float(current_step) / float(warmup_iters)
                    else:
                        progress = float(current_step - warmup_iters) / float(max(1, max_iters - warmup_iters))
                        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))

                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_fn)
                
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step",
                        "frequency": 1,
                    },
                }

        return optimizer

if __name__ == '__main__':
    # --- Configs ---
    cfg_path = 'projects/configs/bevformer/bevformer_tiny.py'
    cfg = Config.fromfile(cfg_path)

    # --- Init Model and Data ---
    model = LitBevFormer(cfg_path=cfg_path)
    datamodule = BEVFormerDataModule(cfg=cfg)

    # --- Logger and Callbacks ---
    logger = TensorBoardLogger("tb_logs", name="bevformer_tiny")
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        every_n_epochs=cfg.checkpoint_config.interval
    )

    # # --- Trainer ---
    # trainer = pl.Trainer(
    #     max_epochs=cfg.runner.max_epochs,
    #     logger=logger,
    #     callbacks=[checkpoint_callback],
    #     gradient_clip_val=cfg.optimizer_config.grad_clip.max_norm,
    #     gradient_clip_algorithm="norm",
    #     log_every_n_steps=cfg.log_config.interval,
    #     check_val_every_n_epoch=0,#cfg.evaluation.interval,
    #     accelerator="gpu" if torch.cuda.is_available() else "cpu",
    #     devices=1,
    #     overfit_batches=1,
    #     num_sanity_val_steps=0       # sanity check 끔
    # )

    trainer = pl.Trainer(
        max_epochs=10000,
        logger=logger,
        overfit_batches=1,
        num_sanity_val_steps=0,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1
    )

    # --- Start Training ---
    trainer.fit(model, datamodule=datamodule)

    print("Training complete!")