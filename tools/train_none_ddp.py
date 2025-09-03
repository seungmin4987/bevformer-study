#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from os import path as osp
import warnings
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import mmcv

from mmcv import Config
from mmcv.runner import build_optimizer, load_checkpoint
from mmcv.parallel import scatter

from mmdet.apis import set_random_seed
from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmseg import __version__ as mmseg_version

from mmdet3d.utils import collect_env, get_root_logger
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
try:
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
except Exception:
    from mmdet3d.datasets import build_dataloader

# -------------------------------
# 하드코딩 섹션 (필요에 맞게 수정)
# -------------------------------
CONFIG_FILE   = 'projects/configs/bevformer/bevformer_tiny.py'
WORK_DIR      = 'work_dirs/solo_bevformer_tiny'
DEVICE        = 'cuda:0'   # 'cpu' 가능
SEED          = 0
DETERMINISTIC = True

# CUDNN 옵션
CUDNN_BENCHMARK     = False
CUDNN_DETERMINISTIC = True
DISABLE_CUDNN       = False

# 체크포인트/재개
LOAD_FROM   = None                   # 사전학습 가중치(.pth)
RESUME_FROM = None                   # 학습 재개 체크포인트(.pth)

# 학습 관련 오버라이드(없으면 cfg 값 사용)
MAX_EPOCHS_OVERRIDE   = None         # None이면 cfg.runner.max_epochs 사용
ACCUMULATION_STEPS    = None         # None이면 cfg에서 읽고 없으면 1
VAL_INTERVAL_OVERRIDE = None         # None이면 cfg/evaluation에서 추론
CKPT_INTERVAL_OVERRIDE= None         # None이면 cfg.checkpoint_config.interval 사용

# Precise BN 설정(작은 배치에서 권장)
PRECISE_BN_ENABLE   = False
PRECISE_BN_INTERVAL = 1     # 에폭 간격
PRECISE_BN_ITERS    = 200   # train 로더 배치 수

# -------------------------------

def import_plugins_if_needed(cfg: Config):
    """cfg.plugin_dir가 있으면 해당 패키지 import."""
    import importlib, os
    if hasattr(cfg, 'plugin_dir'):
        plugin_dir = cfg.plugin_dir
        _module_dir = os.path.dirname(plugin_dir)
        parts = _module_dir.split('/')
        module_path = parts[0]
        for p in parts[1:]:
            module_path = module_path + '.' + p
        importlib.import_module(module_path)

def make_workdir(work_dir: str, cfg_path: str):
    mmcv.mkdir_or_exist(osp.abspath(work_dir))
    ts = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(work_dir, f'{ts}.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO', name='solo')
    logger.info(f'Config path: {cfg_path}')
    return logger, ts

def build_solo_model(cfg: Config, device: str, load_from: Optional[str]):
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    model.to(device)
    if load_from:
        load_checkpoint(model, load_from, map_location='cpu', strict=False)
    return model

def build_solo_datasets_and_loaders(cfg: Config, seed: int):
    train_dataset = build_dataset(cfg.data.train)
    val_dataset = None
    if hasattr(cfg, 'workflow') and len(cfg.workflow) == 2 and hasattr(cfg.data, 'val'):
        val_dataset = build_dataset(cfg.data.val)

    samples_per_gpu = cfg.data.get('samples_per_gpu', 1)
    workers_per_gpu = cfg.data.get('workers_per_gpu', 2)

    train_loader = build_dataloader(
        train_dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=True,
        seed=seed
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=workers_per_gpu,
            num_gpus=1,
            dist=False,
            shuffle=False,
            seed=seed
        )
    return train_dataset, val_dataset, train_loader, val_loader

def create_scheduler_from_cfg(optimizer, cfg, iters_per_epoch):
    """cfg.lr_config → torch scheduler 매핑 (warmup 포함, by_epoch flag 포함)."""
    lr_cfg = cfg.get('lr_config', None)
    runner = cfg.get('runner', dict(type='EpochBasedRunner', max_epochs=12))
    max_epochs = runner.get('max_epochs', 12)
    policy = None
    if lr_cfg is not None:
        policy = lr_cfg.get('policy', None)

    warmup = None if lr_cfg is None else lr_cfg.get('warmup', None)
    warmup_iters = 0 if lr_cfg is None else int(lr_cfg.get('warmup_iters', 0))
    warmup_ratio = 0.1 if lr_cfg is None else float(lr_cfg.get('warmup_ratio', 0.1))

    base_scheduler = None
    by_epoch = True  # 기본적으로 epoch 단위

    if policy is None:
        base_scheduler = None
    elif policy.lower() == 'step':
        steps = lr_cfg.get('step', [8, 11])
        gamma = lr_cfg.get('gamma', 0.1)
        base_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=gamma)
    elif policy.lower() in ['cosine', 'cosineannealing']:
        # min_lr_ratio 우선 적용 → 없으면 min_lr 직접 읽기
        base_lr = max(pg['lr'] for pg in optimizer.param_groups)
        min_lr = lr_cfg.get('min_lr', None)
        min_lr_ratio = lr_cfg.get('min_lr_ratio', None)
        if min_lr_ratio is not None:
            eta_min = base_lr * float(min_lr_ratio)
        elif min_lr is not None:
            eta_min = float(min_lr)
        else:
            eta_min = 0.0

        base_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=eta_min
        )
    elif policy.lower() == 'poly':
        power = lr_cfg.get('power', 1.0)
        min_lr = lr_cfg.get('min_lr', 0.0)
        def poly_lambda(epoch):
            return max((1 - epoch / max_epochs) ** power, min_lr / max(pg['lr'] for pg in optimizer.param_groups))
        base_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lambda)
    elif policy.lower() == 'one_cycle':
        base_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[pg['lr'] for pg in optimizer.param_groups],
            epochs=max_epochs,
            steps_per_epoch=iters_per_epoch
        )
        by_epoch = False  # OneCycle은 보통 iter 단위로 스텝
    else:
        warnings.warn(f'[solo] Unsupported lr policy: {policy}, no scheduler will be used.')
        base_scheduler = None

    # warmup 래퍼
    initial_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def warmup_step(iter_idx):
        if warmup is None or warmup_iters <= 0:
            return
        if iter_idx < warmup_iters:
            alpha = (iter_idx + 1) / float(warmup_iters)
            factor = warmup_ratio + alpha * (1.0 - warmup_ratio)
            for i, pg in enumerate(optimizer.param_groups):
                pg['lr'] = initial_lrs[i] * factor

    return base_scheduler, warmup_step, by_epoch

def reduce_losses(losses: Dict[str, Any]):
    total_loss = 0.0
    log_vars = {}
    for name, value in losses.items():
        if isinstance(value, torch.Tensor):
            l = value.mean()
            log_vars[name] = float(l.detach().cpu())
            total_loss = total_loss + l
        elif isinstance(value, list):
            if len(value) == 0:
                continue
            l = sum(v.mean() for v in value)
            log_vars[name] = float(l.detach().cpu())
            total_loss = total_loss + l
        else:
            log_vars[name] = float(value)
            total_loss = total_loss + torch.tensor(value, device='cpu')
    if not isinstance(total_loss, torch.Tensor):
        total_loss = torch.tensor(total_loss)
    return total_loss, log_vars

def run_precise_bn(model, data_loader, device, num_iters: int, gpu_id: int, logger=None):
    """BN 통계 보정: train 모드에서 no_grad로 몇 배치 forward."""
    if not PRECISE_BN_ENABLE:
        return
    model_was_train = model.training
    model.train()
    it = iter(data_loader)
    with torch.no_grad():
        for i in range(num_iters):
            try:
                data_batch = next(it)
            except StopIteration:
                break
            data = scatter(data_batch, [gpu_id])[0]
            # forward_train은 GT 필요(Train loader 사용 권장)
            _ = model.forward_train(**data)
    if logger:
        logger.info(f'[solo] Precise BN updated with {min(num_iters, len(data_loader))} iters.')
    if not model_was_train:
        model.eval()

def single_gpu_validate(model, val_loader, val_dataset, device, gpu_id, cfg, logger):
    """공식 single_gpu_test 사용 → dataset.evaluate로 metric 계산."""
    try:
        from mmdet3d.apis import single_gpu_test
    except Exception as e:
        logger.warning(f'[solo] single_gpu_test import 실패: {e}. 간단 루프로 대체(정식 metric 불가).')
        # 간이 루프(정식 metric 계산 어려움 → None 반환)
        model.eval()
        with torch.no_grad():
            for _data in val_loader:
                data = scatter(_data, [gpu_id])[0]
                # 보통 apis가 처리하는 전처리/후처리가 빠짐
                _ = model(**data, return_loss=False)
        return None, None

    model.eval()
    outputs = single_gpu_test(model, val_loader, show=False)
    eval_cfg = cfg.get('evaluation', {})
    metric = eval_cfg.get('metric', None)
    metric_options = eval_cfg.get('metric_options', None)
    if metric is None:
        eval_res = val_dataset.evaluate(outputs)
    else:
        if metric_options is not None:
            eval_res = val_dataset.evaluate(outputs, metric=metric, metric_options=metric_options)
        else:
            eval_res = val_dataset.evaluate(outputs, metric=metric)

    # best를 고를 키 선택
    save_best = eval_cfg.get('save_best', None)
    key = None
    if save_best and save_best in eval_res:
        key = save_best
    elif 'NDS' in eval_res:
        key = 'NDS'
    elif 'mAP' in eval_res:
        key = 'mAP'
    else:
        # 첫 번째 scalar 키
        for k, v in eval_res.items():
            if isinstance(v, (int, float)):
                key = k
                break
    curr = eval_res.get(key, None) if key else None
    logger.info(f'[solo] Val metrics: {eval_res} | best_key={key}, curr={curr}')
    return curr, key

def save_checkpoint(path: str, model, optimizer, scheduler, meta: Dict[str, Any]):
    obj = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else None,
        'scheduler': scheduler.state_dict() if scheduler else None,
        'meta': meta
    }
    torch.save(obj, path)

def main():
    # CUDNN
    torch.backends.cudnn.enabled = not DISABLE_CUDNN
    torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
    torch.backends.cudnn.deterministic = CUDNN_DETERMINISTIC

    # cfg 로드
    cfg = Config.fromfile(CONFIG_FILE)

    # 플러그인 import
    import_plugins_if_needed(cfg)

    # work_dir/로그 준비
    cfg.work_dir = WORK_DIR
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(CONFIG_FILE)))
    logger, timestamp = make_workdir(cfg.work_dir, CONFIG_FILE)

    # 환경 로그
    env_info = '\n'.join([f'{k}: {v}' for k, v in collect_env().items()])
    logger.info('Env:\n' + env_info)
    logger.info(f'mmdet={mmdet_version}, mmdet3d={mmdet3d_version}, mmseg={mmseg_version}')
    logger.info(f'Config snippet:\n{cfg.pretty_text[:1000]}...\n')

    # 시드
    set_random_seed(SEED, deterministic=DETERMINISTIC)
    cfg.seed = SEED

    # 모델/데이터/로더
    device = DEVICE
    gpu_id = int(device.split(':')[-1]) if device.startswith('cuda') else 0
    model = build_solo_model(cfg, device=device, load_from=LOAD_FROM)

    train_dataset, val_dataset, train_loader, val_loader = build_solo_datasets_and_loaders(cfg, seed=SEED)
    model.CLASSES = getattr(train_dataset, 'CLASSES', None)

    # 옵티마이저/스케줄러
    optimizer = build_optimizer(model, cfg.optimizer)
    iters_per_epoch = len(train_loader)
    runner_cfg = cfg.get('runner', dict(type='EpochBasedRunner', max_epochs=12))
    max_epochs = MAX_EPOCHS_OVERRIDE if MAX_EPOCHS_OVERRIDE is not None else runner_cfg.get('max_epochs', 12)
    scheduler, warmup_step, by_epoch = create_scheduler_from_cfg(optimizer, cfg, iters_per_epoch)

    # Accumulation
    accum_steps = ACCUMULATION_STEPS if ACCUMULATION_STEPS is not None else getattr(cfg, 'accumulation_steps', 1)
    if accum_steps < 1: accum_steps = 1

    # 간격 설정
    ckpt_interval = CKPT_INTERVAL_OVERRIDE
    if ckpt_interval is None:
        ckpt_interval = 1
        if hasattr(cfg, 'checkpoint_config') and cfg.checkpoint_config is not None:
            ckpt_interval = int(cfg.checkpoint_config.get('interval', 1))
    val_interval = VAL_INTERVAL_OVERRIDE
    if val_interval is None:
        val_interval = cfg.get('val_interval', 1)

    # Resume
    start_epoch = 1
    global_iter = 0
    best_metric = None
    best_key = None
    if RESUME_FROM and osp.isfile(RESUME_FROM):
        ckpt = torch.load(RESUME_FROM, map_location='cpu')
        model.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
        if ckpt.get('optimizer', None): optimizer.load_state_dict(ckpt['optimizer'])
        if scheduler and ckpt.get('scheduler', None): scheduler.load_state_dict(ckpt['scheduler'])
        meta = ckpt.get('meta', {})
        start_epoch = meta.get('epoch', 0) + 1
        global_iter = meta.get('iter', 0)
        best_metric = meta.get('best', None)
        best_key = meta.get('best_key', None)
        logger.info(f'[solo] Resumed from {RESUME_FROM}: start_epoch={start_epoch}, global_iter={global_iter}, best={best_metric}')

    logger.info(f'[solo] Start training: epochs={max_epochs}, iters/epoch={iters_per_epoch}, accum={accum_steps}, by_epoch_scheduler={by_epoch}')

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        running_loss = 0.0

        # Precise BN (옵션)
        if PRECISE_BN_ENABLE and (epoch % PRECISE_BN_INTERVAL == 0):
            run_precise_bn(model, train_loader, device, PRECISE_BN_ITERS, gpu_id, logger)

        for it, data_batch in enumerate(train_loader):
            data = scatter(data_batch, [gpu_id])[0]

            # warmup (iter 기준)
            warmup_step(global_iter)

            # forward
            losses = model.forward_train(**data)
            total_loss, log_vars = reduce_losses(losses)

            # backward with accumulation
            (total_loss / accum_steps).backward()

            # step 시점(누적)
            step_now = ((it + 1) % accum_steps == 0)
            if step_now:
                # grad clip
                opt_cfg = cfg.get('optimizer_config', {})
                grad_clip_cfg = opt_cfg.get('grad_clip', {}) if isinstance(opt_cfg, dict) else {}
                if grad_clip_cfg:
                    max_norm = grad_clip_cfg.get('max_norm', 0)
                    if max_norm and max_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm, grad_clip_cfg.get('norm_type', 2))

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # by_iter 스케줄러면 (warmup 이후) iter마다 step
            if scheduler is not None and not by_epoch:
                lr_cfg = cfg.get('lr_config', {})
                warmup_iters = int(lr_cfg.get('warmup_iters', 0)) if lr_cfg else 0
                if global_iter >= warmup_iters:
                    scheduler.step()

            running_loss += float(total_loss.detach().cpu())
            global_iter += 1

            if global_iter % 20 == 0:
                lr = optimizer.param_groups[0]['lr']
                logger.info(f'[E{epoch:03d} I{global_iter:06d}] loss={float(total_loss):.4f} | lr={lr:.6f}')

        # by_epoch 스케줄러면 에폭 끝에서 step
        if scheduler is not None and by_epoch:
            scheduler.step()

        avg_loss = running_loss / max(1, iters_per_epoch)
        logger.info(f'==> Epoch {epoch}/{max_epochs} | avg_loss={avg_loss:.4f}')

        # Validation + best 저장
        if val_loader is not None and (epoch % val_interval == 0):
            curr, key = single_gpu_validate(model, val_loader, val_dataset, device, gpu_id, cfg, logger)
            if key is not None:
                if best_metric is None or (curr is not None and curr > best_metric):
                    best_metric = curr
                    best_key = key
                    best_path = osp.join(cfg.work_dir, f'best_{key}.pth')
                    save_checkpoint(best_path, model, optimizer, scheduler, meta={
                        'epoch': epoch, 'iter': global_iter, 'best': best_metric, 'best_key': best_key,
                        'seed': cfg.seed, 'CLASSES': getattr(train_dataset, 'CLASSES', None),
                        'mmdet_version': mmdet_version, 'mmdet3d_version': mmdet3d_version,
                        'mmseg_version': mmseg_version, 'timestamp': time.time()
                    })
                    logger.info(f'[solo] New best({key}={best_metric:.5f}) saved: {best_path}')

        # 주기적 체크포인트
        if (epoch % ckpt_interval) == 0:
            save_path = osp.join(cfg.work_dir, f'epoch_{epoch}.pth')
            save_checkpoint(save_path, model, optimizer, scheduler, meta={
                'epoch': epoch, 'iter': global_iter, 'best': best_metric, 'best_key': best_key,
                'seed': cfg.seed, 'CLASSES': getattr(train_dataset, 'CLASSES', None),
                'mmdet_version': mmdet_version, 'mmdet3d_version': mmdet3d_version,
                'mmseg_version': mmseg_version, 'timestamp': time.time()
            })
            logger.info(f'[solo] Saved: {save_path}')

    logger.info('[solo] Training finished.')

if __name__ == '__main__':
    # 실행 전 환경에 맞게 상단 상수만 수정하면 됩니다.
    main()
