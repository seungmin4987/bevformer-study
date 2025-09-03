# import pytorch_lightning as pl
# from functools import partial

# from torch.utils.data import DataLoader

# from mmcv import Config
# from mmcv.parallel import collate
# from mmdet.datasets import DistributedGroupSampler
# from mmdet.datasets.builder import worker_init_fn

# from projects.mmdet3d_plugin.datasets.nuscenes_dataset import CustomNuScenesDataset


# class BEVFormerDataModule(pl.LightningDataModule):
#     def __init__(self, cfg: Config):
#         super().__init__()
#         self.cfg = cfg

#     def setup(self, stage=None):
#         if stage == 'fit' or stage is None:
#             train_cfg = self.cfg.data.train.copy()
#             train_cfg.pop('type', None)
#             train_cfg.pop('samples_per_gpu', None)
#             self.train_dataset = CustomNuScenesDataset(**train_cfg)

#             val_cfg = self.cfg.data.val.copy()
#             val_cfg.pop('type', None)
#             val_cfg.pop('samples_per_gpu', None)
#             self.val_dataset = CustomNuScenesDataset(**val_cfg)

#     def train_dataloader(self):
#         rank = self.trainer.global_rank
#         num_gpus = self.trainer.world_size
#         samples_per_gpu = self.cfg.data.samples_per_gpu
        
#         sampler = DistributedGroupSampler(self.train_dataset, samples_per_gpu, num_gpus, rank)
        
#         batch_size = samples_per_gpu
#         num_workers = self.cfg.data.workers_per_gpu

#         init_fn = partial(
#             worker_init_fn, num_workers=num_workers, rank=rank,
#             seed=self.cfg.get('seed', 42))

#         return DataLoader(
#             self.train_dataset,
#             batch_size=batch_size,
#             sampler=sampler,
#             num_workers=num_workers,
#             collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
#             pin_memory=False,
#             worker_init_fn=init_fn,
#             persistent_workers=(num_workers > 0)
#         )

#     def val_dataloader(self):
#         batch_size = self.cfg.data.samples_per_gpu
#         num_workers = self.cfg.data.workers_per_gpu

#         return DataLoader(
#             self.val_dataset,
#             batch_size=batch_size,
#             sampler=None,
#             num_workers=num_workers,
#             collate_fn=partial(collate, samples_per_gpu=batch_size),
#             pin_memory=False,
#             shuffle=False,
#             persistent_workers=(num_workers > 0)
#         )

import pytorch_lightning as pl
from functools import partial

from torch.utils.data import DataLoader

from mmcv import Config
from mmcv.parallel import collate
from mmdet.datasets import DistributedGroupSampler
from mmdet.datasets.builder import worker_init_fn

from projects.mmdet3d_plugin.datasets.nuscenes_dataset import CustomNuScenesDataset

class BEVFormerDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_cfg = self.cfg.data.train.copy()
            train_cfg.pop('type', None)
            train_cfg.pop('samples_per_gpu', None)
            self.train_dataset = CustomNuScenesDataset(**train_cfg)

            val_cfg = self.cfg.data.val.copy()
            val_cfg.pop('type', None)
            val_cfg.pop('samples_per_gpu', None)
            self.val_dataset = CustomNuScenesDataset(**val_cfg)

    def train_dataloader(self):
        samples_per_gpu = self.cfg.data.samples_per_gpu
        num_workers = self.cfg.data.workers_per_gpu

        # trainer가 attach 안 됐을 때 기본값
        rank = getattr(self.trainer, "global_rank", 0) if self.trainer else 0
        num_gpus = getattr(self.trainer, "world_size", 1) if self.trainer else 1

        # 오버피팅 모드에서는 sampler 사용하지 않음
        if self.trainer is not None and getattr(self.trainer, "overfit_batches", 0) > 0:
            sampler = None
        else:
            sampler = DistributedGroupSampler(
                self.train_dataset, samples_per_gpu, num_gpus, rank
            )

        init_fn = partial(
            worker_init_fn, num_workers=num_workers, rank=rank,
            seed=self.cfg.get('seed', 42)
        )

        return DataLoader(
            self.train_dataset,
            batch_size=samples_per_gpu,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
            pin_memory=False,
            worker_init_fn=init_fn,
            persistent_workers=(num_workers > 0)
        )

    def val_dataloader(self):
        # 오버피팅 모드일 때는 validation 아예 건너뜀
        if self.trainer is not None and getattr(self.trainer, "overfit_batches", 0) > 0:
            return []
        else:
            batch_size = self.cfg.data.samples_per_gpu
            num_workers = self.cfg.data.workers_per_gpu

            return DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                sampler=None,
                num_workers=num_workers,
                collate_fn=partial(collate, samples_per_gpu=batch_size),
                pin_memory=False,
                shuffle=False,
                persistent_workers=(num_workers > 0)
            )
