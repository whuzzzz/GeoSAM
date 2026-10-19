
from loguru import logger

import torch
import pytorch_lightning as pl
from matplotlib import pyplot as plt
plt.switch_backend('agg')

# from src.xoftr import XoFTR_Pretrain
# from src.xoftr import XoFTR_Pretrain_Light
from src.xoftr import XoFTR_Pretrain_dino, XoFTR_Pretrain_Light_convnextv2, XoFTR_Pretrain_dino_resnet_panmatch_light
from src.losses.xoftr_loss_pretrain import XoFTRLossPretrain
from src.optimizers import build_optimizer, build_scheduler
from src.utils.plotting import make_mae_figures
from src.utils.comm import all_gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler
from src.utils.pretrain_utils import generate_random_masks, get_target


class PL_XoFTR_Pretrain(pl.LightningModule):
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        # Misc
        self.config = config  # full config
        
        _config = lower_config(self.config)
        self.xoftr_cfg = lower_config(_config['xoftr'])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)

        # generator to create the same masks for validation
        self.val_seed = self.config.PRETRAIN.VAL_SEED
        self.val_generator = torch.Generator(device="cuda").manual_seed(self.val_seed)
        self.mae_margins = config.PRETRAIN.MAE_MARGINS

        # Matcher: XoFTR
        # self.matcher = XoFTR_Pretrain(config=_config['xoftr'])
        # self.matcher = XoFTR_Pretrain_Light(config=_config['xoftr'])
        # self.matcher = XoFTR_Pretrain_Light_convnextv2(config=_config['xoftr'])
        self.matcher = XoFTR_Pretrain_dino_resnet_panmatch_light(config=_config['xoftr'])
        self.loss = XoFTRLossPretrain(_config)

        # Pretrained weights
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
            self.matcher.load_state_dict(state_dict, strict=False)
            logger.info(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")
        
        # Testing
        self.dump_dir = dump_dir
        
    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]
    
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric=None):
        """
        正确的学习率调度器步骤方法签名
        PyTorch Lightning 1.7+ 版本的要求
        """
        scheduler.step()

    def optimizer_step(
            self, epoch, batch_idx, optimizer, optimizer_idx,
            optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + \
                    (self.trainer.global_step / self.config.TRAINER.WARMUP_STEP) * \
                    abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif self.config.TRAINER.WARMUP_TYPE == 'constant':
                pass
            else:
                raise ValueError(f'Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}')

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
    
    def _trainval_inference(self, batch, generator=None):
        generate_random_masks(batch,
                        patch_size=self.config.PRETRAIN.PATCH_SIZE,
                        mask_ratio=self.config.PRETRAIN.MASK_RATIO,
                        generator=generator,
                        margins=self.mae_margins)
        
        with self.profiler.profile("XoFTR"):
            self.matcher(batch)
            
        with self.profiler.profile("Compute losses"):
            # Create target pacthes to reconstruct
            get_target(batch)
            self.loss(batch)
    
    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        
        # logging
        if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
            # scalars
            for k, v in batch['loss_scalars'].items():
                self.logger[0].experiment.add_scalar(f'train/{k}', v, self.global_step)
                if self.config.TRAINER.USE_WANDB:
                    self.logger[1].log_metrics({f'train/{k}': v}, self.global_step)
            
            if self.config.TRAINER.ENABLE_PLOTTING:
                figures = make_mae_figures(batch)
                for i, figure in enumerate(figures):
                    self.logger[0].experiment.add_figure(
                    f'train_mae/node_{self.trainer.global_rank}-device_{self.device.index}-batch_{i}',
                    figure, self.global_step)

        return {'loss': batch['loss']}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        if self.trainer.global_rank == 0:
            self.logger[0].experiment.add_scalar(
                'train/avg_loss_on_epoch', avg_loss,
                global_step=self.current_epoch)
            if self.config.TRAINER.USE_WANDB:
                self.logger[1].log_metrics(
                    {'train/avg_loss_on_epoch': avg_loss},
                    self.current_epoch)
    
    def validation_step(self, batch, batch_idx):
        self._trainval_inference(batch, self.val_generator)
    
        val_plot_interval = max(self.trainer.num_val_batches[0] // \
                                 (self.trainer.num_gpus * self.n_vals_plot), 1)
        figures = []
        if batch_idx % val_plot_interval == 0:
            figures = make_mae_figures(batch)

        return {
            'loss_scalars': batch['loss_scalars'],
            'figures': figures,
        }
        
    def validation_epoch_end(self, outputs):
        self.val_generator.manual_seed(self.val_seed)
        # handle multiple validation sets
        multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        
        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training
            cur_epoch = self.trainer.current_epoch
            # if not self.trainer.resume_from_checkpoint and self.trainer.running_sanity_check:
            #     cur_epoch = -1

            if not self.trainer.resume_from_checkpoint and self.trainer.sanity_checking:
                cur_epoch = -1

            # 1. loss_scalars: dict of list, on cpu
            _loss_scalars = [o['loss_scalars'] for o in outputs]
            loss_scalars = {k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars])) for k in _loss_scalars[0]}
            
            _figures = [o['figures'] for o in outputs]
            figures = [item for sublist in _figures for item in sublist]

            # tensorboard records only on rank 0
            if self.trainer.global_rank == 0:
                for k, v in loss_scalars.items():
                    mean_v = torch.stack(v).mean()
                    self.logger[0].experiment.add_scalar(f'val_{valset_idx}/avg_{k}', mean_v, global_step=cur_epoch)
                    if self.config.TRAINER.USE_WANDB:
                        self.logger[1].log_metrics({f'val_{valset_idx}/avg_{k}': mean_v}, cur_epoch)

                for plot_idx, fig in enumerate(figures):
                    self.logger[0].experiment.add_figure(
                        f'val_mae_{valset_idx}/pair-{plot_idx}', fig, cur_epoch, close=True)

            plt.close('all')

# class PL_XoFTR_Pretrain(pl.LightningModule):
#     def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
#         """
#         TODO:
#             - use the new version of PL logging API.
#         """
#         super().__init__()
#         # Misc
#         self.config = config  # full config
        
#         _config = lower_config(self.config)
#         self.xoftr_cfg = lower_config(_config['xoftr'])
#         self.profiler = profiler or PassThroughProfiler()
#         self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)

#         # generator to create the same masks for validation
#         self.val_seed = self.config.PRETRAIN.VAL_SEED
#         self.val_generator = torch.Generator(device="cuda").manual_seed(self.val_seed)
#         self.mae_margins = config.PRETRAIN.MAE_MARGINS

#         # Matcher: XoFTR
#         self.matcher = XoFTR_Pretrain_Light(config=_config['xoftr'])
#         self.loss = XoFTRLossPretrain(_config)

#         # Pretrained weights
#         if pretrained_ckpt:
#             state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
#             self.matcher.load_state_dict(state_dict, strict=False)
#             logger.info(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")
        
#         # Testing
#         self.dump_dir = dump_dir
        
#     def configure_optimizers(self):
#         # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
#         optimizer = build_optimizer(self, self.config)
#         scheduler = build_scheduler(self.config, optimizer)
#         return [optimizer], [scheduler]
    
#     def optimizer_step(
#             self, epoch, batch_idx, optimizer, optimizer_idx,
#             optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
#         # learning rate warm up
#         warmup_step = self.config.TRAINER.WARMUP_STEP
#         if self.trainer.global_step < warmup_step:
#             if self.config.TRAINER.WARMUP_TYPE == 'linear':
#                 base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
#                 lr = base_lr + \
#                     (self.trainer.global_step / self.config.TRAINER.WARMUP_STEP) * \
#                     abs(self.config.TRAINER.TRUE_LR - base_lr)
#                 for pg in optimizer.param_groups:
#                     pg['lr'] = lr
#             elif self.config.TRAINER.WARMUP_TYPE == 'constant':
#                 pass
#             else:
#                 raise ValueError(f'Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}')

#         # update params
#         optimizer.step(closure=optimizer_closure)
#         optimizer.zero_grad()
    
#     def _trainval_inference(self, batch, generator=None):
#         # 添加调试信息
#         # print(f"=== DEBUG: Before generate_random_masks ===")
#         # print(f"image0 shape: {batch['image0'].shape}")
#         # print(f"image1 shape: {batch['image1'].shape}")
        
#         generate_random_masks(batch,
#                         patch_size=self.config.PRETRAIN.PATCH_SIZE,
#                         mask_ratio=self.config.PRETRAIN.MASK_RATIO,
#                         generator=generator,
#                         margins=self.mae_margins)
        
#         # print(f"=== DEBUG: After generate_random_masks ===")
#         # print(f"image0 shape: {batch['image0'].shape}")
#         # print(f"image1 shape: {batch['image1'].shape}")
#         # print(f"mae_mask0 shape: {batch['mae_mask0'].shape if 'mae_mask0' in batch else 'N/A'}")
        
#         # 检查并修复通道数问题
#         self._fix_channel_issues(batch)
        
#         with self.profiler.profile("XoFTR"):
#             self.matcher(batch)
            
#         with self.profiler.profile("Compute losses"):
#             # Create target pacthes to reconstruct
#             get_target(batch)
#             self.loss(batch)
    
#     def _fix_channel_issues(self, batch):
#         """修复输入数据的通道数问题"""
#         # 检查image0的通道数
#         if 'image0' in batch and batch['image0'].dim() == 4:
#             channels = batch['image0'].shape[1]
#             if channels != 1:
#                 print(f"WARNING: image0 has {channels} channels, fixing to 1 channel")
#                 if channels == 3:
#                     # RGB转灰度
#                     batch['image0'] = 0.299 * batch['image0'][:, 0:1, :, :] + \
#                                       0.587 * batch['image0'][:, 1:2, :, :] + \
#                                       0.114 * batch['image0'][:, 2:3, :, :]
#                 elif channels == 2:
#                     # 2通道，取第一个通道
#                     batch['image0'] = batch['image0'][:, 0:1, :, :]
#                 else:
#                     # 其他情况，取平均
#                     batch['image0'] = batch['image0'].mean(dim=1, keepdim=True)
        
#         # 检查image1的通道数
#         if 'image1' in batch and batch['image1'].dim() == 4:
#             channels = batch['image1'].shape[1]
#             if channels != 1:
#                 print(f"WARNING: image1 has {channels} channels, fixing to 1 channel")
#                 if channels == 3:
#                     # RGB转灰度
#                     batch['image1'] = 0.299 * batch['image1'][:, 0:1, :, :] + \
#                                       0.587 * batch['image1'][:, 1:2, :, :] + \
#                                       0.114 * batch['image1'][:, 2:3, :, :]
#                 elif channels == 2:
#                     # 2通道，取第一个通道
#                     batch['image1'] = batch['image1'][:, 0:1, :, :]
#                 else:
#                     # 其他情况，取平均
#                     batch['image1'] = batch['image1'].mean(dim=1, keepdim=True)
        
#         print(f"After fix - image0 shape: {batch['image0'].shape}, image1 shape: {batch['image1'].shape}")
    
#     def training_step(self, batch, batch_idx):
#         self._trainval_inference(batch)
        
#         # logging
#         if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
#             # scalars
#             for k, v in batch['loss_scalars'].items():
#                 self.logger[0].experiment.add_scalar(f'train/{k}', v, self.global_step)
#                 if self.config.TRAINER.USE_WANDB:
#                     self.logger[1].log_metrics({f'train/{k}': v}, self.global_step)
            
#             if self.config.TRAINER.ENABLE_PLOTTING:
#                 figures = make_mae_figures(batch)
#                 for i, figure in enumerate(figures):
#                     self.logger[0].experiment.add_figure(
#                     f'train_mae/node_{self.trainer.global_rank}-device_{self.device.index}-batch_{i}',
#                     figure, self.global_step)

#         return {'loss': batch['loss']}

#     def training_epoch_end(self, outputs):
#         avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
#         if self.trainer.global_rank == 0:
#             self.logger[0].experiment.add_scalar(
#                 'train/avg_loss_on_epoch', avg_loss,
#                 global_step=self.current_epoch)
#             if self.config.TRAINER.USE_WANDB:
#                 self.logger[1].log_metrics(
#                     {'train/avg_loss_on_epoch': avg_loss},
#                     self.current_epoch)
    
#     def validation_step(self, batch, batch_idx):
#         self._trainval_inference(batch, self.val_generator)
    
#         val_plot_interval = max(self.trainer.num_val_batches[0] // \
#                                  (self.trainer.num_gpus * self.n_vals_plot), 1)
#         figures = []
#         if batch_idx % val_plot_interval == 0:
#             figures = make_mae_figures(batch)

#         return {
#             'loss_scalars': batch['loss_scalars'],
#             'figures': figures,
#         }
        
#     def validation_epoch_end(self, outputs):
#         self.val_generator.manual_seed(self.val_seed)
#         # handle multiple validation sets
#         multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        
#         for valset_idx, outputs in enumerate(multi_outputs):
#             # since pl performs sanity_check at the very begining of the training
#             cur_epoch = self.trainer.current_epoch
#             if not self.trainer.resume_from_checkpoint and self.trainer.running_sanity_check:
#                 cur_epoch = -1

#             # 1. loss_scalars: dict of list, on cpu
#             _loss_scalars = [o['loss_scalars'] for o in outputs]
#             loss_scalars = {k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars])) for k in _loss_scalars[0]}
            
#             _figures = [o['figures'] for o in outputs]
#             figures = [item for sublist in _figures for item in sublist]

#             # tensorboard records only on rank 0
#             if self.trainer.global_rank == 0:
#                 for k, v in loss_scalars.items():
#                     mean_v = torch.stack(v).mean()
#                     self.logger[0].experiment.add_scalar(f'val_{valset_idx}/avg_{k}', mean_v, global_step=cur_epoch)
#                     if self.config.TRAINER.USE_WANDB:
#                         self.logger[1].log_metrics({f'val_{valset_idx}/avg_{k}': mean_v}, cur_epoch)

#                 for plot_idx, fig in enumerate(figures):
#                     self.logger[0].experiment.add_figure(
#                         f'val_mae_{valset_idx}/pair-{plot_idx}', fig, cur_epoch, close=True)

#             plt.close('all')