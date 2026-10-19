import torch
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR


# def build_optimizer(model, config):
#     name = config.TRAINER.OPTIMIZER
#     lr = config.TRAINER.TRUE_LR
#
#     if name == "adam":
#         return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.TRAINER.ADAM_DECAY)
#     elif name == "adamw":
#         return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.TRAINER.ADAMW_DECAY)
#     else:
#         raise ValueError(f"TRAINER.OPTIMIZER = {name} is not a valid optimizer!")

####################################20260417##################################
def build_optimizer(model, config):
    name = config.TRAINER.OPTIMIZER
    lr = config.TRAINER.TRUE_LR

    # # ================= 【涨点核心：分层学习率分组】 =================
    # base_params = []
    # saliency_params = []
    #
    # for param_name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         continue
    #
    #     # 只要名字里包含 saliency_head，就归入高薪组
    #     if "saliency_head" in param_name:
    #         saliency_params.append(param)
    #     else:
    #         base_params.append(param)
    #
    # # 构建参数组列表
    # # 第一组：主干网络，使用外层传进来的默认 lr，求稳
    # # 第二组：掩码头，强行把 lr 乘以 10.0，快速拉低 Mask 均值
    # param_groups = [
    #     {'params': base_params, 'lr': lr},
    #     {'params': saliency_params, 'lr': lr * 10.0}
    # ]
    # # ==============================================================

    if name == "adam":
        # 注意：这里把 model.parameters() 换成了 param_groups
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.TRAINER.ADAM_DECAY)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.TRAINER.ADAMW_DECAY)
    else:
        raise ValueError(f"TRAINER.OPTIMIZER = {name} is not a valid optimizer!")

def build_scheduler(config, optimizer):
    """
    Returns:
        scheduler (dict):{
            'scheduler': lr_scheduler,
            'interval': 'step',  # or 'epoch'
            'monitor': 'val_f1', (optional)
            'frequency': x, (optional)
        }
    """
    scheduler = {'interval': config.TRAINER.SCHEDULER_INTERVAL}
    name = config.TRAINER.SCHEDULER

    if name == 'MultiStepLR':
        scheduler.update(
            {'scheduler': MultiStepLR(optimizer, config.TRAINER.MSLR_MILESTONES, gamma=config.TRAINER.MSLR_GAMMA)})
    elif name == 'CosineAnnealing':
        scheduler.update(
            {'scheduler': CosineAnnealingLR(optimizer, config.TRAINER.COSA_TMAX)})
    elif name == 'ExponentialLR':
        scheduler.update(
            {'scheduler': ExponentialLR(optimizer, config.TRAINER.ELR_GAMMA)})
    else:
        raise NotImplementedError()

    return scheduler
