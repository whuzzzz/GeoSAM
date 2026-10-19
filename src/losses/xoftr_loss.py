from loguru import logger

import torch
import torch.nn as nn
from kornia.geometry.conversions import convert_points_to_homogeneous
from kornia.geometry.epipolar import numeric

class XoFTRLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace
        self.loss_config = config['xoftr']['loss']
        self.pos_w = self.loss_config['pos_weight']
        self.neg_w = self.loss_config['neg_weight']


    def compute_fine_matching_loss(self, data):
        """ Point-wise Focal Loss with 0 / 1 confidence as gt.
        Args:
        data (dict): {
            conf_matrix_fine (torch.Tensor): (N, W_f^2, W_f^2) 
            conf_matrix_f_gt (torch.Tensor): (N, W_f^2, W_f^2) 
            }
        """
        conf_matrix_fine = data['conf_matrix_fine']
        conf_matrix_f_gt = data['conf_matrix_f_gt']
        pos_mask, neg_mask = conf_matrix_f_gt > 0, conf_matrix_f_gt == 0
        pos_w, neg_w = self.pos_w, self.neg_w

        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            neg_w = 0.

        conf_matrix_fine = torch.clamp(conf_matrix_fine, 1e-6, 1-1e-6)
        alpha = self.loss_config['focal_alpha']
        gamma = self.loss_config['focal_gamma']

        loss_pos = - alpha * torch.pow(1 - conf_matrix_fine[pos_mask], gamma) * (conf_matrix_fine[pos_mask]).log()
        # loss_pos *= conf_matrix_f_gt[pos_mask]
        loss_neg = - alpha * torch.pow(conf_matrix_fine[neg_mask], gamma) * (1 - conf_matrix_fine[neg_mask]).log()

        return pos_w * loss_pos.mean() + neg_w * loss_neg.mean()
    
    def _symmetric_epipolar_distance(self, pts0, pts1, E, K0, K1):
            """Squared symmetric epipolar distance.
            This can be seen as a biased estimation of the reprojection error.
            Args:
                pts0 (torch.Tensor): [N, 2]
                E (torch.Tensor): [3, 3]
            """
            pts0 = (pts0 - K0[:, [0, 1], [2, 2]]) / K0[:, [0, 1], [0, 1]]
            pts1 = (pts1 - K1[:, [0, 1], [2, 2]]) / K1[:, [0, 1], [0, 1]]
            pts0 = convert_points_to_homogeneous(pts0)
            pts1 = convert_points_to_homogeneous(pts1)

            Ep0 = (pts0[:,None,:] @ E.transpose(-2,-1)).squeeze(1)  # [N, 3]
            p1Ep0 = torch.sum(pts1 * Ep0, -1)  # [N,]
            Etp1 = (pts1[:,None,:] @ E).squeeze(1)  # [N, 3]

            d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2 + 1e-9) + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2 + 1e-9))  # N
            return d
    
    def compute_sub_pixel_loss(self, data):
        """ symmetric epipolar distance loss.
        Args:
        data (dict): {
            m_bids (torch.Tensor): (N)
            T_0to1 (torch.Tensor): (B, 4, 4)
            mkpts0_f_train (torch.Tensor): (N, 2) 
            mkpts1_f_train (torch.Tensor): (N, 2) 
            }
        """

        Tx = numeric.cross_product_matrix(data['T_0to1'][:, :3, 3])
        E_mat = Tx @ data['T_0to1'][:, :3, :3]

        m_bids = data['m_bids']
        pts0 = data['mkpts0_f_train']
        pts1 = data['mkpts1_f_train']
        
        sym_dist = self._symmetric_epipolar_distance(pts0, pts1, E_mat[m_bids], data['K0'][m_bids], data['K1'][m_bids])
        # filter matches with high epipolar error (only train approximately correct fine-level matches)
        loss = sym_dist[sym_dist<1e-4]
        if len(loss) == 0:
            return torch.zeros(1, device=loss.device, requires_grad=False)[0]
        return loss.mean()
    
    def compute_coarse_loss(self, data, weight=None):
        """ Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
        data (dict): {
            conf_matrix_0_to_1 (torch.Tensor): (N, HW0, HW1) 
            conf_matrix_1_to_0 (torch.Tensor): (N, HW0, HW1) 
            conf_gt (torch.Tensor): (N, HW0, HW1)
            }
            weight (torch.Tensor): (N, HW0, HW1)
        """

        conf_matrix_0_to_1 = data["conf_matrix_0_to_1"]
        conf_matrix_1_to_0 = data["conf_matrix_1_to_0"]
        conf_gt = data["conf_matrix_gt"]

        pos_mask = conf_gt == 1
        c_pos_w = self.pos_w
        # corner case: no gt coarse-level match at all
        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_pos_w = 0.

        conf_matrix_0_to_1 = torch.clamp(conf_matrix_0_to_1, 1e-6, 1-1e-6)
        conf_matrix_1_to_0 = torch.clamp(conf_matrix_1_to_0, 1e-6, 1-1e-6)
        alpha = self.loss_config['focal_alpha']
        gamma = self.loss_config['focal_gamma']
        
        loss_pos = - alpha * torch.pow(1 - conf_matrix_0_to_1[pos_mask], gamma) * (conf_matrix_0_to_1[pos_mask]).log()
        loss_pos += - alpha * torch.pow(1 - conf_matrix_1_to_0[pos_mask], gamma) * (conf_matrix_1_to_0[pos_mask]).log()
        if weight is not None:
            loss_pos = loss_pos * weight[pos_mask]
        
        loss_c = (c_pos_w * loss_pos.mean())

        return loss_c
                
    @torch.no_grad()
    def compute_c_weight(self, data):
        """ compute element-wise weights for computing coarse-level loss. """
        if 'mask0' in data:
            c_weight = (data['mask0'].flatten(-2)[..., None] * data['mask1'].flatten(-2)[:, None]).float()
        else:
            c_weight = None
        return c_weight

    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}
        # 0. compute element-wise loss weight
        c_weight = self.compute_c_weight(data)

        # 1. coarse-level loss
        loss_c = self.compute_coarse_loss(data, weight=c_weight)
        loss_c *= self.loss_config['coarse_weight'] 
        loss = loss_c 
        loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})

        # 2. fine-level matching loss for windows
        loss_f_match = self.compute_fine_matching_loss(data)
        loss_f_match *= self.loss_config['fine_weight'] 
        loss = loss + loss_f_match 
        loss_scalars.update({"loss_f":  loss_f_match.clone().detach().cpu()})

        # 3. sub-pixel refinement loss 
        loss_sub = self.compute_sub_pixel_loss(data)
        loss_sub *= self.loss_config['sub_weight'] 
        loss = loss + loss_sub 
        loss_scalars.update({"loss_sub":  loss_sub.clone().detach().cpu()})

        # # ================= [新增底牌：掩码稀疏性正则化 L1 Loss] =================
        # # 核心逻辑：直接把 Mask 的均值作为 Loss 的一部分！
        # # Mask 均值越高（比如 0.9），这部分的 Loss 就越大。
        # # 为了把总 Loss 降下来，网络会被迫去削减背景区域的权重。
        # # ================= [升级底牌：带阈值的稀疏性正则化 (Hinge Sparsity)] =================
        # if 'mask_8_0' in data and 'mask_8_1' in data:
        #     # 1. 计算当前的 Mask 均值
        #     current_mean = (data['mask_8_0'].mean() + data['mask_8_1'].mean()) / 2.0
        #
        #     # 2. 设置期望的底线 (Margin)，比如 0.25
        #     # 意味着我们认为一张图里大概有 25% 的区域是有效的前景角点
        #     target_margin = 0.25
        #
        #     # 3. 只有当均值大于 0.25 时，才产生惩罚 Loss！一旦低于 0.25，惩罚为 0
        #     # 使用 torch.clamp 把负数截断为 0
        #     loss_sparsity = torch.clamp(current_mean - target_margin, min=0.0)
        #
        #     # 4. 把惩罚力度降低 10 倍，让它只起“引导”作用，不反客为主
        #     lambda_sparsity = 0.05
        #     loss_sparsity = loss_sparsity * lambda_sparsity
        #
        #     loss = loss + loss_sparsity
        #     loss_scalars.update({"loss_sparsity": loss_sparsity.clone().detach().cpu()})
        # # ====================================================================================


        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})
