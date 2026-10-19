# Run once
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
os.chdir("..")
import torch
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure
from src.xoftr import XoFTR, XoFTR_Light_convnextv2, XoFTR_Light, XoFTR_dino_resnet_panmatch_light_withfpn
from src.config.default import get_cfg_defaults
from src.utils.data_io import DataIOWrapper, lower_config
import time

# Get default configurations
config = get_cfg_defaults(inference=True)
config = lower_config(config)

# Coarse level threshold
config['xoftr']['match_coarse']['thr'] = 0.3 # Default 0.3

# Fine level threshold
config['xoftr']['fine']['thr'] = 0.1 # Default 0.1

# It is posseble to get denser matches
# If True, xoftr returns all fine-level matches for each fine-level window (at 1/2 resolution)
config['xoftr']['fine']['denser'] = False # Default False

# XoFTR model
matcher = XoFTR_dino_resnet_panmatch_light_withfpn(config=config["xoftr"])
total_params = sum(p.numel() for p in matcher.parameters())
print(f"总参数数量: {total_params}")

# The input image sizes for xoftr
# Note: The output matches and output images are in original image size
config['test']['img0_resize'] = 640 # resize the longer side, None for no resize
config['test']['img1_resize'] = 640 # resize the longer side, None for no resize

# The path for weights
ckpt = './logs/tb_logs/train20260128dinov3_vitb_resnet_panmatch_light/version_1没有预训练/checkpoints/epoch=7-auc@5=0.167-auc@10=0.308-auc@20=0.468.ckpt'
# ckpt = "E:/XoFTR-main/logs/tb_logs/megadepth/version_11/checkpoints/last.ckpt"
# ckpt = "./XoFTR-main-more/logs/tb_logs/xoftr/version_8_初始/checkpoints/epoch=26-auc@5=0.179-auc@10=0.336-auc@20=0.495.ckpt"
# ckpt = "./xoftr/XoFTR-main/weights/weights_xoftr_640.ckpt"
# ckpt = './logs/tb_logs/train20260104coarsematch_convnextv2_pico/version_0/checkpoints/epoch=16-auc@5=0.181-auc@10=0.336-auc@20=0.498.ckpt'
# ckpt = './logs/tb_logs/train20251213onlycoarsematching/version_0只更换mamba的结果/checkpoints/epoch=9-auc@5=0.212-auc@10=0.369-auc@20=0.519.ckpt'

# Data I/O wrapper
matcher = DataIOWrapper(matcher, config=config["test"], ckpt=ckpt)
# Paths for example images
img0_pth = './data/MSRS/infrared/00001D.png'
img1_pth = './data/MSRS/visible/00001D.png'

# img0_pth = "C:/Users/wpc20/Desktop/DSC02806.JPG"
# img1_pth = "C:/Users/wpc20/Desktop/DSC07340.JPG"

# Load and match images
# Note: images are converted to grayscale before matching
t1 = time.time()
output_data = matcher.from_paths(img0_pth, img1_pth, read_color=True)
print(f"运行时间: {time.time() - t1}")

# Matched keypoints
mkpts0 = output_data['mkpts0']
mkpts1 = output_data['mkpts1']

# Confidence values for fine-level matching
mconf = output_data['mconf']

# Original images BGR or GRAY
img0 = output_data['img0']
img1 = output_data['img1']

# Mask outliers using RANSAC (Homography or Fundamental Matrix)

inlier_method = 'H' # F: Fundamental Matrix, H: Homography

# RANSAC types: https://opencv.org/blog/evaluating-opencvs-new-ransacs/

if inlier_method == 'F':
    F, inlier_mask = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, ransacReprojThreshold=1, maxIters=10000, confidence=0.9999)
elif inlier_method == 'H':
    H_pred, inlier_mask = cv2.findHomography(mkpts0, mkpts1, cv2.USAC_MAGSAC, ransacReprojThreshold=1, maxIters=10000, confidence=0.9999)

inlier_mask = inlier_mask.ravel() > 0
mkpts0 = mkpts0[inlier_mask]
mkpts1 = mkpts1[inlier_mask]
mconf = mconf[inlier_mask]

# Draw
color = cm.jet(mconf)
text = [
    'XoFTR',
    'Matches: {}'.format(len(mconf)),
]
if len(img0.shape) == 3:
    _img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
else:
    _img0 = img0
if len(img1.shape) == 3:
    _img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
else:
    _img1 = img1
# fig_org = make_matching_figure(_img0, _img1, np.zeros(0),  np.zeros(0),  np.zeros(0), text=["Original"], dpi=125)
# fig_match = make_matching_figure(_img0, _img1, mkpts0, mkpts1, color, text=text, dpi=125, path=path)