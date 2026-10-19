import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


# =========================
# utils
# =========================
def resize_to(x, target):
    return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def norm(x):
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x


def to_heatmap(x):
    x = norm(x.detach().cpu().float().numpy())
    x = (x * 255).astype(np.uint8)
    return cv2.applyColorMap(x, cv2.COLORMAP_JET)


def save(img, path):
    cv2.imwrite(path, img)


# =========================
# 1. GGSA Visualization
# =========================

def visualize_ggsa(
    cnn_feat,
    dino_feat,
    ggsa_feat,
    save_dir,
    name="ggsa"
):
    """
    CNN vs DINO vs naive fusion vs GGSA
    """

    ensure_dir(save_dir)

    # CNN
    cnn_map = cnn_feat[0].mean(0)
    cnn_heat = to_heatmap(cnn_map)

    # DINO
    dino_map = dino_feat[0].mean(0)
    dino_heat = to_heatmap(dino_map)

    # naive concat
    # naive = torch.cat([cnn_feat, dino_feat], dim=1).mean(1)[0]
    cnn_feat = cnn_feat
    dino_feat = F.interpolate(
        dino_feat,
        size=cnn_feat.shape[-2:],
        mode="bilinear",
        align_corners=False
    )

    naive = torch.cat([cnn_feat, dino_feat], dim=1).mean(1)[0]
    naive_heat = to_heatmap(naive)

    # GGSA output
    ggsa_map = ggsa_feat[0].mean(0)
    ggsa_heat = to_heatmap(ggsa_map)

    cv2.imwrite(f"{save_dir}/{name}_cnn.png", cnn_heat)
    cv2.imwrite(f"{save_dir}/{name}_dino.png", dino_heat)
    cv2.imwrite(f"{save_dir}/{name}_naive.png", naive_heat)
    cv2.imwrite(f"{save_dir}/{name}_ggsa.png", ggsa_heat)

    print("[GGSA VIS] saved")


# =========================
# 2. SASSM Visualization
# =========================

def visualize_sassm(
    feature,
    mask,
    alpha=2.0,
    save_dir="./sassm",
    name="sassm"
):
    """
    feature: [1,C,H,W]
    mask: [1,1,H,W]
    """

    ensure_dir(save_dir)

    feat = feature[0].mean(0)
    mask = mask[0, 0]

    # raw
    raw = to_heatmap(feat)

    # mask
    mask_hm = to_heatmap(mask)

    # gated
    gated = feat * (1 + alpha * mask)
    gated_hm = to_heatmap(gated)

    # simulating mamba output (nonlinear smoothing)
    smooth = torch.nn.functional.avg_pool2d(feature, 3, 1, 1)[0].mean(0)
    smooth_hm = to_heatmap(smooth)

    cv2.imwrite(f"{save_dir}/{name}_raw.png", raw)
    cv2.imwrite(f"{save_dir}/{name}_mask.png", mask_hm)
    cv2.imwrite(f"{save_dir}/{name}_gated.png", gated_hm)
    cv2.imwrite(f"{save_dir}/{name}_mamba_out.png", smooth_hm)

    print("[SASSM VIS] saved")


# =========================
# 3. Mamba scan simulation
# =========================

def visualize_mamba_scan(feature, save_dir="./mamba", name="mamba"):
    """
    模拟 scan + accumulation
    """

    ensure_dir(save_dir)

    feat = feature[0].mean(0)
    H, W = feat.shape

    flat = feat.flatten()

    # accumulation (state recursion simulation)
    state = torch.cumsum(flat, dim=0)
    state = state.reshape(H, W)

    acc_map = to_heatmap(state)
    raw_map = to_heatmap(feat)

    cv2.imwrite(f"{save_dir}/{name}_raw.png", raw_map)
    cv2.imwrite(f"{save_dir}/{name}_accumulation.png", acc_map)

    # scan path visualization (zigzag)
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    step = W // 20
    for i in range(H):
        if i % 2 == 0:
            for j in range(W - 1):
                cv2.line(canvas, (j, i), (j + 1, i), (0, 255, 0), 1)
        else:
            for j in range(W - 1):
                cv2.line(canvas, (W - j - 1, i), (W - j - 2, i), (0, 255, 0), 1)

    cv2.imwrite(f"{save_dir}/{name}_scan.png", canvas)

    print("[MAMBA VIS] saved")


# =========================
# 4. MAIN API (hook output)
# =========================

def run_visualization(hook_mgr, img_dir="./vis_out"):
    """
    hook_mgr.features = 你现有代码的输出
    """

    ensure_dir(img_dir)
    f = hook_mgr.features

    # ===== GGSA =====
    if "cnn_guides" in f and "dino_proj_l4" in f and "aligned_1_8" in f:
        visualize_ggsa(
            cnn_feat=f["cnn_guides"][2],      # 1/8
            dino_feat=f["dino_proj_l4"],
            ggsa_feat=f["aligned_1_8"],
            save_dir=img_dir
        )

    # ===== SASSM =====
    if "backbone_out" in f:
        feat_c, feat_m, feat_f, mask = f["backbone_out"]

        visualize_sassm(
            feature=feat_c,
            mask=mask,
            save_dir=img_dir
        )

        visualize_mamba_scan(
            feature=feat_c,
            save_dir=img_dir
        )


# =========================
# 5. USAGE
# =========================
import json
import os
import re
import time
from pathlib import Path
# Set CUDA_VISIBLE_DEVICES in the shell when selecting GPUs.

import cv2
import torch
import numpy as np
import torch.nn.functional as F

from src.xoftr import (
    XoFTR,
    XoFTR_dino_resnet_panmatch_light_withfpn,
    XoFTR_Light_convnextv2,
    XoFTR_Light,
)
from src.config.default import get_cfg_defaults
from src.utils.data_io import DataIOWrapper, lower_config


# ============================================================
# 1. 路径配置：你主要改这里
# ============================================================

L_FOLDER = "./data/METU_VisTIR/sunny/scene_1/visible/images"
R_FOLDER = "./data/METU_VisTIR/sunny/scene_1/thermal/images"

# 匹配结果保存路径
MATCH_SAVE_DIR = "./without_mask_test/METU_VisTIR_sunny1-geosam-match"

# 中间特征图保存路径
VIS_SAVE_DIR = "./without_mask_test/METU_VisTIR_sunny1-geosam-feature-vis"

# 你的 GeoSAM / XoFTR_dino_resnet_panmatch_light_withfpn 权重
CKPT_PATH = "./logs/tb_logs/dinomatch/version_36_wiithout_mask/checkpoints/epoch=14-auc@5=0.208-auc@10=0.372-auc@20=0.546.ckpt"

# 是否只保存前几对图像的中间特征图，避免生成太多图片
SAVE_VIS = True
SAVE_VIS_NUM = 5

# feature map 可视化方式：
# "norm" 最稳定，适合 CNN / FPN / AG-RoPE；
# "pca" 更适合 DINO 语义特征，但稍慢。
DINO_VIS_MODE = "pca"
DEFAULT_VIS_MODE = "norm"

# 输入图像长边 resize
IMG0_RESIZE = 640
IMG1_RESIZE = 640


# ============================================================
# 2. 基础工具函数
# ============================================================

def natural_sort(file_list):
    return sorted(
        file_list,
        key=lambda x: [int(text) if text.isdigit() else text.lower()
                       for text in re.split(r"([0-9]+)", x)]
    )


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_to_uint8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.nanmin(x)
    x = x / (np.nanmax(x) + 1e-8)
    return (x * 255).clip(0, 255).astype(np.uint8)


def read_image_bgr(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def resize_long_edge(img, target_long_edge=640):
    if target_long_edge is None:
        return img
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge == target_long_edge:
        return img
    scale = target_long_edge / long_edge
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def overlay_heatmap_on_image(img_bgr, heatmap_bgr, alpha=0.45):
    if heatmap_bgr.shape[:2] != img_bgr.shape[:2]:
        heatmap_bgr = cv2.resize(
            heatmap_bgr,
            (img_bgr.shape[1], img_bgr.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
    return cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0)


def tensor_to_feature_map(feat: torch.Tensor, mode="norm") -> np.ndarray:
    """
    将 [C,H,W] 或 [1,C,H,W] tensor 转成单通道 feature map。
    mode:
        norm: 通道 L2 范数
        mean: 通道均值
        max: 通道最大值
        pca: 第一主成分，适合 DINO 语义特征
    """
    if feat is None:
        raise ValueError("feat is None")

    if isinstance(feat, (list, tuple)):
        feat = feat[0]

    if feat.dim() == 4:
        feat = feat[0]

    feat = feat.detach().float().cpu()

    if feat.dim() != 3:
        raise ValueError(f"Expected [C,H,W], got {tuple(feat.shape)}")

    if mode == "norm":
        fmap = torch.norm(feat, p=2, dim=0)
    elif mode == "mean":
        fmap = feat.mean(dim=0)
    elif mode == "max":
        fmap = feat.max(dim=0)[0]
    elif mode == "pca":
        c, h, w = feat.shape
        x = feat.reshape(c, -1).T  # [HW, C]
        x = x - x.mean(dim=0, keepdim=True)
        # 对流程图可视化够用；如果显存/内存紧张，把 pca 改成 norm
        try:
            _, _, v = torch.pca_lowrank(x, q=1)
            fmap = (x @ v[:, :1]).reshape(h, w)
        except Exception:
            fmap = torch.norm(feat, p=2, dim=0)
    else:
        raise ValueError(f"Unknown visualization mode: {mode}")

    return fmap.numpy()


def feature_to_heatmap(feat: torch.Tensor, out_size=None, mode="norm") -> np.ndarray:
    fmap = tensor_to_feature_map(feat, mode=mode)
    fmap = normalize_to_uint8(fmap)

    if out_size is not None:
        fmap = cv2.resize(fmap, out_size, interpolation=cv2.INTER_CUBIC)

    heatmap = cv2.applyColorMap(fmap, cv2.COLORMAP_JET)
    return heatmap


def mask_to_heatmap(mask: torch.Tensor, out_size=None) -> np.ndarray:
    """
    mask: [B,1,H,W] / [1,H,W] / [H,W]
    """
    if isinstance(mask, (list, tuple)):
        mask = mask[0]

    mask = mask.detach().float().cpu()

    if mask.dim() == 4:
        mask = mask[0, 0]
    elif mask.dim() == 3:
        mask = mask[0]
    elif mask.dim() == 2:
        pass
    else:
        raise ValueError(f"Unexpected mask shape: {tuple(mask.shape)}")

    arr = normalize_to_uint8(mask.numpy())

    if out_size is not None:
        arr = cv2.resize(arr, out_size, interpolation=cv2.INTER_CUBIC)

    heatmap = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
    return heatmap


# def save_feature_overlay(save_dir, name, feat, img_bgr, mode="norm"):
#     """
#     保存 heatmap 和 overlay 两张图。
#     """
#     ensure_dir(save_dir)
#
#     out_size = (img_bgr.shape[1], img_bgr.shape[0])
#     heat = feature_to_heatmap(feat, out_size=out_size, mode=mode)
#     overlay = overlay_heatmap_on_image(img_bgr, heat, alpha=0.45)
#
#     cv2.imwrite(os.path.join(save_dir, f"{name}_heatmap.png"), heat)
#     cv2.imwrite(os.path.join(save_dir, f"{name}_overlay.png"), overlay)

def save_feature_overlay(save_dir, name, feat, img_bgr, mode="norm", img_long_edge=640):
    """
    保存 feature heatmap 和 overlay。
    自动裁掉 padding 区域。
    """
    ensure_dir(save_dir)

    valid_h, valid_w = get_resized_shape_by_long_edge(img_bgr, img_long_edge)

    # 用 resize 后的真实图像作为 overlay 背景
    img_vis = cv2.resize(img_bgr, (valid_w, valid_h), interpolation=cv2.INTER_LINEAR)

    # 裁掉 feature map 里的 padding 区域
    feat = crop_padding_from_feature(
        feat,
        valid_img_shape=(valid_h, valid_w),
        assume_square_padding=True,
    )

    out_size = (valid_w, valid_h)
    heat = feature_to_heatmap(feat, out_size=out_size, mode=mode)
    overlay = overlay_heatmap_on_image(img_vis, heat, alpha=0.45)

    cv2.imwrite(os.path.join(save_dir, f"{name}_heatmap.png"), heat)
    cv2.imwrite(os.path.join(save_dir, f"{name}_overlay.png"), overlay)



# def save_mask_overlay(save_dir, name, mask, img_bgr):
#     ensure_dir(save_dir)
#
#     out_size = (img_bgr.shape[1], img_bgr.shape[0])
#     heat = mask_to_heatmap(mask, out_size=out_size)
#     overlay = overlay_heatmap_on_image(img_bgr, heat, alpha=0.45)
#
#     cv2.imwrite(os.path.join(save_dir, f"{name}_heatmap.png"), heat)
#     cv2.imwrite(os.path.join(save_dir, f"{name}_overlay.png"), overlay)

def save_mask_overlay(save_dir, name, mask, img_bgr, img_long_edge=640):
    """
    保存 saliency mask heatmap 和 overlay。
    自动裁掉 padding 区域。
    """
    ensure_dir(save_dir)

    valid_h, valid_w = get_resized_shape_by_long_edge(img_bgr, img_long_edge)

    img_vis = cv2.resize(img_bgr, (valid_w, valid_h), interpolation=cv2.INTER_LINEAR)

    mask = crop_padding_from_feature(
        mask,
        valid_img_shape=(valid_h, valid_w),
        assume_square_padding=True,
    )

    out_size = (valid_w, valid_h)
    heat = mask_to_heatmap(mask, out_size=out_size)
    overlay = overlay_heatmap_on_image(img_vis, heat, alpha=0.45)

    cv2.imwrite(os.path.join(save_dir, f"{name}_heatmap.png"), heat)
    cv2.imwrite(os.path.join(save_dir, f"{name}_overlay.png"), overlay)


# ============================================================
# 3. 匹配结果可视化
# ============================================================
def get_resized_shape_by_long_edge(img_bgr, target_long_edge=640):
    """
    计算 DataIOWrapper 长边 resize 后的有效图像尺寸。
    返回: resized_h, resized_w
    """
    h, w = img_bgr.shape[:2]
    long_edge = max(h, w)

    if target_long_edge is None:
        return h, w

    scale = target_long_edge / long_edge
    resized_w = int(round(w * scale))
    resized_h = int(round(h * scale))
    return resized_h, resized_w


def crop_padding_from_feature(feat, valid_img_shape, assume_square_padding=True):
    """
    从特征图中裁掉 padding 区域。

    feat: [1,C,Hf,Wf] / [C,Hf,Wf] / [1,1,Hf,Wf] / [Hf,Wf]
    valid_img_shape: resize 后真实图像尺寸, (valid_h, valid_w)
    assume_square_padding:
        如果模型把图像 padding 到近似方形，例如 640x640，则设为 True。
    """
    if feat is None:
        return feat

    if isinstance(feat, (list, tuple)):
        return feat

    if not torch.is_tensor(feat):
        return feat

    valid_h, valid_w = valid_img_shape

    # 取特征图空间尺寸
    if feat.dim() == 4:
        _, _, hf, wf = feat.shape
    elif feat.dim() == 3:
        _, hf, wf = feat.shape
    elif feat.dim() == 2:
        hf, wf = feat.shape
    else:
        return feat

    if assume_square_padding:
        # 常见情况：输入被 padding 到 max(valid_h, valid_w) 的方形
        pad_size = max(valid_h, valid_w)

        valid_h_f = int(round(valid_h / pad_size * hf))
        valid_w_f = int(round(valid_w / pad_size * wf))
    else:
        # 如果只是 pad 到整除尺度，通常 padding 不大，可以不裁
        valid_h_f, valid_w_f = hf, wf

    valid_h_f = max(1, min(valid_h_f, hf))
    valid_w_f = max(1, min(valid_w_f, wf))

    if feat.dim() == 4:
        return feat[:, :, :valid_h_f, :valid_w_f]
    elif feat.dim() == 3:
        return feat[:, :valid_h_f, :valid_w_f]
    else:
        return feat[:valid_h_f, :valid_w_f]



def resize_image_and_keypoints(img, keypoints, target_size=(640, 640)):
    """
    将图像强制 resize 到固定尺寸，并同步缩放关键点坐标。
    target_size: (W, H)
    """
    h, w = img.shape[:2]
    target_w, target_h = target_size

    scale_x = target_w / w
    scale_y = target_h / h

    img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    keypoints = np.asarray(keypoints, dtype=np.float32).copy()
    if len(keypoints) > 0:
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y

    return img_resized, keypoints

######左右排列##########################
# def draw_matches(img1_path, img2_path, keypoints1, keypoints2, path, target_size=(640, 640)):
#     """
#     将两张图统一 resize 到 target_size 后再画匹配。
#     注意：这里只影响可视化，不影响 RMSE/PCR 计算。
#     """
#     img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
#     img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)
#
#     if img1 is None or img2 is None:
#         print(f"[WARN] Cannot read images for drawing: {img1_path}, {img2_path}")
#         return None
#
#     keypoints1 = np.asarray(keypoints1, dtype=np.float32)
#     keypoints2 = np.asarray(keypoints2, dtype=np.float32)
#
#     img1, keypoints1 = resize_image_and_keypoints(img1, keypoints1, target_size=target_size)
#     img2, keypoints2 = resize_image_and_keypoints(img2, keypoints2, target_size=target_size)
#
#     height1, width1 = img1.shape[:2]
#     height2, width2 = img2.shape[:2]
#
#     # 两张图已经同尺寸，这里 height1 == height2
#     height = max(height1, height2)
#     width = width1 + width2
#
#     img_combined = np.zeros((height, width, 3), dtype=np.uint8)
#     img_combined[:height1, :width1] = img1
#     img_combined[:height2, width1:width1 + width2] = img2
#
#     if len(keypoints1) == 0:
#         if path:
#             cv2.imwrite(path, img_combined)
#         return img_combined
#
#     keypoints2_vis = keypoints2.copy()
#     keypoints2_vis[:, 0] += width1
#
#     matched_points = np.hstack((keypoints1, keypoints2_vis))
#
#     for i in range(matched_points.shape[0]):
#         pt1 = tuple(map(int, matched_points[i, :2]))
#         pt2 = tuple(map(int, matched_points[i, 2:]))
#
#         cv2.line(img_combined, pt1, pt2, (0, 255, 255), 1)
#         cv2.circle(img_combined, pt1, 5, (0, 0, 255), 1)
#         cv2.drawMarker(
#             img_combined,
#             pt2,
#             (0, 255, 0),
#             markerType=cv2.MARKER_CROSS,
#             markerSize=10,
#             thickness=1,
#         )
#
#     if path:
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         cv2.imwrite(path, img_combined)
#
#     return img_combined



######上下排列##########################
def draw_matches(img1_path, img2_path, keypoints1, keypoints2, path, target_size=(640, 640)):
    """
    将两张图统一 resize 到 target_size 后，上下排列画匹配。
    上面：image0 / optical
    下面：image1 / infrared
    注意：这里只影响可视化，不影响 RMSE/PCR 计算。
    """
    img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)

    if img1 is None or img2 is None:
        print(f"[WARN] Cannot read images for drawing: {img1_path}, {img2_path}")
        return None

    keypoints1 = np.asarray(keypoints1, dtype=np.float32)
    keypoints2 = np.asarray(keypoints2, dtype=np.float32)

    # 统一 resize，并同步缩放关键点
    img1, keypoints1 = resize_image_and_keypoints(img1, keypoints1, target_size=target_size)
    img2, keypoints2 = resize_image_and_keypoints(img2, keypoints2, target_size=target_size)

    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    # 上下拼接：宽度取最大，高度相加
    height = height1 + height2
    width = max(width1, width2)

    img_combined = np.zeros((height, width, 3), dtype=np.uint8)

    # image0 放上面
    img_combined[:height1, :width1] = img1

    # image1 放下面
    img_combined[height1:height1 + height2, :width2] = img2

    if len(keypoints1) == 0:
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, img_combined)
        return img_combined

    # 第二张图在下面，所以只需要给 y 坐标加 height1
    keypoints2_vis = keypoints2.copy()
    keypoints2_vis[:, 1] += height1

    matched_points = np.hstack((keypoints1, keypoints2_vis))

    for i in range(matched_points.shape[0]):
        pt1 = tuple(map(int, matched_points[i, :2]))   # 上方图像点
        pt2 = tuple(map(int, matched_points[i, 2:]))   # 下方图像点

        # 黄色线连接上下图像中的匹配点
        cv2.line(img_combined, pt1, pt2, (0, 255, 255), 1)

        # 上图：红色空心圆
        cv2.circle(img_combined, pt1, 5, (0, 0, 255), 1)

        # 下图：绿色十字
        cv2.drawMarker(
            img_combined,
            pt2,
            (0, 255, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=10,
            thickness=1,
        )

    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, img_combined)

    return img_combined


# def draw_matches(img1_path, img2_path, keypoints1, keypoints2, path):
#     img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
#     img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)
#
#     if img1 is None or img2 is None:
#         print(f"[WARN] Cannot read images for drawing: {img1_path}, {img2_path}")
#         return None
#
#     height1, width1 = img1.shape[:2]
#     height2, width2 = img2.shape[:2]
#     height = max(height1, height2)
#     width = width1 + width2
#
#     img_combined = np.zeros((height, width, 3), dtype=np.uint8)
#     img_combined[:height1, :width1] = img1
#     img_combined[:height2, width1:width1 + width2] = img2
#
#     if len(keypoints1) == 0:
#         cv2.imwrite(path, img_combined)
#         return img_combined
#
#     keypoints1 = np.asarray(keypoints1).copy()
#     keypoints2 = np.asarray(keypoints2).copy()
#     keypoints2[:, 0] += width1
#
#     matched_points = np.hstack((keypoints1, keypoints2))
#
#     for i in range(matched_points.shape[0]):
#         pt1 = tuple(map(int, matched_points[i, :2]))
#         pt2 = tuple(map(int, matched_points[i, 2:]))
#
#         cv2.line(img_combined, pt1, pt2, (0, 255, 255), 1)
#         cv2.circle(img_combined, pt1, 5, (0, 0, 255), 1)
#         cv2.drawMarker(
#             img_combined,
#             pt2,
#             (0, 255, 0),
#             markerType=cv2.MARKER_CROSS,
#             markerSize=10,
#             thickness=1,
#         )
#
#     if path:
#         ensure_dir(os.path.dirname(path))
#         cv2.imwrite(path, img_combined)
#     return img_combined


# ============================================================
# 4. RMSE / PCR 计算
# ============================================================

def calculate_rmse_with_homography_and_filtering(pts1, pts2, threshold=3.0):
    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)

    if len(pts1) < 4 or len(pts2) < 4:
        return 5.0, np.empty((0, 2)), np.empty((0, 2))

    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)

    if H is None:
        return 5.0, np.empty((0, 2)), np.empty((0, 2))

    pts1_homo = np.concatenate([pts1, np.ones((pts1.shape[0], 1))], axis=1)
    pts1_trans_homo = (H @ pts1_homo.T).T

    denom = pts1_trans_homo[:, 2:3]
    denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
    pts1_trans = pts1_trans_homo[:, :2] / denom

    distances = np.sqrt(np.sum((pts1_trans - pts2) ** 2, axis=1))
    keep = distances <= threshold

    filtered_pts1 = pts1[keep]
    filtered_pts2 = pts2[keep]
    filtered_pts1_trans = pts1_trans[keep]

    if len(filtered_pts1) == 0:
        return 5.0, filtered_pts1, filtered_pts2

    sq_diff = np.sum((filtered_pts1_trans - filtered_pts2) ** 2, axis=1)
    rmse = float(np.sqrt(np.mean(sq_diff)))

    return rmse, filtered_pts1, filtered_pts2


# ============================================================
# 5. Hook 捕获中间特征
# ============================================================

class FeatureHookManager:
    """
    用 forward hook 抓取中间特征，不需要改模型源码。
    """

    def __init__(self):
        self.features = {}
        self.handles = []

    def clear(self):
        self.features.clear()

    def _detach_output(self, output):
        if torch.is_tensor(output):
            return output.detach().cpu()

        if isinstance(output, (list, tuple)):
            out_list = []
            for item in output:
                if torch.is_tensor(item):
                    out_list.append(item.detach().cpu())
                else:
                    out_list.append(item)
            return out_list

        if isinstance(output, dict):
            out_dict = {}
            for k, v in output.items():
                if torch.is_tensor(v):
                    out_dict[k] = v.detach().cpu()
                else:
                    out_dict[k] = v
            return out_dict

        return output

    def add_hook(self, module, name):
        def hook_fn(module_, input_, output_):
            self.features[name] = self._detach_output(output_)
        handle = module.register_forward_hook(hook_fn)
        self.handles.append(handle)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def get_module_by_path(model, module_path):
    """
    支持类似：
    backbone.feature.linear.3
    backbone.feature.cnn_encoder
    backbone.feature.resize.2
    """
    obj = model
    for p in module_path.split("."):
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    return obj


def safe_register_hook(hook_mgr, model, module_path, save_name):
    try:
        module = get_module_by_path(model, module_path)
        hook_mgr.add_hook(module, save_name)
        print(f"[Hook] Registered: {module_path} -> {save_name}")
    except Exception as e:
        print(f"[Hook][WARN] Failed to register {module_path}: {e}")


def register_geosam_hooks(raw_model):
    """
    针对 XoFTR_dino_resnet_panmatch_light_withfpn 注册中间特征 hook。
    如果某些路径不存在，不会中断程序。
    """
    hook_mgr = FeatureHookManager()

    # DINO projection 后的多层语义特征
    safe_register_hook(hook_mgr, raw_model, "backbone.feature.linear.0", "dino_proj_l1")
    safe_register_hook(hook_mgr, raw_model, "backbone.feature.linear.1", "dino_proj_l2")
    safe_register_hook(hook_mgr, raw_model, "backbone.feature.linear.2", "dino_proj_l3")
    safe_register_hook(hook_mgr, raw_model, "backbone.feature.linear.3", "dino_proj_l4")

    # MASE 输出：模态无关结构图
    safe_register_hook(hook_mgr, raw_model, "backbone.feature.masg_extractor", "mase_structure")

    # CNN 结构分支输出，通常是 list/tuple: guide_1, guide_2, guide_3, guide_4
    safe_register_hook(hook_mgr, raw_model, "backbone.feature.cnn_encoder", "cnn_guides")

    # AG-RoPE 对齐层输出
    safe_register_hook(hook_mgr, raw_model, "backbone.feature.resize.0", "aligned_1_2")
    safe_register_hook(hook_mgr, raw_model, "backbone.feature.resize.1", "aligned_1_4")
    safe_register_hook(hook_mgr, raw_model, "backbone.feature.resize.2", "aligned_1_8")
    safe_register_hook(hook_mgr, raw_model, "backbone.feature.resize.3", "aligned_1_16")

    # FPN decoder 输出，通常是 (out_list, saliency_mask)
    safe_register_hook(hook_mgr, raw_model, "backbone.feature.cnn_decoder", "fpn_decoder_out")

    # backbone 最终输出，通常是 (feat_c, feat_m, feat_f, saliency_mask)
    safe_register_hook(hook_mgr, raw_model, "backbone", "backbone_out")

    return hook_mgr


# ============================================================
# 6. 保存当前 pair 的所有中间特征图
# ============================================================

def split_batch_feature(feat, index=0):
    """
    backbone 通常把 image0/image1 cat 后一起前向，所以 batch=2。
    index=0 保存 image0，index=1 保存 image1。
    """
    if feat is None:
        return None

    if isinstance(feat, (list, tuple)):
        return feat

    if not torch.is_tensor(feat):
        return None

    if feat.dim() >= 1 and feat.shape[0] > index:
        return feat[index:index + 1]

    return feat


# def save_visualizations_for_pair(hook_mgr, img0_path, pair_save_dir, img_long_edge=640):
#     """
#     保存 image0 对应的中间特征图。流程图通常展示一张图的中间特征即可。
#     """
#     ensure_dir(pair_save_dir)
#
#     img0_bgr = read_image_bgr(img0_path)
#     img0_bgr = resize_long_edge(img0_bgr, img_long_edge)
#
#     feats = hook_mgr.features
#
#     # 1) DINO semantic priors
#     if "dino_proj_l4" in feats:
#         save_feature_overlay(
#             pair_save_dir,
#             "01_dino_semantic_l4",
#             split_batch_feature(feats["dino_proj_l4"], index=0),
#             img0_bgr,
#             mode=DINO_VIS_MODE,
#         )
#
#     if "dino_proj_l3" in feats:
#         save_feature_overlay(
#             pair_save_dir,
#             "01_dino_semantic_l3",
#             split_batch_feature(feats["dino_proj_l3"], index=0),
#             img0_bgr,
#             mode=DINO_VIS_MODE,
#         )
#
#     # 2) MASE structure
#     if "mase_structure" in feats:
#         save_feature_overlay(
#             pair_save_dir,
#             "02_mase_structure",
#             split_batch_feature(feats["mase_structure"], index=0),
#             img0_bgr,
#             mode=DEFAULT_VIS_MODE,
#         )
#
#     # 3) CNN guide features
#     if "cnn_guides" in feats:
#         guides = feats["cnn_guides"]
#         if isinstance(guides, (list, tuple)) and len(guides) >= 4:
#             names = [
#                 "03_cnn_guide_1_2",
#                 "03_cnn_guide_1_4",
#                 "03_cnn_guide_1_8",
#                 "03_cnn_guide_1_16",
#             ]
#             for name, g in zip(names, guides):
#                 save_feature_overlay(
#                     pair_save_dir,
#                     name,
#                     split_batch_feature(g, index=0),
#                     img0_bgr,
#                     mode=DEFAULT_VIS_MODE,
#                 )
#
#     # 4) AG-RoPE aligned features
#     for key, name in [
#         ("aligned_1_2", "04_agrope_aligned_1_2"),
#         ("aligned_1_4", "04_agrope_aligned_1_4"),
#         ("aligned_1_8", "04_agrope_aligned_1_8"),
#         ("aligned_1_16", "04_agrope_aligned_1_16"),
#     ]:
#         if key in feats:
#             save_feature_overlay(
#                 pair_save_dir,
#                 name,
#                 split_batch_feature(feats[key], index=0),
#                 img0_bgr,
#                 mode=DEFAULT_VIS_MODE,
#             )
#
#     # 5) FPN decoder outputs
#     # fpn_decoder_out 通常是 [out_list, saliency_mask]
#     if "fpn_decoder_out" in feats:
#         fpn_out = feats["fpn_decoder_out"]
#         if isinstance(fpn_out, (list, tuple)) and len(fpn_out) >= 2:
#             out_list = fpn_out[0]
#             saliency = fpn_out[1]
#
#             if isinstance(out_list, (list, tuple)) and len(out_list) >= 4:
#                 # 注意：你的 decoder out 通常顺序是 layer_4, layer_3, layer_2, layer_1
#                 fpn_names = [
#                     "05_fpn_recon_1_16",
#                     "05_fpn_recon_1_8",
#                     "05_fpn_recon_1_4",
#                     "05_fpn_recon_1_2",
#                 ]
#                 for name, f in zip(fpn_names, out_list):
#                     save_feature_overlay(
#                         pair_save_dir,
#                         name,
#                         split_batch_feature(f, index=0),
#                         img0_bgr,
#                         mode=DEFAULT_VIS_MODE,
#                     )
#
#             # saliency mask
#             if torch.is_tensor(saliency):
#                 save_mask_overlay(
#                     pair_save_dir,
#                     "06_saliency_mask",
#                     split_batch_feature(saliency, index=0),
#                     img0_bgr,
#                 )
#
#     # 6) backbone final outputs
#     # backbone_out 通常是 (feat_c, feat_m, feat_f, saliency_mask)
#     if "backbone_out" in feats:
#         bb_out = feats["backbone_out"]
#         if isinstance(bb_out, (list, tuple)) and len(bb_out) >= 4:
#             feat_c, feat_m, feat_f, saliency_mask = bb_out[:4]
#
#             save_feature_overlay(
#                 pair_save_dir,
#                 "07_final_coarse_feature_1_8",
#                 split_batch_feature(feat_c, index=0),
#                 img0_bgr,
#                 mode=DEFAULT_VIS_MODE,
#             )
#             save_feature_overlay(
#                 pair_save_dir,
#                 "07_final_mid_feature_1_4",
#                 split_batch_feature(feat_m, index=0),
#                 img0_bgr,
#                 mode=DEFAULT_VIS_MODE,
#             )
#             save_feature_overlay(
#                 pair_save_dir,
#                 "07_final_fine_feature_1_2",
#                 split_batch_feature(feat_f, index=0),
#                 img0_bgr,
#                 mode=DEFAULT_VIS_MODE,
#             )
#             save_mask_overlay(
#                 pair_save_dir,
#                 "07_final_saliency_mask",
#                 split_batch_feature(saliency_mask, index=0),
#                 img0_bgr,
#             )
#
#     print(f"[VIS] Saved feature maps to: {pair_save_dir}")

def save_visualizations_for_pair(hook_mgr, img0_path, img1_path, pair_save_dir, img_long_edge=640):
    """
    同时保存 image0 和 image1 的中间特征图。
    index=0 对应 image0，index=1 对应 image1。
    """
    ensure_dir(pair_save_dir)

    img0_bgr = read_image_bgr(img0_path)
    img1_bgr = read_image_bgr(img1_path)

    img0_bgr = resize_long_edge(img0_bgr, img_long_edge)
    img1_bgr = resize_long_edge(img1_bgr, img_long_edge)

    feats = hook_mgr.features

    for img_index, img_bgr, prefix in [
        (0, img0_bgr, "image0_optical"),
        (1, img1_bgr, "image1_ir_sar"),
    ]:
        # 1) DINO semantic priors
        if "dino_proj_l4" in feats:
            save_feature_overlay(
                pair_save_dir,
                f"{prefix}_01_dino_semantic_l4",
                split_batch_feature(feats["dino_proj_l4"], index=img_index),
                img_bgr,
                mode=DINO_VIS_MODE,
            )

        if "dino_proj_l3" in feats:
            save_feature_overlay(
                pair_save_dir,
                f"{prefix}_01_dino_semantic_l3",
                split_batch_feature(feats["dino_proj_l3"], index=img_index),
                img_bgr,
                mode=DINO_VIS_MODE,
            )

        # 2) MASE structure
        if "mase_structure" in feats:
            save_feature_overlay(
                pair_save_dir,
                f"{prefix}_02_mase_structure",
                split_batch_feature(feats["mase_structure"], index=img_index),
                img_bgr,
                mode=DEFAULT_VIS_MODE,
            )

        # 3) CNN guide features
        if "cnn_guides" in feats:
            guides = feats["cnn_guides"]
            if isinstance(guides, (list, tuple)) and len(guides) >= 4:
                names = [
                    "03_cnn_guide_1_2",
                    "03_cnn_guide_1_4",
                    "03_cnn_guide_1_8",
                    "03_cnn_guide_1_16",
                ]
                for name, g in zip(names, guides):
                    save_feature_overlay(
                        pair_save_dir,
                        f"{prefix}_{name}",
                        split_batch_feature(g, index=img_index),
                        img_bgr,
                        mode=DEFAULT_VIS_MODE,
                    )

        # 4) AG-RoPE aligned features
        for key, name in [
            ("aligned_1_2", "04_agrope_aligned_1_2"),
            ("aligned_1_4", "04_agrope_aligned_1_4"),
            ("aligned_1_8", "04_agrope_aligned_1_8"),
            ("aligned_1_16", "04_agrope_aligned_1_16"),
        ]:
            if key in feats:
                save_feature_overlay(
                    pair_save_dir,
                    f"{prefix}_{name}",
                    split_batch_feature(feats[key], index=img_index),
                    img_bgr,
                    mode=DEFAULT_VIS_MODE,
                )

        # 5) FPN decoder outputs
        if "fpn_decoder_out" in feats:
            fpn_out = feats["fpn_decoder_out"]
            if isinstance(fpn_out, (list, tuple)) and len(fpn_out) >= 2:
                out_list = fpn_out[0]
                saliency = fpn_out[1]

                if isinstance(out_list, (list, tuple)) and len(out_list) >= 4:
                    fpn_names = [
                        "05_fpn_recon_1_16",
                        "05_fpn_recon_1_8",
                        "05_fpn_recon_1_4",
                        "05_fpn_recon_1_2",
                    ]
                    for name, f in zip(fpn_names, out_list):
                        save_feature_overlay(
                            pair_save_dir,
                            f"{prefix}_{name}",
                            split_batch_feature(f, index=img_index),
                            img_bgr,
                            mode=DEFAULT_VIS_MODE,
                        )

                if torch.is_tensor(saliency):
                    save_mask_overlay(
                        pair_save_dir,
                        f"{prefix}_06_saliency_mask",
                        split_batch_feature(saliency, index=img_index),
                        img_bgr,
                    )

        # 6) backbone final outputs
        if "backbone_out" in feats:
            bb_out = feats["backbone_out"]
            if isinstance(bb_out, (list, tuple)) and len(bb_out) >= 4:
                feat_c, feat_m, feat_f, saliency_mask = bb_out[:4]

                save_feature_overlay(
                    pair_save_dir,
                    f"{prefix}_07_final_coarse_feature_1_8",
                    split_batch_feature(feat_c, index=img_index),
                    img_bgr,
                    mode=DEFAULT_VIS_MODE,
                )
                save_feature_overlay(
                    pair_save_dir,
                    f"{prefix}_07_final_mid_feature_1_4",
                    split_batch_feature(feat_m, index=img_index),
                    img_bgr,
                    mode=DEFAULT_VIS_MODE,
                )
                save_feature_overlay(
                    pair_save_dir,
                    f"{prefix}_07_final_fine_feature_1_2",
                    split_batch_feature(feat_f, index=img_index),
                    img_bgr,
                    mode=DEFAULT_VIS_MODE,
                )
                save_mask_overlay(
                    pair_save_dir,
                    f"{prefix}_07_final_saliency_mask",
                    split_batch_feature(saliency_mask, index=img_index),
                    img_bgr,
                )

    print(f"[VIS] Saved image0 and image1 feature maps to: {pair_save_dir}")
# ============================================================
# 7. 主程序
# ============================================================

if __name__ == "__main__":
    ensure_dir(MATCH_SAVE_DIR)
    ensure_dir(VIS_SAVE_DIR)

    L_files = natural_sort([
        f for f in os.listdir(L_FOLDER)
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"))
    ])
    R_files = natural_sort([
        f for f in os.listdir(R_FOLDER)
        if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"))
    ])

    print(f"Found {len(L_files)} left images and {len(R_files)} right images.")

    # ------------------------------------------------------------
    # 配置
    # ------------------------------------------------------------
    config = get_cfg_defaults(inference=True)
    config = lower_config(config)

    config["xoftr"]["match_coarse"]["thr"] = 0.3
    config["xoftr"]["fine"]["thr"] = 0.1
    config["xoftr"]["fine"]["denser"] = False

    config["test"]["img0_resize"] = IMG0_RESIZE
    config["test"]["img1_resize"] = IMG1_RESIZE

    # ------------------------------------------------------------
    # 模型
    # ------------------------------------------------------------
    # 这里一定要用你的方法，而不是 XoFTR_Light
    raw_model = XoFTR_dino_resnet_panmatch_light_withfpn(config=config["xoftr"])

    # 注册 hook：必须在 DataIOWrapper 包装前注册
    hook_mgr = register_geosam_hooks(raw_model)

    matcher = DataIOWrapper(raw_model, config=config["test"], ckpt=CKPT_PATH)
    matcher = matcher.eval()

    # ------------------------------------------------------------
    # 指标统计
    # ------------------------------------------------------------
    points = 0
    NCM = 0
    RMSE = 0
    PCR = 0
    SR = 0

    num_pairs = min(len(L_files), len(R_files))

    for idx, (L_file, R_file) in enumerate(zip(L_files, R_files)):
        img0_pth = os.path.join(L_FOLDER, L_file)
        img1_pth = os.path.join(R_FOLDER, R_file)

        print("\n" + "=" * 80)
        print(f"[{idx + 1}/{num_pairs}] Processing:")
        print(f"  image0: {img0_pth}")
        print(f"  image1: {img1_pth}")

        # 每一对图像前清空 hook 缓存
        hook_mgr.clear()

        # --------------------------------------------------------
        # 前向匹配
        # --------------------------------------------------------
        start_time = time.time()

        try:
            output_data = matcher.from_paths(img0_pth, img1_pth, read_color=True)
        except Exception as e:
            print(f"[ERROR] Matching failed for pair {idx + 1}: {e}")
            continue
        run_visualization(hook_mgr, "./paper_fig_vis")
        elapsed = time.time() - start_time
        print(f"[Time] {elapsed:.3f}s")

        # --------------------------------------------------------
        # 保存中间特征图
        # --------------------------------------------------------
        if SAVE_VIS and idx < SAVE_VIS_NUM:
            pair_vis_dir = os.path.join(
                VIS_SAVE_DIR,
                f"{idx + 1:04d}_{Path(L_file).stem}_{Path(R_file).stem}"
            )
            try:
                # save_visualizations_for_pair(
                #     hook_mgr,
                #     img0_pth,
                #     pair_vis_dir,
                #     img_long_edge=IMG0_RESIZE,
                # )
                save_visualizations_for_pair(
                    hook_mgr,
                    img0_path=img0_pth,
                    img1_path=img1_pth,
                    pair_save_dir=pair_vis_dir,
                    img_long_edge=IMG0_RESIZE,
                )
            except Exception as e:
                print(f"[VIS][WARN] Failed to save feature maps: {e}")

        # --------------------------------------------------------
        # 匹配点与指标
        # --------------------------------------------------------
        mkpts0 = output_data.get("mkpts0", np.empty((0, 2)))
        mkpts1 = output_data.get("mkpts1", np.empty((0, 2)))
        mconf = output_data.get("mconf", np.empty((0,)))

        mkpts0 = np.asarray(mkpts0)
        mkpts1 = np.asarray(mkpts1)

        if len(mkpts0) > 8:
            rmse, filtered_pts1, filtered_pts2 = calculate_rmse_with_homography_and_filtering(
                mkpts0, mkpts1, threshold=3.0
            )

            output_path = os.path.join(
                MATCH_SAVE_DIR,
                f"{idx + 1:04d}_match_{len(filtered_pts1)}.png"
            )
            draw_matches(img0_pth, img1_pth, filtered_pts1, filtered_pts2, output_path)

            if len(filtered_pts1) < 20:
                rmse = 5.0

            if len(filtered_pts1) > 20 and (len(filtered_pts1) / max(len(mkpts0), 1)) > 0.2:
                SR += 1
                print("图像匹配成功")
        else:
            rmse = 5.0
            filtered_pts1 = np.empty((0, 2))
            filtered_pts2 = np.empty((0, 2))
            mkpts0 = np.asarray([[1, 1]])

        pcr = len(filtered_pts1) / max(len(mkpts0), 1)

        print(f"图像匹配的 RMSE 为: {rmse:.4f}")
        print(f"图像匹配的点数为: {len(mkpts0):.4f}")
        print(f"过滤后的点对数: {len(filtered_pts1)}")
        print(f"图像匹配的 PCR 为: {pcr:.4f}")

        points += len(mkpts0)
        NCM += len(filtered_pts1)
        RMSE += rmse
        PCR += pcr

    # ------------------------------------------------------------
    # 总体结果
    # ------------------------------------------------------------
    denom = max(num_pairs, 1)
    data = {
        "pairs": num_pairs,
        "avg_配对点数": points / denom,
        "avg_NCM": NCM / denom,
        "avg_RMSE": RMSE / denom,
        "avg_PCR": PCR / denom,
        "SR": SR,
    }

    print("\n" + "=" * 80)
    print("Final results:")
    print(data)

    json_path = os.path.join(MATCH_SAVE_DIR, "summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Summary saved to: {json_path}")

    # 移除 hooks
    hook_mgr.remove()
"""
在你的主程序 forward 之后加：

run_visualization(hook_mgr, "./paper_fig_vis")

即可输出：

paper_fig_vis/
    ggsa_cnn.png
    ggsa_dino.png
    ggsa_naive.png
    ggsa_ggsa.png

    sassm_raw.png
    sassm_mask.png
    sassm_gated.png
    sassm_mamba_out.png

    mamba_scan.png
    mamba_accumulation.png
"""

