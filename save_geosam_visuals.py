import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F


def normalize_to_uint8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return (x * 255).astype(np.uint8)


def feature_to_heatmap(feat: torch.Tensor, out_size=None, mode="norm") -> np.ndarray:
    """
    feat: [C,H,W] or [1,C,H,W]
    mode:
        - "norm": channel-wise L2 norm, most stable
        - "mean": channel mean
        - "max": channel max
        - "pca": first PCA component, more semantic-looking
    """
    if feat.dim() == 4:
        feat = feat[0]
    feat = feat.detach().float().cpu()

    if mode == "norm":
        fmap = torch.norm(feat, p=2, dim=0)
    elif mode == "mean":
        fmap = feat.mean(dim=0)
    elif mode == "max":
        fmap = feat.max(dim=0)[0]
    elif mode == "pca":
        C, H, W = feat.shape
        x = feat.reshape(C, -1).T  # [HW,C]
        x = x - x.mean(dim=0, keepdim=True)
        # torch.pca_lowrank may be slow for huge feature maps; okay for visual use
        _, _, v = torch.pca_lowrank(x, q=1)
        fmap = (x @ v[:, :1]).reshape(H, W)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    fmap = fmap.numpy()
    fmap = normalize_to_uint8(fmap)

    if out_size is not None:
        # out_size: (W,H) for cv2
        fmap = cv2.resize(fmap, out_size, interpolation=cv2.INTER_CUBIC)

    heatmap = cv2.applyColorMap(fmap, cv2.COLORMAP_JET)
    return heatmap


def mask_to_heatmap(mask: torch.Tensor, out_size=None) -> np.ndarray:
    """
    mask: [1,H,W] or [B,1,H,W]
    """
    if mask.dim() == 4:
        mask = mask[0, 0]
    elif mask.dim() == 3:
        mask = mask[0]
    mask = mask.detach().float().cpu().numpy()
    mask = normalize_to_uint8(mask)

    if out_size is not None:
        mask = cv2.resize(mask, out_size, interpolation=cv2.INTER_CUBIC)

    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    return heatmap


def tensor_image_to_bgr(img: torch.Tensor) -> np.ndarray:
    """
    img: [1,H,W] or [3,H,W] or [B,C,H,W]
    return BGR uint8 image for cv2
    """
    if img.dim() == 4:
        img = img[0]
    img = img.detach().float().cpu()

    if img.shape[0] == 1:
        arr = img[0].numpy()
        arr = normalize_to_uint8(arr)
        bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    else:
        arr = img[:3].permute(1, 2, 0).numpy()
        arr = normalize_to_uint8(arr)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    return bgr


def overlay_heatmap_on_image(img_bgr: np.ndarray, heatmap_bgr: np.ndarray, alpha=0.45) -> np.ndarray:
    if heatmap_bgr.shape[:2] != img_bgr.shape[:2]:
        heatmap_bgr = cv2.resize(heatmap_bgr, (img_bgr.shape[1], img_bgr.shape[0]))
    return cv2.addWeighted(img_bgr, 1 - alpha, heatmap_bgr, alpha, 0)


def save_bgr(path: str, img_bgr: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img_bgr)