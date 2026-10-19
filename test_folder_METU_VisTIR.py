import json
import os
# Set CUDA_VISIBLE_DEVICES in the shell when selecting GPUs.
import torch
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure

from src.xoftr import XoFTR, XoFTR_dino_resnet_panmatch_light_withfpn, XoFTR_Light_convnextv2
from src.config.default import get_cfg_defaults
from src.utils.data_io import DataIOWrapper, lower_config
import time
import re

def resize_image_and_points(img, points, target_long_edge=640):
    """
    缩放图像，并同步缩放点坐标
    Args:
        img: 输入图像
        points: 点坐标列表，格式为 [(x1, y1), (x2, y2), ...]
        target_long_edge: 目标长边长度
    Returns:
        resized_img: 缩放后的图像
        scaled_points: 缩放后的点坐标
        scale: 使用的缩放比例
    """
    height, width = img.shape[:2]
    long_edge = max(height, width)
    
    # 计算缩放比例
    scale = target_long_edge / long_edge
    
    # 缩放图像
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # 缩放点坐标
    scaled_points = [(int(x * scale), int(y * scale)) for (x, y) in points]
    scaled_points = np.array(scaled_points)
    
    return resized_img, scaled_points
def draw_matches(img1, img2, keypoints1, keypoints2, path):
    # 拼接两张图像
    img1 = cv2.imread(img1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2, cv2.IMREAD_COLOR)
    # 调整两张图像的大小
    if img1 is not None:
        img1, keypoints1 = resize_image_and_points(img1, keypoints1, 640)

    if img2 is not None:
        img2, keypoints2 = resize_image_and_points(img2, keypoints2, 640)
        
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]
    height = max(height1, height2)
    width = width1 + width2
    img_combined = np.zeros((height, width, 3), dtype='uint8')

    # 确保两张图像都是三通道的彩色图像
    # if len(img1.shape) == 2:
    #     img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    # if len(img2.shape) == 2:
    #     img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    img_combined[:height1, :width1] = img1
    img_combined[:height2, width1:] = img2

    # 调整右侧图像的匹配点坐标
    keypoints2[:, 0] += width1

    # 将匹配点保存在一个二维数组中
    matched_points = np.hstack((keypoints1, keypoints2))  # 组合成 (num_matches, 4) 的数组，前两列为 pts1，后两列为 pts2

    # 根据二维数组画出连接线，并标记红色圆点
    for i in range(matched_points.shape[0]):
        pt1 = tuple(map(int, matched_points[i, :2]))  # 左边图像的点，确保为整数元组
        pt2 = tuple(map(int, matched_points[i, 2:]))  # 右边图像的点，确保为整数元组

        # 画黄色的线连接点对
        cv2.line(img_combined, pt1, pt2, (0, 255, 255), 1)  # 黄色线

        # 在每个同名点对的位置画红色圆点
        # cv2.circle(img_combined, pt1, 5, (0, 0, 255), -1)  # 红色圆，左边图像点
        # cv2.circle(img_combined, pt2, 5, (0, 0, 255), -1)  # 红色圆，右边图像点
        # 在左边图像中的匹配点画红色空心圆
        cv2.circle(img_combined, pt1, 5, (0, 0, 255), 1)  # 红色空心圆，厚度为2

        # 在右边图像中的匹配点画绿色加号
        cv2.drawMarker(img_combined, pt2, (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)  # 绿色加号

    # save or return figure
    if path:
        # plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        # plt.close()
        cv2.imwrite(path, img_combined)
    else:
        return img_combined


def calculate_rmse_with_homography(pts1, pts2):
    """
    计算匹配点对之间的 RMSE（均方根误差），使用单应性矩阵将左图中的点转换到右图坐标系

    参数:
    pts1: np.array，左边图像的匹配点坐标，形状为 (N, 2)
    pts2: np.array，右边图像的匹配点坐标，形状为 (N, 2)

    返回:
    RMSE: float，匹配点对之间的均方根误差
    """
    # 计算单应性矩阵 H，将左图的点 pts1 转换到右图坐标系
    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)

    # 将 pts1 点转换为齐次坐标形式，适合透视变换
    pts1_homogeneous = np.concatenate([pts1, np.ones((pts1.shape[0], 1))], axis=1)  # 形状 (N, 3)

    # 使用单应性矩阵 H 将左图的点 pts1 变换到右图坐标系中
    pts1_transformed_homogeneous = (H @ pts1_homogeneous.T).T  # 透视变换后的点，形状 (N, 3)

    # 将齐次坐标还原为 2D 坐标 (x, y)
    pts1_transformed = pts1_transformed_homogeneous[:, :2] / pts1_transformed_homogeneous[:, 2].reshape(-1, 1)

    # 计算转换后的点与右图点之间的欧氏距离平方
    squared_diff = np.sum((pts1_transformed - pts2) ** 2, axis=1)

    # 计算 MSE（均方误差）
    mse = np.mean(squared_diff)

    # 计算 RMSE（均方根误差）
    rmse = np.sqrt(mse)
    print("rmse",rmse)

    return rmse


def calculate_rmse_with_homography_and_filtering(pts1, pts2, threshold=3.0):
    """
    计算匹配点对之间的 RMSE，使用单应性矩阵将左图中的点转换到右图坐标系，去除误差超过阈值的点

    参数:
    pts1: np.array，左边图像的匹配点坐标，形状为 (N, 2)
    pts2: np.array，右边图像的匹配点坐标，形状为 (N, 2)
    threshold: float，误差阈值，单位为像素，默认是3.0像素

    返回:
    RMSE: float，匹配点对之间的均方根误差
    filtered_pts1: 过滤后的左图匹配点
    filtered_pts2: 过滤后的右图匹配点
    """
    # 计算单应性矩阵 H，将左图的点 pts1 转换到右图坐标系
    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)

    # 将 pts1 点转换为齐次坐标形式
    pts1_homogeneous = np.concatenate([pts1, np.ones((pts1.shape[0], 1))], axis=1)  # 形状 (N, 3)

    # 使用单应性矩阵 H 将左图的点 pts1 变换到右图坐标系中
    pts1_transformed_homogeneous = (H @ pts1_homogeneous.T).T  # 透视变换后的点，形状 (N, 3)

    # 将齐次坐标还原为 2D 坐标 (x, y)
    pts1_transformed = pts1_transformed_homogeneous[:, :2] / pts1_transformed_homogeneous[:, 2].reshape(-1, 1)

    # 计算转换后的点与右图点之间的欧氏距离平方
    distances = np.sqrt(np.sum((pts1_transformed - pts2) ** 2, axis=1))

    # 过滤误差超过阈值的点对
    filtered_indices = distances <= threshold
    filtered_pts1 = pts1[filtered_indices]
    filtered_pts2 = pts2[filtered_indices]
    filtered_pts1_transformed = pts1_transformed[filtered_indices]

    if len(filtered_pts1) == 0:
        print("所有点的误差都超过了阈值，无法计算 RMSE")
        return None, filtered_pts1, filtered_pts2

    # 重新计算过滤后的欧氏距离平方
    filtered_squared_diff = np.sum((filtered_pts1_transformed - filtered_pts2) ** 2, axis=1)

    # 计算过滤后的 MSE 和 RMSE
    mse = np.mean(filtered_squared_diff)
    rmse = np.sqrt(mse)

    # if rmse is not None:
    #     print(f"图像匹配的 RMSE 为: {rmse:.4f}")
    #     print(f"过滤后的点对数: {len(filtered_pts1)}")
    # else:
    #     print("无法计算 RMSE")

    return rmse, filtered_pts1, filtered_pts2

# 获取L文件夹和R文件夹中的图像文件名并排序
def natural_sort(file_list):
    """按照自然顺序对文件列表进行排序"""
    return sorted(file_list, key=lambda x: [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', x)])

if __name__ == "__main__":
    L_folder = "./data/day_night/night"
    R_folder = "./data/day_night/day"
    L_files = natural_sort([f for f in os.listdir(L_folder) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif')])
    R_files = natural_sort([f for f in os.listdir(R_folder) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tif')])
    # output_folder = r"G:\XoFTR-main\test\MultiResSAR_datasets\result1.json"

    # Get default configurations
    config = get_cfg_defaults(inference=True)
    config = lower_config(config)

    # Coarse level threshold
    config['xoftr']['match_coarse']['thr'] = 0.3  # Default 0.3

    # Fine level threshold
    config['xoftr']['fine']['thr'] = 0.1  # Default 0.1

    # It is posseble to get denser matches
    # If True, xoftr returns all fine-level matches for each fine-level window (at 1/2 resolution)
    config['xoftr']['fine']['denser'] = False  # Default False

    # XoFTR model
    matcher = XoFTR_dino_resnet_panmatch_light_withfpn(config=config["xoftr"])
    # matcher = XoFTR_Light_convnextv2(config=config["xoftr"])


    # The input image sizes for xoftr
    # Note: The output matches and output images are in original image size
    config['test']['img0_resize'] = 640  # resize the longer side, None for no resize
    config['test']['img1_resize'] = 640  # resize the longer side, None for no resize

    # The path for weights
    ckpt = "./logs/tb_logs/train20260128dinov3_vitb_resnet_panmatch_light/version_1没有预训练/checkpoints/epoch=25-auc@5=0.194-auc@10=0.368-auc@20=0.540.ckpt"
    # ckpt = "./logs/tb_logs/train20260102coarsematch_convnextv2/version_0convnextv2_femto48_96_192/checkpoints/epoch=18-auc@5=0.186-auc@10=0.343-auc@20=0.504.ckpt"
    # ckpt = r"C:\Users\PC\Desktop\fsdownload\epoch=8-auc@5=0.153-auc@10=0.283-auc@20=0.435.ckpt"

    # Data I/O wrapper
    matcher = DataIOWrapper(matcher, config=config["test"], ckpt=ckpt)
    matcher = matcher.eval()  # .cuda()

    # 配对点数
    points = 0
    # NCM
    NCM = 0
    # RMSE
    RMSE = 0
    # PCR
    PCR = 0
    # 成功匹配的图片
    SR = 0
    # 批量处理图像
    for idx, (L_file, R_file) in enumerate(zip(L_files, R_files)):
        img0_pth = os.path.join(L_folder, L_file)
        img1_pth = os.path.join(R_folder, R_file)
        # img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
        # img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)
        # Load and match images
        # Note: images are converted to grayscale before matching
        output_data = matcher.from_paths(img0_pth, img1_pth, read_color=True)

        # Matched keypoints
        mkpts0 = output_data['mkpts0']
        mkpts1 = output_data['mkpts1']

        # Confidence values for fine-level matching
        mconf = output_data['mconf']

        if len(mkpts0) > 8:
            # 剔除误差超过三个像素的匹配点，同时计算rmse和pcr
            rmse, filtered_pts1, filtered_pts2 = calculate_rmse_with_homography_and_filtering(mkpts0, mkpts1,
                                                                                              threshold=3.0)
            output_path1 = './test/day-night'
            output_path = os.path.join(output_path1, f"{idx + 1}_match_{len(filtered_pts1)}.png")
            fig = draw_matches(img0_pth, img1_pth, filtered_pts1, filtered_pts2, output_path)
            if len(filtered_pts1) < 20:
                rmse = 5
            if len(filtered_pts1) > 20 and (len(filtered_pts1) / len(mkpts0)) > 0.2:
                SR += 1
                print(f"图像匹配成功")
        else:
            rmse, filtered_pts1, filtered_pts2 = 5, [], []
            mkpts0 = [[1, 1]]

        print(f"图像匹配的 RMSE 为: {rmse:.4f}")
        print(f"图像匹配的点数为: {len(mkpts0):.4f}")
        print(f"过滤后的点对数: {len(filtered_pts1)}")
        print(f"图像匹配的PCR为: {(len(filtered_pts1) / len(mkpts0)):.4f}")
        points += len(mkpts0)
        NCM += len(filtered_pts1)
        RMSE += rmse
        PCR += (len(filtered_pts1) / len(mkpts0))
        print(f"Processed and saved: Low-Resolution{idx + 1}.jpg")
    idx = len(L_files)
    data = {
        "配对点数": points / idx,
        "NCM": NCM / idx,
        "RMSE": RMSE / idx,
        "PCR": PCR / idx,
        "SR": SR,
    }
    print(data)
    # json_file_path = os.path.join(output_folder, "pair2match.json")
    # with open(output_folder, "w", encoding="utf-8") as json_file:
    #     json.dump(data, json_file, indent=4, ensure_ascii=False)