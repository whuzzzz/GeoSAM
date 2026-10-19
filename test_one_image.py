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

from src.xoftr import XoFTR, XoFTR_dino_resnet_panmatch_light_withfpn, XoFTR_Light_convnextv2, XoFTR_Light, \
    XoFTR_dino_resnet_panmatch_light_withfpn
from src.config.default import get_cfg_defaults
from src.utils.data_io import DataIOWrapper, lower_config
import time
import re



# def draw_matches(img1, img2, keypoints1, keypoints2, path):
#     # 拼接两张图像
#     img1 = cv2.imread(img1, cv2.IMREAD_COLOR)
#     img2 = cv2.imread(img2, cv2.IMREAD_COLOR)
#     # # 调整两张图像的大小
#     # if img1 is not None:
#     #     img1, keypoints1 = resize_image_and_points(img1, keypoints1, 640)
#
#     # if img2 is not None:
#     #     img2, keypoints2 = resize_image_and_points(img2, keypoints2, 640)
#
#     height1, width1 = img1.shape[:2]
#     height2, width2 = img2.shape[:2]
#     height = max(height1, height2)
#     width = width1 + width2
#     img_combined = np.zeros((height, width, 3), dtype='uint8')
#
#     # 确保两张图像都是三通道的彩色图像
#     # if len(img1.shape) == 2:
#     #     img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
#     # if len(img2.shape) == 2:
#     #     img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
#
#     img_combined[:height1, :width1] = img1
#     img_combined[:height2, width1:] = img2
#
#     # 调整右侧图像的匹配点坐标
#     keypoints2[:, 0] += width1
#
#     # 将匹配点保存在一个二维数组中
#     matched_points = np.hstack((keypoints1, keypoints2))  # 组合成 (num_matches, 4) 的数组，前两列为 pts1，后两列为 pts2
#
#     # 根据二维数组画出连接线，并标记红色圆点
#     for i in range(matched_points.shape[0]):
#         pt1 = tuple(map(int, matched_points[i, :2]))  # 左边图像的点，确保为整数元组
#         pt2 = tuple(map(int, matched_points[i, 2:]))  # 右边图像的点，确保为整数元组
#
#         # 画黄色的线连接点对
#         cv2.line(img_combined, pt1, pt2, (0, 255, 255), 1)  # 黄色线
#
#         # 在每个同名点对的位置画红色圆点
#         # cv2.circle(img_combined, pt1, 5, (0, 0, 255), -1)  # 红色圆，左边图像点
#         # cv2.circle(img_combined, pt2, 5, (0, 0, 255), -1)  # 红色圆，右边图像点
#         # 在左边图像中的匹配点画红色空心圆
#         cv2.circle(img_combined, pt1, 5, (0, 0, 255), 1)  # 红色空心圆，厚度为2
#
#         # 在右边图像中的匹配点画绿色加号
#         cv2.drawMarker(img_combined, pt2, (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)  # 绿色加号
#
#     # save or return figure
#     if path:
#         # plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
#         # plt.close()
#         cv2.imwrite(path, img_combined)
#     else:
#         return img_combined
def draw_matches(img1_path, img2_path, keypoints1, keypoints2, path,
                 target_height=640):

    # 读取图像
    img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)

    if img1 is None or img2 is None:
        raise ValueError("图像读取失败")

    # ==========================================================
    # 1. 分别等比例缩放到同样高度
    # ==========================================================
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    scale1 = target_height / h1
    scale2 = target_height / h2

    new_w1 = int(w1 * scale1)
    new_w2 = int(w2 * scale2)

    img1 = cv2.resize(
        img1,
        (new_w1, target_height),
        interpolation=cv2.INTER_LINEAR
    )

    img2 = cv2.resize(
        img2,
        (new_w2, target_height),
        interpolation=cv2.INTER_LINEAR
    )

    # ==========================================================
    # 2. 同步缩放匹配点坐标
    # ==========================================================
    keypoints1 = keypoints1.copy().astype(np.float32)
    keypoints2 = keypoints2.copy().astype(np.float32)

    keypoints1[:, 0] *= scale1
    keypoints1[:, 1] *= scale1

    keypoints2[:, 0] *= scale2
    keypoints2[:, 1] *= scale2

    # ==========================================================
    # 3. 横向拼接两张图
    # ==========================================================
    combined_width = new_w1 + new_w2

    img_combined = np.zeros(
        (target_height, combined_width, 3),
        dtype=np.uint8
    )

    img_combined[:, :new_w1] = img1
    img_combined[:, new_w1:] = img2

    # 右图匹配点加上横向偏移
    keypoints2[:, 0] += new_w1

    # ==========================================================
    # 4. 绘制匹配
    # ==========================================================
    for pt1, pt2 in zip(keypoints1, keypoints2):

        p1 = tuple(np.round(pt1).astype(int))
        p2 = tuple(np.round(pt2).astype(int))

        # 黄色匹配线
        cv2.line(
            img_combined,
            p1,
            p2,
            (0, 255, 255),
            1
        )

        # 左图：红色空心圆
        cv2.circle(
            img_combined,
            p1,
            4,
            (0, 0, 255),
            1
        )

        # 右图：绿色十字
        cv2.drawMarker(
            img_combined,
            p2,
            (0, 255, 0),
            markerType=cv2.MARKER_CROSS,
            markerSize=8,
            thickness=1
        )

    # ==========================================================
    # 5. 保存
    # ==========================================================
    if path:
        cv2.imwrite(path, img_combined)
        print(f"匹配图已保存到: {path}")

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
    print("rmse", rmse)

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
    return sorted(file_list,
                  key=lambda x: [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', x)])


if __name__ == "__main__":

    # ============================================================
    # 1. 输入一组待匹配图像
    # ============================================================
    img0_pth = "./data/METU_VisTIR/sunny/scene_1/visible/images/IM_10308.jpg"
    img1_pth = "./data/METU_VisTIR/sunny/scene_1/thermal/images/IM_10308.jpg"

    # 匹配结果保存位置
    output_folder = "./single_pair_result"
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, "match.png")

    # ============================================================
    # 2. 加载模型配置
    # ============================================================
    config = get_cfg_defaults(inference=True)
    config = lower_config(config)

    # Coarse level threshold
    config['xoftr']['match_coarse']['thr'] = 0.3

    # Fine level threshold
    config['xoftr']['fine']['thr'] = 0.1

    # 是否使用更稠密匹配
    config['xoftr']['fine']['denser'] = False

    # ============================================================
    # 3. 创建模型
    # ============================================================
    matcher = XoFTR_dino_resnet_panmatch_light_withfpn(
        config=config["xoftr"]
    )

    # 输入图像长边缩放到640
    config['test']['img0_resize'] = 640
    config['test']['img1_resize'] = 640

    # ============================================================
    # 4. 模型权重
    # ============================================================
    ckpt = "./logs/tb_logs/dinomatch/version_32/checkpoints/epoch=27-auc@5=0.206-auc@10=0.378-auc@20=0.554.ckpt"

    matcher = DataIOWrapper(
        matcher,
        config=config["test"],
        ckpt=ckpt
    )

    matcher = matcher.eval()

    # ============================================================
    # 5. 开始匹配
    # ============================================================
    print("=" * 60)
    print("Image 0:", img0_pth)
    print("Image 1:", img1_pth)
    print("=" * 60)

    start_time = time.time()

    with torch.no_grad():
        output_data = matcher.from_paths(
            img0_pth,
            img1_pth,
            read_color=True
        )

    elapsed_time = time.time() - start_time

    # 原始匹配点
    mkpts0 = output_data['mkpts0']
    mkpts1 = output_data['mkpts1']

    # 匹配置信度
    mconf = output_data['mconf']

    num_matches = len(mkpts0)

    # ============================================================
    # 6. 单应矩阵几何验证
    # ============================================================
    if num_matches >= 4:

        rmse, filtered_pts0, filtered_pts1 = \
            calculate_rmse_with_homography_and_filtering(
                mkpts0,
                mkpts1,
                threshold=3.0
            )

        # 防止无法计算
        if rmse is None:
            rmse = 5.0

        ncm = len(filtered_pts0)

        if num_matches > 0:
            pcr = ncm / num_matches
        else:
            pcr = 0.0

        # 与原代码保持一致：
        # 正确点数 > 20 且 PCR > 0.2，认为匹配成功
        success = (ncm > 20) and (pcr > 0.2)

        # ========================================================
        # 7. 保存匹配图
        # ========================================================
        if ncm > 0:
            draw_matches(
                img0_pth,
                img1_pth,
                filtered_pts0.copy(),
                filtered_pts1.copy(),
                output_path
            )

    else:
        rmse = 5.0
        filtered_pts0 = np.empty((0, 2))
        filtered_pts1 = np.empty((0, 2))

        ncm = 0
        pcr = 0.0
        success = False

    # ============================================================
    # 8. 输出单张图像对指标
    # ============================================================
    print("\n" + "=" * 60)
    print("Single Pair Matching Result")
    print("=" * 60)

    print(f"Raw Matches : {num_matches}")
    print(f"NCM         : {ncm}")
    print(f"RMSE        : {rmse:.4f} px")
    print(f"PCR         : {pcr:.4f}")
    print(f"PCR         : {pcr * 100:.2f}%")
    print(f"Success     : {success}")
    print(f"Time        : {elapsed_time:.4f} s")

    if ncm > 0:
        print(f"Match Image : {output_path}")

    print("=" * 60)

    # ============================================================
    # 9. 保存指标为JSON
    # ============================================================
    result = {
        "image0": os.path.basename(img0_pth),
        "image1": os.path.basename(img1_pth),
        "raw_matches": int(num_matches),
        "NCM": int(ncm),
        "RMSE": float(rmse),
        "PCR": float(pcr),
        "success": bool(success),
        "time": float(elapsed_time)
    }

    json_path = os.path.join(output_folder, "result.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            result,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("JSON saved to:", json_path)