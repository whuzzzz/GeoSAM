import argparse
import csv
import json
import logging
import os
import os.path as osp
import time
import warnings
from collections import defaultdict, OrderedDict
from math import pi

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia.geometry.transform import warp_perspective
from tqdm import tqdm

from src.utils.load_model import load_model
from src.utils.metrics2 import error_auc
from src.utils.plotting2 import dynamic_alpha, make_matching_figure2
from src.utils.sample_h import sample_homography


def make_matching_plot_fast(img1, img2, mkpts0, mkpts1, mkpts0_f, mkpts1_f, color, text, show_keypoints=True,
                            mask=None):
    """
    Generates a visualization of image matches.
    """
    H1, W1, _ = img1.shape
    H2, W2, _ = img2.shape
    out = np.zeros((max(H1, H2), W1 + W2, 3), dtype=np.uint8)
    out[:H1, :W1, :] = img1
    out[:H2, W1:, :] = img2

    # Draw matches
    for i, (pt1, pt2, c) in enumerate(zip(mkpts0, mkpts1, color)):
        if mask[i] == 1:
            c = (0, 255, 0)
        else:
            c = (0, 0, 255)
        pt1 = (int(pt1[0]), int(pt1[1]))
        pt2 = (int(pt2[0] + W1), int(pt2[1]))
        cv2.line(out, pt1, pt2, color=(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)), thickness=1)

    if show_keypoints:
        for pt1, pt2 in zip(mkpts0_f, mkpts1_f):
            pt1 = (int(pt1[0]), int(pt1[1]))
            pt2 = (int(pt2[0] + W1), int(pt2[1]))
            cv2.circle(out, pt1, 2, (0, 255, 0), 1)
            cv2.circle(out, pt2, 2, (0, 255, 0), 1)

    scale = 1
    for i, t in enumerate(text):
        cv2.putText(out, t, (5, 40 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def save_matching_figure2(path, img0, img1, mkpts0, mkpts1, mean_distance, correct_mask, svg=False):
    """ Make and save matching figures
    """
    # bool---float
    correct_mask = correct_mask.astype(float)

    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = np.sum(correct_mask)
    n = mkpts0.shape[0]
    # matching info
    alpha = dynamic_alpha(len(correct_mask))
    mask = correct_mask
    color = np.zeros((n, 3), dtype=np.uint8)

    color[mask == 0] = (255, 0, 0)

    color[mask == 1] = (0, 255, 0)
    n_correct = int(n_correct)

    text_precision = [
        f'Precision(3px) ({100 * precision:.1f}%): {n_correct}/{len(mkpts0)}']

    # if name is not None:
    #     text = [name]
    # else:
    text = []

    error_text = [f"Mean Distance: {mean_distance:.2f} px"]
    text += error_text

    text += text_precision
    # color=
    # make the figure
    img1 = img1 / 255.  # float32
    figure = make_matching_figure2(img0, img1, mkpts0, mkpts1,
                                   color, text=text, path=path, dpi=100, svg=svg)


# def load_vis_depth_pairs_npz(multi_model_data_root, cvs_path):
#     data_root = multi_model_data_root
#
#     scene_pairs = {}
#     pairs = []
#     config = {
#         'perspective': True,  # Enable perspective transformation
#         'scaling': True,  # Enable scaling transformation
#         'rotation': True,  # Enable rotation transformation
#         'translation': True,  # Enable translation
#         'n_scales': 5,  # Number of scaling samples
#         'n_angles': 25,  # Number of rotation samples
#         'scaling_amplitude': 0.05,
#         'perspective_amplitude_x': 0.05,
#         'perspective_amplitude_y': 0.05,
#         'patch_ratio': 0.8,
#         'max_angle': 10 * (pi / 180),
#         'allow_artifacts': False,
#         'translation_overflow': 0.0
#     }
#     with open(cvs_path, mode='r', encoding='utf-8') as file:
#         reader = csv.reader(file)
#         for row in reader:
#             path = row[0].replace('./', '')
#             image_path = os.path.join(data_root, path)
#             depth_path = image_path.replace('/outdoor/', '/outdoor_depth/')
#
#             image0_shape = [480, 640]  # we fix the image size to 480x640
#             image1_shape = [480, 640]
#             H0 = sample_homography(image0_shape, image_path, config).squeeze().numpy()
#
#             H1 = sample_homography(image1_shape, depth_path, config).squeeze().numpy()
#
#             pairs.append({'im0': image_path, 'im1': depth_path, 'H': H0})
#             pairs.append({'im0': depth_path, 'im1': image_path, 'H': H1})
#     # pairs.sort()
#     scene_pairs['DIO-depth'] = pairs
#
#     return scene_pairs


def load_aligned_image_pairs(folder1, folder2, csv_path=None):
    """
    加载两个文件夹中的对齐图像对，并生成随机单应性矩阵进行变换

    Args:
        folder1: 第一个图像文件夹路径
        folder2: 第二个图像文件夹路径
        csv_path: 可选的CSV文件路径，包含要处理的图像对列表

    Returns:
        scene_pairs: 包含图像对和随机单应性矩阵的字典
    """
    scene_pairs = {}
    pairs = []

    # 配置单应性矩阵采样参数
    config = {
        'perspective': True,  # Enable perspective transformation
        'scaling': True,  # Enable scaling transformation
        'rotation': True,  # Enable rotation transformation
        'translation': True,  # Enable translation
        'n_scales': 5,  # Number of scaling samples
        'n_angles': 25,  # Number of rotation samples
        'scaling_amplitude': 0.05,
        'perspective_amplitude_x': 0.05,
        'perspective_amplitude_y': 0.05,
        'patch_ratio': 0.8,
        'max_angle': 10 * (pi / 180),
        'allow_artifacts': False,
        'translation_overflow': 0.0
    }

    if csv_path:
        # 使用CSV文件中的图像对列表
        with open(csv_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) > 0:
                    image_name = row[0].strip()
                    image_path1 = osp.join(folder1, image_name)
                    image_path2 = osp.join(folder2, image_name)

                    # 检查文件是否存在
                    if osp.exists(image_path1) and osp.exists(image_path2):
                        # 为每对图像生成两个随机变换（双向）
                        image_shape = [480, 640]  # 固定图像大小

                        # 对第一张图像应用变换
                        H1 = sample_homography(image_shape, image_path1, config).squeeze().numpy()
                        pairs.append({'im0': image_path1, 'im1': image_path2, 'H': H1})

                        # 对第二张图像应用变换
                        H2 = sample_homography(image_shape, image_path2, config).squeeze().numpy()
                        pairs.append({'im0': image_path2, 'im1': image_path1, 'H': H2})
    else:
        # 自动匹配两个文件夹中同名的图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

        # 获取文件夹1中的图像文件
        images1 = []
        for f in os.listdir(folder1):
            if any(f.lower().endswith(ext) for ext in image_extensions):
                images1.append(f)

        # 获取文件夹2中的图像文件
        images2 = []
        for f in os.listdir(folder2):
            if any(f.lower().endswith(ext) for ext in image_extensions):
                images2.append(f)

        # 找到共同的文件名
        common_images = set(images1) & set(images2)

        # 创建图像对并生成随机变换
        for image_name in common_images:
            image_path1 = osp.join(folder1, image_name)
            image_path2 = osp.join(folder2, image_name)

            # 为每对图像生成两个随机变换（双向）
            image_shape = [480, 640]  # 固定图像大小

            # 对第一张图像应用变换
            H1 = sample_homography(image_shape, image_path1, config).squeeze().numpy()
            pairs.append({'im0': image_path1, 'im1': image_path2, 'H': H1})

            # 对第二张图像应用变换
            H2 = sample_homography(image_shape, image_path2, config).squeeze().numpy()
            pairs.append({'im0': image_path2, 'im1': image_path1, 'H': H2})

    scene_pairs['DIO-depth'] = pairs

    return scene_pairs


def H_transform(img2_tensor, homography):
    image_shape = img2_tensor.shape[2:]
    img2_tensor = warp_perspective(img2_tensor, homography, image_shape, align_corners=True)

    return img2_tensor


def aggregiate_scenes(scene_pose_auc, thresholds):
    """Averages the auc results for cloudy_cloud and cloudy_sunny scenes
    """
    temp_pose_auc = {}
    for npz_name in scene_pose_auc.keys():
        scene_name = npz_name.split("_scene")[0]
        temp_pose_auc[scene_name] = [np.zeros(len(thresholds), dtype=np.float32), 0]  # [sum, total_number]
    for npz_name in scene_pose_auc.keys():
        scene_name = npz_name.split("_scene")[0]
        temp_pose_auc[scene_name][0] += scene_pose_auc[npz_name]
        temp_pose_auc[scene_name][1] += 1

    agg_pose_auc = {}
    for scene_name in temp_pose_auc.keys():
        agg_pose_auc[scene_name] = temp_pose_auc[scene_name][0] / temp_pose_auc[scene_name][1]

    return agg_pose_auc


def order_corners(corners):
    rect = np.zeros((4, 2), dtype="float32")
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]

    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]

    return rect


def draw_homography_comparison(image1, image2, real_warped_corners, warped_corners, mean_dist, file_name, save_path,
                               method):
    # Ensure the input image is in the correct format
    border_size = 100
    image2 = expand_image(image2, border_size)
    image1 = expand_image(image1, border_size)
    real_warped_corners += border_size
    warped_corners += border_size
    if image1.dtype != np.uint8:
        image1 = cv2.convertScaleAbs(image1)
    if image2.dtype != np.uint8:
        image2 = cv2.convertScaleAbs(image2)
    real_warped_corners = np.array(real_warped_corners, dtype=np.int32)
    warped_corners = np.array(warped_corners, dtype=np.int32)

    real_warped_corners = real_warped_corners.reshape((-1, 1, 2))
    warped_corners = warped_corners.reshape((-1, 1, 2))

    # combined_image = np.hstack((image1, image2))
    combined_image = np.hstack((image2, image1))

    combined_image = cv2.polylines(combined_image, [real_warped_corners], isClosed=True, color=(0, 255, 0),
                                   thickness=2)
    combined_image = cv2.polylines(combined_image, [warped_corners], isClosed=True, color=(0, 0, 255),
                                   thickness=2)

    plt.figure(figsize=(12, 6))
    combined_image = combined_image.astype('uint8')
    plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Homography Comparison ({method} Mean Distance: {mean_dist:.2f})')
    plt.axis('off')

    save_full_path = os.path.join(save_path, f"{file_name}_homography_comparison.png")
    plt.savefig(save_full_path)
    plt.close()
    print(f"Saved homography comparison image to: {save_full_path}")


def expand_image(image, border_size):
    return cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT,
                              value=[0, 0, 0])


def ensure_three_channels(image):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def compute_mean_distance(real_H, pred_H, H, W, visualize=False, save_path=None, file_name=None, image1=None,
                          image2=None, method=None):
    corners = np.array([[0, 0, 1],
                        [W - 1, 0, 1],
                        [0, H - 1, 1],
                        [W - 1, H - 1, 1]])
    real_H = np.linalg.inv(real_H)
    pred_H = np.linalg.inv(pred_H)
    # Compute warped corners using both estimated and real homographies
    real_warped_corners = np.dot(corners, np.transpose(real_H))
    real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
    warped_corners = np.dot(corners, np.transpose(pred_H))
    warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
    # Order corners to form a rectangle
    real_warped_corners = order_corners(real_warped_corners)
    warped_corners = order_corners(warped_corners)
    mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
    if visualize:
        draw_homography_comparison(image2, image1, real_warped_corners, warped_corners, mean_dist, file_name, save_path,
                                   method)
    return mean_dist


def compute_homography_match_statistics(
        mkpts0,
        mkpts1,
        real_H,
        thresholds=(1, 3, 5),
):
    """
    Compute correspondence statistics using the ground-truth homography.

    A match is regarded as correct when its symmetric reprojection error is
    no larger than the specified pixel threshold.

    Returns:
        dict containing:
            num_matches
            symmetric_errors
            correct_masks
            correct_counts
            precisions
    """
    mkpts0 = np.asarray(mkpts0, dtype=np.float64).reshape(-1, 2)
    mkpts1 = np.asarray(mkpts1, dtype=np.float64).reshape(-1, 2)
    thresholds = tuple(float(t) for t in thresholds)

    num_matches = int(len(mkpts0))

    empty_masks = {
        str(int(t) if float(t).is_integer() else t):
            np.zeros((0,), dtype=bool)
        for t in thresholds
    }

    if num_matches == 0:
        return {
            "num_matches": 0,
            "symmetric_errors": np.empty((0,), dtype=np.float64),
            "correct_masks": empty_masks,
            "correct_counts": {
                key: 0 for key in empty_masks
            },
            "precisions": {
                key: 0.0 for key in empty_masks
            },
        }

    if len(mkpts1) != num_matches:
        raise ValueError(
            "mkpts0 and mkpts1 must contain the same number of points, "
            f"but got {num_matches} and {len(mkpts1)}."
        )

    real_H = np.asarray(real_H, dtype=np.float64).reshape(3, 3)

    try:
        inv_H = np.linalg.inv(real_H)
    except np.linalg.LinAlgError as exc:
        raise ValueError("The ground-truth homography is singular.") from exc

    ones = np.ones((num_matches, 1), dtype=np.float64)
    mkpts0_h = np.hstack([mkpts0, ones])
    mkpts1_h = np.hstack([mkpts1, ones])

    projected_01_h = (real_H @ mkpts0_h.T).T
    projected_10_h = (inv_H @ mkpts1_h.T).T

    eps = np.finfo(np.float64).eps
    valid_01 = np.abs(projected_01_h[:, 2]) > eps
    valid_10 = np.abs(projected_10_h[:, 2]) > eps
    valid = valid_01 & valid_10

    projected_01 = np.full((num_matches, 2), np.inf, dtype=np.float64)
    projected_10 = np.full((num_matches, 2), np.inf, dtype=np.float64)

    projected_01[valid_01] = (
        projected_01_h[valid_01, :2]
        / projected_01_h[valid_01, 2:3]
    )
    projected_10[valid_10] = (
        projected_10_h[valid_10, :2]
        / projected_10_h[valid_10, 2:3]
    )

    error_01 = np.linalg.norm(projected_01 - mkpts1, axis=1)
    error_10 = np.linalg.norm(projected_10 - mkpts0, axis=1)
    symmetric_errors = 0.5 * (error_01 + error_10)
    symmetric_errors[~valid] = np.inf

    correct_masks = {}
    correct_counts = {}
    precisions = {}

    for threshold in thresholds:
        key = str(
            int(threshold)
            if float(threshold).is_integer()
            else threshold
        )
        mask = symmetric_errors <= threshold
        count = int(mask.sum())

        correct_masks[key] = mask
        correct_counts[key] = count
        precisions[key] = float(count / num_matches)

    return {
        "num_matches": num_matches,
        "symmetric_errors": symmetric_errors,
        "correct_masks": correct_masks,
        "correct_counts": correct_counts,
        "precisions": precisions,
    }


def compute_matching_accuracy(mkpts0, mkpts1, H):
    """
    Backward-compatible wrapper.

    Returns Precision@1px, Precision@3px and Precision@5px, followed by the
    total number of predicted matches.
    """
    stats = compute_homography_match_statistics(
        mkpts0,
        mkpts1,
        H,
        thresholds=(1, 3, 5),
    )

    accuracies = [
        stats["precisions"]["1"],
        stats["precisions"]["3"],
        stats["precisions"]["5"],
    ]
    return accuracies, stats["num_matches"]


def compute_mask(real_H, mkpts02, mkpts12, threshold=3):
    """Return the GT-correct mask under symmetric homography error."""
    stats = compute_homography_match_statistics(
        mkpts02,
        mkpts12,
        real_H,
        thresholds=(threshold,),
    )
    key = str(
        int(threshold)
        if float(threshold).is_integer()
        else float(threshold)
    )
    return stats["correct_masks"][key]


def plot2(file_name, scene_dir, img0, img1, mkpts0, mkpts1, color, method, mask1, number_inliers, number_mask,
          mean_dist):
    save_path_cv = os.path.join(scene_dir, f'{file_name}_match_result_cv_{method}.png')
    output_image = make_matching_plot_fast(
        img0, img1, mkpts0, mkpts1, mkpts0, mkpts1, color,
        text=['Mean Distance: {:.2f}'.format(mean_dist),
              '# Matches: {}'.format(number_inliers),
              f'Precision(3px)({100 * number_mask / number_inliers:.1f}%): {number_mask} / {number_inliers}', ],
        show_keypoints=False, mask=mask1
    )
    cv2.imwrite(save_path_cv, output_image)


def eval_relapose(
        matcher,
        scene_pairs,
        save_figs,
        figures_dir=None,
        method=None,
        print_out=False,
        debug=False,
        read_color=False,
        match_thresholds=(1, 3, 5),
        max_save_figs=10,
        svg=False,
):
    """
    Evaluate homography estimation and correspondence quality.

    Matching statistics are computed for every evaluated image pair directly
    from the predicted correspondences and the ground-truth homography. They
    do not depend on whether RANSAC succeeds.
    """
    scene_pose_auc = {}
    scene_match_stats = {}

    # Retained for compatibility with the existing caller.
    precs = {}
    precs_no_inlier = {}

    for scene_name in scene_pairs.keys():
        scene_dir = osp.join(
            figures_dir,
            scene_name.split(".")[0],
        )
        if (
            save_figs
            and max_save_figs > 0
            and not osp.exists(scene_dir)
        ):
            os.makedirs(scene_dir)

        statis = defaultdict(list)
        groups = scene_pairs[scene_name]

        logging.info("\nStart evaluation on homography pairs\n")

        for i, pair in tqdm(
            enumerate(groups),
            smoothing=.1,
            total=len(groups),
        ):
            if debug and i > 10:
                break

            im0_path = pair["im0"]
            im1_path = pair["im1"]
            real_H = pair["H"]

            if read_color:
                im0 = cv2.imread(im0_path, cv2.IMREAD_COLOR)
                im1 = cv2.imread(im1_path, cv2.IMREAD_COLOR)

                if im0 is None or im1 is None:
                    raise FileNotFoundError(
                        f"Failed to read image pair: "
                        f"{im0_path}, {im1_path}"
                    )

                im0 = cv2.resize(im0, (640, 480))
                im1 = cv2.resize(im1, (640, 480))

                im1_tensor = (
                    torch.tensor(
                        im1,
                        dtype=torch.float32,
                    )
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    / 255.0
                )
                real_H_tensor = torch.tensor(
                    real_H,
                    dtype=torch.float32,
                ).unsqueeze(0)

                im1_tensor = H_transform(
                    im1_tensor,
                    real_H_tensor,
                )
                im1 = (
                    im1_tensor
                    .squeeze()
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                    * 255.0
                )
                real_H = (
                    real_H_tensor
                    .squeeze()
                    .cpu()
                    .numpy()
                )
            else:
                im0 = cv2.imread(
                    im0_path,
                    cv2.IMREAD_GRAYSCALE,
                )
                im1 = cv2.imread(
                    im1_path,
                    cv2.IMREAD_GRAYSCALE,
                )

                if im0 is None or im1 is None:
                    raise FileNotFoundError(
                        f"Failed to read image pair: "
                        f"{im0_path}, {im1_path}"
                    )

                im0 = cv2.resize(im0, (640, 480))
                im1 = cv2.resize(im1, (640, 480))

                im1_tensor = (
                    torch.tensor(
                        im1,
                        dtype=torch.float32,
                    )
                    .unsqueeze(0)
                    .unsqueeze(0)
                    / 255.0
                )
                real_H_tensor = torch.tensor(
                    real_H,
                    dtype=torch.float32,
                ).unsqueeze(0)

                im1_tensor = H_transform(
                    im1_tensor,
                    real_H_tensor,
                )
                im1 = (
                    im1_tensor
                    .squeeze()
                    .cpu()
                    .numpy()
                    * 255.0
                )
                real_H = (
                    real_H_tensor
                    .squeeze()
                    .cpu()
                    .numpy()
                )

            match_res = matcher(im0, im1)

            matches = match_res["matches"]
            mkpts0 = np.asarray(
                match_res["mkpts0"],
                dtype=np.float64,
            ).reshape(-1, 2)
            mkpts1 = np.asarray(
                match_res["mkpts1"],
                dtype=np.float64,
            ).reshape(-1, 2)
            img0 = match_res["img0"]
            img1 = match_res["img1"]

            if len(img0.shape) == 2:
                H, W = img0.shape
                img0 = cv2.cvtColor(
                    img0,
                    cv2.COLOR_GRAY2BGR,
                )
                img1 = cv2.cvtColor(
                    img1,
                    cv2.COLOR_GRAY2BGR,
                )
            else:
                H, W, _ = img0.shape

            img0_name = (
                f"{'vis' if 'depth' in im0_path else 'depth'}_"
                f"{osp.basename(im0_path).split('.')[0]}"
            )
            img1_name = (
                f"{'vis' if 'depth' in im1_path else 'depth'}_"
                f"{osp.basename(im1_path).split('.')[0]}"
            )
            file_name = f"{img0_name}_{img1_name}"

            # -----------------------------------------------------
            # Reviewer-requested correspondence statistics.
            # These are always computed, even when RANSAC fails.
            # -----------------------------------------------------
            pair_stats = compute_homography_match_statistics(
                mkpts0,
                mkpts1,
                real_H,
                thresholds=match_thresholds,
            )

            statis["num_matches"].append(
                pair_stats["num_matches"]
            )

            for threshold in match_thresholds:
                key = str(
                    int(threshold)
                    if float(threshold).is_integer()
                    else float(threshold)
                )
                statis[
                    f"num_correct_{key}px"
                ].append(
                    pair_stats["correct_counts"][key]
                )
                statis[
                    f"precision_{key}px"
                ].append(
                    pair_stats["precisions"][key]
                )

            # -----------------------------------------------------
            # Homography estimation and corner reprojection error.
            # -----------------------------------------------------
            ret_H = None
            inliers = None
            mean_dist = np.inf

            if len(mkpts0) >= 4:
                ret_H, inliers = cv2.findHomography(
                    mkpts0,
                    mkpts1,
                    cv2.RANSAC,
                )

                if ret_H is not None:
                    if np.linalg.matrix_rank(ret_H) < 3:
                        ret_H = None
                    else:
                        mean_dist = compute_mean_distance(
                            real_H,
                            ret_H,
                            H,
                            W,
                            visualize=(
                                save_figs
                                and max_save_figs > 0
                                and i < max_save_figs
                            ),
                            save_path=scene_dir,
                            file_name=file_name,
                            image1=img0,
                            image2=img1,
                            method=method,
                        )

            if ret_H is None:
                statis["mean_dist"].append(np.inf)
                statis["failed"].append(i)
                ransac_inlier_ratio = 0.0
            else:
                statis["mean_dist"].append(mean_dist)
                ransac_inlier_ratio = (
                    float(np.asarray(inliers).sum())
                    / pair_stats["num_matches"]
                    if (
                        inliers is not None
                        and pair_stats["num_matches"] > 0
                    )
                    else 0.0
                )

            statis["ransac_inlier_ratio"].append(
                ransac_inlier_ratio
            )

            if print_out:
                precision_text = ", ".join(
                    (
                        f"P@{threshold:g}px="
                        f"{100.0 * pair_stats['precisions'][str(int(threshold) if float(threshold).is_integer() else float(threshold))]:.2f}%"
                    )
                    for threshold in match_thresholds
                )
                logging.info(
                    f"#M={pair_stats['num_matches']} "
                    f"{precision_text} "
                    f"RANSAC-IR="
                    f"{100.0 * ransac_inlier_ratio:.2f}% "
                    f"corner_error={mean_dist:.3f}"
                )

            # Save at most the first max_save_figs pairs per scene.
            if (
                save_figs
                and max_save_figs > 0
                and i < max_save_figs
            ):
                mask_3px = pair_stats[
                    "correct_masks"
                ].get(
                    "3",
                    compute_mask(
                        real_H,
                        mkpts0,
                        mkpts1,
                        threshold=3,
                    ),
                )

                fig_path = osp.join(
                    scene_dir,
                    f"{img0_name}_{img1_name}_{method}.jpg",
                )
                save_matching_figure2(
                    path=fig_path,
                    img0=img0,
                    img1=img1,
                    mkpts0=mkpts0,
                    mkpts1=mkpts1,
                    mean_distance=mean_dist,
                    correct_mask=mask_3px,
                    svg=svg,
                )

        num_pairs = len(statis["num_matches"])
        denom = max(num_pairs, 1)

        mean_dist_all = np.asarray(
            statis["mean_dist"],
            dtype=np.float64,
        )
        auc_thresholds = [1, 3, 5, 7, 10, 15, 20]
        homography_auc = error_auc(
            mean_dist_all,
            auc_thresholds,
        )

        finite_mean_dist = mean_dist_all[
            np.isfinite(mean_dist_all)
        ]
        average_mean_dist = (
            float(np.mean(finite_mean_dist))
            if len(finite_mean_dist) > 0
            else float("inf")
        )

        scene_stats = {
            "num_pairs": int(num_pairs),
            "failed_pairs": int(len(statis["failed"])),
            "avg_matches": float(
                np.sum(statis["num_matches"]) / denom
            ),
            "avg_ransac_inlier_ratio": float(
                np.sum(statis["ransac_inlier_ratio"])
                / denom
            ),
            "average_corner_error": average_mean_dist,
        }

        for threshold in match_thresholds:
            key = str(
                int(threshold)
                if float(threshold).is_integer()
                else float(threshold)
            )
            correct_key = f"num_correct_{key}px"
            precision_key = f"precision_{key}px"

            total_correct = float(
                np.sum(statis[correct_key])
            )
            total_matches = float(
                np.sum(statis["num_matches"])
            )

            scene_stats[
                f"avg_correct_matches_{key}px"
            ] = total_correct / denom

            # Macro average: first compute each pair's precision,
            # then average over all evaluated pairs.
            scene_stats[
                f"avg_precision_{key}px"
            ] = float(
                np.sum(statis[precision_key]) / denom
            )

            # Micro average: pool all matches across the dataset.
            scene_stats[
                f"pooled_precision_{key}px"
            ] = (
                total_correct / total_matches
                if total_matches > 0
                else 0.0
            )

        scene_match_stats[scene_name] = scene_stats

        logging.info(
            f"Scene: {scene_name} "
            f"Evaluated: {num_pairs} "
            f"Failed: {len(statis['failed'])}"
        )
        logging.info(
            "Avg. Matches: %.4f",
            scene_stats["avg_matches"],
        )

        for threshold in match_thresholds:
            key = str(
                int(threshold)
                if float(threshold).is_integer()
                else float(threshold)
            )
            logging.info(
                (
                    "Threshold %spx | "
                    "Avg. Correct Matches: %.4f | "
                    "Avg. Precision: %.4f%% | "
                    "Pooled Precision: %.4f%%"
                ),
                key,
                scene_stats[
                    f"avg_correct_matches_{key}px"
                ],
                100.0 * scene_stats[
                    f"avg_precision_{key}px"
                ],
                100.0 * scene_stats[
                    f"pooled_precision_{key}px"
                ],
            )

        logging.info(
            "Avg. RANSAC Inlier Ratio: %.4f%%",
            100.0 * scene_stats[
                "avg_ransac_inlier_ratio"
            ],
        )
        logging.info(
            "Average finite corner error: %s",
            average_mean_dist,
        )

        # Concise summary for the paper statistics (3 px is the main threshold).
        key3 = "3"
        if f"avg_correct_matches_{key3}px" in scene_stats:
            summary = (
                f"[{scene_name}] "
                f"Avg. Matches={scene_stats['avg_matches']:.2f}, "
                f"Avg. Correct Matches@3px="
                f"{scene_stats[f'avg_correct_matches_{key3}px']:.2f}, "
                f"CMR@3px="
                f"{100.0 * scene_stats[f'avg_precision_{key3}px']:.2f}%, "
                f"RANSAC-IR="
                f"{100.0 * scene_stats['avg_ransac_inlier_ratio']:.2f}%"
            )
            print(summary)
            logging.info(summary)

        scene_pose_auc[scene_name] = 100 * np.array(
            [
                homography_auc[f"auc@{t}"]
                for t in auc_thresholds
            ]
        )
        logging.info(
            "%s %s",
            scene_name,
            homography_auc,
        )

        # Preserve the former return structure with meaningful values.
        precs[scene_name] = {
            f"{threshold:g}px": scene_stats[
                f"avg_precision_{str(int(threshold) if float(threshold).is_integer() else float(threshold))}px"
            ]
            for threshold in match_thresholds
        }
        precs_no_inlier[scene_name] = {
            f"{threshold:g}px": scene_stats[
                f"pooled_precision_{str(int(threshold) if float(threshold).is_integer() else float(threshold))}px"
            ]
            for threshold in match_thresholds
        }

    auc_thresholds = [1, 3, 5, 7, 10, 15, 20]
    agg_pose_auc = aggregiate_scenes(
        scene_pose_auc,
        auc_thresholds,
    )
    agg_precs, agg_precs_no_inlier = aggregate_precisions(
        precs,
        precs_no_inlier,
    )

    return (
        scene_pose_auc,
        agg_pose_auc,
        precs,
        precs_no_inlier,
        agg_precs,
        agg_precs_no_inlier,
        scene_match_stats,
    )


def aggregate_precisions(precs, precs_no_inlier):
    """Aggregate precision values across cloudy_cloud and cloudy_sunny scenes."""
    temp_precs = defaultdict(lambda: defaultdict(list))
    temp_precs_no_inlier = defaultdict(lambda: defaultdict(list))

    for scene_name, precision_dict in precs.items():
        main_scene = scene_name.split("_scene")[0]
        for threshold, precision in precision_dict.items():
            temp_precs[main_scene][threshold].append(precision)

    for scene_name, precision_dict in precs_no_inlier.items():
        main_scene = scene_name.split("_scene")[0]
        for threshold, precision in precision_dict.items():
            temp_precs_no_inlier[main_scene][threshold].append(precision)

    agg_precs = {scene: {threshold: np.mean(values) for threshold, values in thresholds_dict.items()}
                 for scene, thresholds_dict in temp_precs.items()}

    agg_precs_no_inlier = {scene: {threshold: np.mean(values) for threshold, values in thresholds_dict.items()}
                           for scene, thresholds_dict in temp_precs_no_inlier.items()}

    return agg_precs, agg_precs_no_inlier


def test_relative_pose_visdepth(
        method="xoftr",
        exp_name="VisDepth",
        ransac_thres=1.5,
        print_out=False,
        save_dir=None,
        save_figs=False,
        debug=False,
        args=None

):
    # save_dir = osp.join(save_dir, time)

    if method == "roma":
        if args.ckpt is None:
            save_ = "roma"
        else:
            save_ = args.ckpt.split("/")[-1].replace(".ckpt", "")
    else:
        save_ = args.ckpt.split("/")[-1].replace(".ckpt", "")
    path_ = osp.join(save_dir, method, save_)
    read_color = False
    if method == 'roma':
        read_color = True
    if args.debug:
        path_ = osp.join(save_dir, method, save_, "debug")
    if not osp.exists(path_):
        os.makedirs(path_)

    counter = 0
    if hasattr(args, 'thr'):
        path = osp.join(path_, f"{exp_name}_thresh_{args.thr}" + "_{}")
    else:
        path = osp.join(path_, f"{exp_name}" + "_{}")
    while osp.exists(path.format(counter)):
        counter += 1
    exp_dir = path.format(counter)
    os.mkdir(exp_dir)
    results_file = osp.join(exp_dir, "results.json")
    logging.basicConfig(
        filename=results_file.replace('.json', '.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    figures_dir = osp.join(exp_dir, "match_figures")
    if save_figs and args.max_save_figs > 0:
        os.mkdir(figures_dir)

    # Log args
    logging.info(f"args: {args}")

    # Init paths
    # multi_model_data_root = "./data/DIODE/"
    cvs_path = None
    folder1 = "./data/MSRS/visible"

    folder2 = "./data/MSRS/infrared"
    # Load pairs
    scene_pairs = load_aligned_image_pairs(folder1, folder2, cvs_path)

    # Load method
    # matcher = eval(f"load_{method}")(args)
    matcher = load_model(method, args, use_path=False)
    thresholds = [5, 10, 20]
    # Eval
    (
        scene_pose_auc,
        agg_pose_auc,
        precs,
        precs_no_inlier,
        agg_precs,
        agg_precs_no_inlier,
        scene_match_stats,
    ) = eval_relapose(
        matcher,
        scene_pairs,
        save_figs=save_figs,
        figures_dir=figures_dir,
        method=method,
        print_out=print_out,
        debug=debug,
        read_color=read_color,
        match_thresholds=(1, 3, 5),
        max_save_figs=args.max_save_figs,
        svg=args.svg,
    )

    # Create result dict
    results = OrderedDict({"method": method,
                           "exp_name": exp_name,
                           "ransac_thres": ransac_thres,
                           "auc_thresholds": thresholds})
    results.update({key: value for key, value in vars(args).items() if key not in results})
    results.update({key: value.tolist() for key, value in agg_pose_auc.items()})
    results.update({key: value.tolist() for key, value in scene_pose_auc.items()})

    results.update({f"precs_{key}": value for key, value in precs.items()})

    results.update({f"precs_no_inlier_{key}": value for key, value in precs_no_inlier.items()})

    results.update({f"agg_precs_{key}": value for key, value in agg_precs.items()})

    results.update({f"agg_precs_no_inlier_{key}": value for key, value in agg_precs_no_inlier.items()})

    # Reviewer-requested per-pair correspondence statistics.
    results["scene_match_statistics"] = scene_match_stats

    # Save a compact CSV that can be used directly to prepare the paper table.
    match_stats_csv = osp.join(exp_dir, "match_statistics.csv")
    fieldnames = [
        "scene",
        "num_pairs",
        "failed_pairs",
        "avg_matches",
        "avg_correct_matches_1px",
        "avg_cmr_1px",
        "avg_correct_matches_3px",
        "avg_cmr_3px",
        "avg_correct_matches_5px",
        "avg_cmr_5px",
        "avg_ransac_inlier_ratio",
        "average_corner_error",
    ]

    with open(match_stats_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for scene, stats in scene_match_stats.items():
            writer.writerow({
                "scene": scene,
                "num_pairs": stats.get("num_pairs", 0),
                "failed_pairs": stats.get("failed_pairs", 0),
                "avg_matches": stats.get("avg_matches", 0.0),
                "avg_correct_matches_1px": stats.get("avg_correct_matches_1px", 0.0),
                "avg_cmr_1px": stats.get("avg_precision_1px", 0.0),
                "avg_correct_matches_3px": stats.get("avg_correct_matches_3px", 0.0),
                "avg_cmr_3px": stats.get("avg_precision_3px", 0.0),
                "avg_correct_matches_5px": stats.get("avg_correct_matches_5px", 0.0),
                "avg_cmr_5px": stats.get("avg_precision_5px", 0.0),
                "avg_ransac_inlier_ratio": stats.get("avg_ransac_inlier_ratio", 0.0),
                "average_corner_error": stats.get("average_corner_error", float("inf")),
            })

    logging.info(f"Matching statistics saved to {match_stats_csv}")

    logging.info(f"Results: {json.dumps(results, indent=4)}")

    # Save to json file
    with open(results_file, 'w') as outfile:
        json.dump(results, outfile, indent=4)

    logging.info(f"Results saved to {results_file}")


if __name__ == '__main__':
    # IMPORTANT: keep the sampled homographies identical across methods.
    torch.manual_seed(42)
    np.random.seed(42)
    import random
    random.seed(42)

    def add_common_arguments(parser):
        parser.add_argument('--exp_name', type=str, default="VisDepth")
        parser.add_argument('--save_dir', type=str, default="./isprs_ablation/homo/MSRS")
        parser.add_argument('--e_name', type=str, default=None)
        parser.add_argument('--ransac_thres', type=float, default=1.5)
        parser.add_argument('--print_out', action='store_true')
        parser.add_argument('--debug', action='store_true')
        parser.add_argument('--save_figs', action='store_true')
        parser.add_argument(
            '--max_save_figs',
            type=int,
            default=0,
            help='Maximum matching figures saved per scene; 0 disables saving.',
        )
        parser.add_argument('--svg', action='store_true')


    def add_method_arguments(parser, method):
        if method == "xoftr":
            parser.add_argument('--match_threshold', type=float, default=0.3)
            parser.add_argument('--fine_threshold', type=float, default=0.1)
            # parser.add_argument('--ckpt', type=str, default="./logs/tb_logs/dinomatch/version_32/checkpoints/epoch=27-auc@5=0.206-auc@10=0.378-auc@20=0.554.ckpt")
            parser.add_argument('--ckpt', type=str,
                                default="./logs/tb_logs/dinomatch/version_32/checkpoints/epoch=27-auc@5=0.206-auc@10=0.378-auc@20=0.554.ckpt")
            # parser.add_argument('--ckpt', type=str, default="./weights/minima_xoftr.ckpt")

        elif method == "loftr":
            parser.add_argument('--ckpt', type=str,
                                default="./weights/minima_loftr.ckpt")
            parser.add_argument('--thr', type=float, default=0.2)
        elif method == "eloftr":
            parser.add_argument('--ckpt', type=str,
                                default="./weights/minima_eloftr.ckpt")
            # parser.add_argument('--ckpt', type=str,
            #                     default="./weights/eloftr_outdoor.ckpt")
            parser.add_argument('--thr', type=float, default=0.2)
        elif method == "sp_lg":
            # parser.add_argument('--ckpt', type=str, default="./weights/minima_lightglue.pth")
            parser.add_argument('--ckpt', type=str,
                                default="./weights/sp+lg_megadepth_weights.pth")
        elif method == "xfeat":
            parser.add_argument('--ckpt', type=str,
                                default="./weights/xfeat.pt")
        elif method == "reasoning_xfeat":
            parser.add_argument('--ckpt', type=str,
                                default="./weights/reasoning_xfeat.pt")
        elif method == "semaglue":
            parser.add_argument('--ckpt', type=str,
                                default="./weights/semaglue.pth")
        elif method == "omniglue":
            parser.add_argument('--ckpt', type=str,
                                default="./weights/omniglue_models")
        elif method == "roma":
            parser.add_argument('--ckpt2', type=str,
                                default="large")
            parser.add_argument('--ckpt', type=str, default='./weights/minima_roma.pth')

        else:
            raise ValueError(f"Unknown method: {method}")

        add_common_arguments(parser)


    parser = argparse.ArgumentParser(description='Benchmark Relative Pose')

    parser.add_argument('--method', type=str, default="xoftr",
                        choices=["xoftr", 'sp_lg', 'loftr', 'roma', 'eloftr', 'xfeat', 'omniglue', 'reasoning_xfeat',
                                 'semaglue'],
                        help="Select the method to use: xoftr, sp_lg, loftr, roma")

    args, remaining_args = parser.parse_known_args()

    add_method_arguments(parser, args.method)

    args = parser.parse_args()

    print(args)

    if args.e_name is not None:
        save_dir = osp.join(args.save_dir, args.e_name)
    else:
        save_dir = args.save_dir
# Set CUDA_VISIBLE_DEVICES in the shell when selecting GPUs.
    tt = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_relative_pose_visdepth(
            args.method,
            args.exp_name,
            ransac_thres=args.ransac_thres,
            print_out=args.print_out,
            save_dir=save_dir,
            save_figs=args.save_figs,
            debug=args.debug,
            args=args
        )
    print(f"Elapsed time: {time.time() - tt}")
