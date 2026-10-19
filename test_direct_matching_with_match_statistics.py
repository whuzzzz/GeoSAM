import csv
import json
import os
import re
import time
# Set CUDA_VISIBLE_DEVICES in the shell when selecting GPUs.

import cv2
import numpy as np

from src.xoftr import (
    XoFTR_dino_resnet_panmatch_light_withfpn,
)
from src.config.default import get_cfg_defaults
from src.utils.data_io import DataIOWrapper, lower_config


# ============================================================
# Utility functions
# ============================================================

def natural_sort(file_list):
    """Sort file names in natural numeric order."""
    return sorted(
        file_list,
        key=lambda x: [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r"([0-9]+)", x)
        ],
    )


def resize_image_and_points(img, points, target_long_edge=640):
    """Resize image by long edge and scale point coordinates accordingly."""
    height, width = img.shape[:2]
    long_edge = max(height, width)

    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)

    if long_edge <= 0:
        return img, points

    scale = float(target_long_edge) / float(long_edge)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized_img = cv2.resize(
        img,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )

    if len(points) == 0:
        scaled_points = np.empty((0, 2), dtype=np.float64)
    else:
        scaled_points = points * scale

    return resized_img, scaled_points


def draw_matches(
    img1_path,
    img2_path,
    keypoints1,
    keypoints2,
    path,
    target_long_edge=640,
):
    """
    Draw the RANSAC-consistent correspondences only.

    Left points: red hollow circles
    Right points: green crosses
    Connecting lines: yellow
    """
    keypoints1 = np.asarray(keypoints1, dtype=np.float64).reshape(-1, 2).copy()
    keypoints2 = np.asarray(keypoints2, dtype=np.float64).reshape(-1, 2).copy()

    img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)

    if img1 is None or img2 is None:
        raise FileNotFoundError(
            f"Cannot read image pair: {img1_path}, {img2_path}"
        )

    img1, keypoints1 = resize_image_and_points(
        img1,
        keypoints1,
        target_long_edge,
    )
    img2, keypoints2 = resize_image_and_points(
        img2,
        keypoints2,
        target_long_edge,
    )

    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    canvas = np.zeros(
        (max(height1, height2), width1 + width2, 3),
        dtype=np.uint8,
    )
    canvas[:height1, :width1] = img1
    canvas[:height2, width1:] = img2

    if len(keypoints1) > 0 and len(keypoints2) > 0:
        keypoints2_draw = keypoints2.copy()
        keypoints2_draw[:, 0] += width1

        matched_points = np.hstack((keypoints1, keypoints2_draw))

        for i in range(matched_points.shape[0]):
            pt1 = tuple(np.round(matched_points[i, :2]).astype(int))
            pt2 = tuple(np.round(matched_points[i, 2:]).astype(int))

            cv2.line(canvas, pt1, pt2, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(canvas, pt1, 5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.drawMarker(
                canvas,
                pt2,
                (0, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=10,
                thickness=1,
                line_type=cv2.LINE_AA,
            )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, canvas)


# ============================================================
# Matching statistics
# ============================================================

def compute_ransac_statistics(
    pts1,
    pts2,
    threshold=3.0,
):
    """
    Compute correspondence statistics using a RANSAC-estimated homography.

    IMPORTANT:
    This direct-matching dataset does not provide a GT homography in this script.
    Therefore, "correct matches" are operationally defined as RANSAC
    geometrically consistent inliers. This corresponds to the reviewer's
    suggested inlier ratio rather than a GT-based correct-match rate.

    Returns:
        num_matches: total predicted correspondences M_i
        num_inliers: RANSAC-consistent correspondences C_i
        ncm: alias of num_inliers for backward compatibility
        inlier_ratio: C_i / M_i (per-pair CMR/IR)
        rmse: RMSE of reprojection errors over RANSAC inliers
        inlier_mask: boolean mask over all predicted correspondences
        H: estimated homography
    """
    pts1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 2)
    pts2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 2)

    num_matches = int(len(pts1))

    empty = {
        "num_matches": num_matches,
        "num_inliers": 0,
        "ncm": 0,
        "inlier_ratio": 0.0,
        "rmse": None,
        "inlier_mask": np.zeros(num_matches, dtype=bool),
        "H": None,
    }

    if num_matches < 4 or len(pts2) != num_matches:
        return empty

    H, mask = cv2.findHomography(
        pts1,
        pts2,
        cv2.RANSAC,
        ransacReprojThreshold=float(threshold),
        maxIters=10000,
        confidence=0.999,
    )

    if H is None or mask is None:
        return empty

    if not np.isfinite(H).all() or np.linalg.matrix_rank(H) < 3:
        return empty

    mask = mask.ravel().astype(bool)

    pts1_h = np.hstack(
        [pts1, np.ones((num_matches, 1), dtype=np.float64)]
    )
    projected_h = (H @ pts1_h.T).T

    eps = np.finfo(np.float64).eps
    valid = np.abs(projected_h[:, 2]) > eps

    projected = np.full_like(pts1, np.inf, dtype=np.float64)
    projected[valid] = (
        projected_h[valid, :2] / projected_h[valid, 2:3]
    )

    errors = np.linalg.norm(projected - pts2, axis=1)
    valid_inlier_mask = mask & np.isfinite(errors)

    num_inliers = int(valid_inlier_mask.sum())
    inlier_ratio = (
        float(num_inliers) / float(num_matches)
        if num_matches > 0
        else 0.0
    )

    rmse = (
        float(
            np.sqrt(
                np.mean(errors[valid_inlier_mask] ** 2)
            )
        )
        if num_inliers > 0
        else None
    )

    return {
        "num_matches": num_matches,
        "num_inliers": num_inliers,
        "ncm": num_inliers,
        "inlier_ratio": inlier_ratio,
        "rmse": rmse,
        "inlier_mask": valid_inlier_mask,
        "H": H,
    }


# ============================================================
# Main evaluation
# ============================================================

if __name__ == "__main__":
    # --------------------------------------------------------
    # Dataset / output settings
    # --------------------------------------------------------
    L_folder = "./data/day_night/day"
    R_folder = "./data/day_night/night"

    output_root = (
        "./isprs_ablation/"
        "day_night"
    )
    os.makedirs(output_root, exist_ok=True)

    # --------------------------------------------------------
    # Evaluation settings
    # Keep these consistent with the reference implementation.
    # --------------------------------------------------------
    RANSAC_THRESHOLD = 3.0
    MIN_MATCHES = 20
    MIN_INLIERS = 20
    MIN_INLIER_RATIO = 0.20
    FAILURE_RMSE = 5.0

    SAVE_FIGS = True
    MAX_SAVE_FIGS = 20
    VIS_LONG_EDGE = 640

    # --------------------------------------------------------
    # File list
    # --------------------------------------------------------
    valid_ext = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

    L_files = natural_sort([
        f for f in os.listdir(L_folder)
        if f.lower().endswith(valid_ext)
    ])
    R_files = natural_sort([
        f for f in os.listdir(R_folder)
        if f.lower().endswith(valid_ext)
    ])

    if len(L_files) != len(R_files):
        raise ValueError(
            f"L/R image counts differ: {len(L_files)} vs {len(R_files)}"
        )

    if len(L_files) == 0:
        raise RuntimeError("No image pairs found.")

    # --------------------------------------------------------
    # GeoSAM / XoFTR-style matcher configuration
    # --------------------------------------------------------
    config = get_cfg_defaults(inference=True)
    config = lower_config(config)

    config["xoftr"]["match_coarse"]["thr"] = 0.3
    config["xoftr"]["fine"]["thr"] = 0.1
    config["xoftr"]["fine"]["denser"] = False

    # Keep the model definition exactly as in the original script.
    matcher = XoFTR_dino_resnet_panmatch_light_withfpn(
        config=config["xoftr"]
    )

    config["test"]["img0_resize"] = 640
    config["test"]["img1_resize"] = 640

    ckpt = (
        "./logs/tb_logs/dinomatch/"
        "version_32/checkpoints/"
        "epoch=27-auc@5=0.206-auc@10=0.378-auc@20=0.554.ckpt"
    )

    matcher = DataIOWrapper(
        matcher,
        config=config["test"],
        ckpt=ckpt,
    )
    matcher = matcher.eval()

    # --------------------------------------------------------
    # Accumulators
    # --------------------------------------------------------
    total_matches = 0
    total_correct_matches = 0
    sum_pair_cmr = 0.0
    total_rmse = 0.0
    successful_pairs = 0

    per_pair = []
    start = time.time()

    figure_dir = os.path.join(output_root, "match_figures")

    # --------------------------------------------------------
    # Evaluate all pairs
    # --------------------------------------------------------
    for idx, (L_file, R_file) in enumerate(zip(L_files, R_files)):
        img0_path = os.path.join(L_folder, L_file)
        img1_path = os.path.join(R_folder, R_file)

        output_data = matcher.from_paths(
            img0_path,
            img1_path,
            read_color=True,
        )

        mkpts0 = np.asarray(
            output_data.get("mkpts0", []),
            dtype=np.float64,
        ).reshape(-1, 2)
        mkpts1 = np.asarray(
            output_data.get("mkpts1", []),
            dtype=np.float64,
        ).reshape(-1, 2)

        stats = compute_ransac_statistics(
            mkpts0,
            mkpts1,
            threshold=RANSAC_THRESHOLD,
        )

        num_matches = stats["num_matches"]
        num_correct = stats["num_inliers"]
        cmr = stats["inlier_ratio"]
        raw_rmse = stats["rmse"]

        # Accumulate reviewer-requested per-pair statistics.
        total_matches += num_matches
        total_correct_matches += num_correct
        sum_pair_cmr += cmr

        # Same failure-penalty logic as the reference code.
        if raw_rmse is None or num_correct < MIN_INLIERS:
            rmse = float(FAILURE_RMSE)
            rmse_penalized = True
        else:
            rmse = float(raw_rmse)
            rmse_penalized = False

        success = (
            num_matches >= MIN_MATCHES
            and num_correct > MIN_INLIERS
            and cmr > MIN_INLIER_RATIO
        )

        successful_pairs += int(success)
        total_rmse += rmse

        print(
            f"[{idx + 1}/{len(L_files)}] "
            f"{L_file} <-> {R_file} | "
            f"#Matches={num_matches}, "
            f"#Correct={num_correct}, "
            f"CMR/IR={100.0 * cmr:.2f}%, "
            f"RMSE={rmse:.4f}, "
            f"Success={success}"
        )

        per_pair.append({
            "image_index": idx + 1,
            "left_image": L_file,
            "right_image": R_file,
            "num_matches": int(num_matches),
            "num_correct_matches": int(num_correct),
            "NCM": int(num_correct),
            "correct_match_ratio": float(cmr),
            "ransac_inlier_ratio": float(cmr),
            "raw_inlier_rmse": (
                None if raw_rmse is None else float(raw_rmse)
            ),
            "RMSE": float(rmse),
            "rmse_penalized": bool(rmse_penalized),
            "successful_pair": bool(success),
        })

        if (
            SAVE_FIGS
            and MAX_SAVE_FIGS > 0
            and idx < MAX_SAVE_FIGS
        ):
            inlier_mask = stats["inlier_mask"]
            inlier_pts0 = mkpts0[inlier_mask]
            inlier_pts1 = mkpts1[inlier_mask]

            output_path = os.path.join(
                figure_dir,
                f"{idx + 1:04d}_{L_file}_{R_file}.png",
            )

            draw_matches(
                img0_path,
                img1_path,
                inlier_pts0,
                inlier_pts1,
                output_path,
                target_long_edge=VIS_LONG_EDGE,
            )

    # --------------------------------------------------------
    # Dataset-level statistics
    # --------------------------------------------------------
    num_pairs = len(per_pair)
    pair_denom = max(num_pairs, 1)
    match_denom = max(total_matches, 1)

    avg_matches = total_matches / pair_denom
    avg_correct_matches = total_correct_matches / pair_denom

    # Macro CMR: calculate C_i/M_i for each pair first, then average.
    # This matches the "per-pair" statistic requested by the reviewer.
    avg_cmr = sum_pair_cmr / pair_denom

    # Optional pooled/micro ratio, kept only for diagnostic purposes.
    pooled_cmr = total_correct_matches / match_denom

    avg_rmse = total_rmse / pair_denom
    success_rate = successful_pairs / pair_denom

    overall = {
        "num_pairs": int(num_pairs),
        "avg_matches": float(avg_matches),
        "avg_correct_matches": float(avg_correct_matches),
        "CMR": float(avg_cmr),
        "CMR_percent": float(100.0 * avg_cmr),
        "pooled_CMR": float(pooled_cmr),
        "pooled_CMR_percent": float(100.0 * pooled_cmr),
        # Backward-compatible names used in the old code/table.
        "avg_NCM": float(avg_correct_matches),
        "total_NCM": int(total_correct_matches),
        "avg_ransac_inlier_ratio": float(avg_cmr),
        "RMSE": float(avg_rmse),
        "failure_rmse": float(FAILURE_RMSE),
        "successful_pairs": int(successful_pairs),
        "success_rate": float(success_rate),
        "ransac_threshold_px": float(RANSAC_THRESHOLD),
        "success_definition": {
            "min_matches": int(MIN_MATCHES),
            "min_inliers_strictly_greater_than": int(MIN_INLIERS),
            "min_inlier_ratio_strictly_greater_than": float(MIN_INLIER_RATIO),
        },
        "metric_note": (
            "No ground-truth homography is available in this direct-matching "
            "evaluation. Therefore, Avg. Correct Matches/NCM denotes the "
            "average number of RANSAC geometrically consistent inliers, and "
            "CMR denotes the macro-average per-pair RANSAC inlier ratio."
        ),
    }

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------
    json_data = {
        "per_image_matches": per_pair,
        "overall_statistics": overall,
    }

    json_path = os.path.join(output_root, "pair2match.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            json_data,
            f,
            indent=4,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Save CSV for direct use in paper tables
    # --------------------------------------------------------
    csv_path = os.path.join(output_root, "match_statistics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "num_pairs",
            "avg_matches",
            "avg_correct_matches",
            "CMR_percent",
            "pooled_CMR_percent",
            "RMSE",
            "successful_pairs",
            "success_rate_percent",
        ])
        writer.writerow([
            num_pairs,
            f"{avg_matches:.6f}",
            f"{avg_correct_matches:.6f}",
            f"{100.0 * avg_cmr:.6f}",
            f"{100.0 * pooled_cmr:.6f}",
            f"{avg_rmse:.6f}",
            successful_pairs,
            f"{100.0 * success_rate:.6f}",
        ])

    # --------------------------------------------------------
    # Final printout
    # --------------------------------------------------------
    print("\n" + "=" * 64)
    print("Final direct-matching correspondence statistics")
    print("=" * 64)
    print(f"Pairs: {num_pairs}")
    print(f"Avg. Matches: {avg_matches:.2f}")
    print(f"Avg. Correct Matches: {avg_correct_matches:.2f}")
    print(f"CMR (macro per-pair): {100.0 * avg_cmr:.2f}%")
    print(f"Pooled CMR: {100.0 * pooled_cmr:.2f}%")
    print(f"RMSE (with failure penalty): {avg_rmse:.4f}")
    print(
        f"Successful pairs: {successful_pairs}/{num_pairs} "
        f"({100.0 * success_rate:.2f}%)"
    )
    print(f"JSON saved to: {json_path}")
    print(f"CSV saved to: {csv_path}")
    print(f"Elapsed time: {time.time() - start:.2f}s")
