import argparse
import json
import os
import re
import time

import cv2
import numpy as np

from src.utils.load_model import load_model


METHOD_CHOICES = [
    "xoftr",
    "sp_lg",
    "loftr",
    "roma",
    "eloftr",
    "xfeat",
    "omniglue",
    "reasoning_xfeat",
    "dgim",
    "swinmatcher",
    "miftr",
    "mapglue",
    "jamma",
    "semaglue",
]


# 根据你工程中的实际权重位置修改这些默认值。
DEFAULT_CKPTS = {
    "xoftr": "weights/weights_xoftr_640.ckpt",
    "sp_lg": "weights/sp+lg_megadepth_weights.pth",
    "loftr": "weights/minima_loftr.ckpt",
    "roma": "weights/minima_roma.pth",
    "eloftr": "weights/minima_eloftr.ckpt",
    "xfeat": "weights/xfeat.pt",
    "omniglue": "weights/omniglue_models",
    "reasoning_xfeat": "weights/reasoning_xfeat.pt",
    "dgim": "weights/dgim_512.ckpt",
    "swinmatcher": "weights/swinmatcher_512.ckpt",
    "miftr": "weights/miftr_full_tune.ckpt",
    "mapglue": "weights/fastmapglue_model.pt",
    "jamma": "official",
    "semaglue": "weights/semaglue.pth",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch cross-modal matching with multiple methods"
    )

    parser.add_argument(
        "--method",
        type=str,
        default="sp_lg",
        choices=METHOD_CHOICES,
        help="Select matching method",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="6",
    )
    parser.add_argument(
        "--L_folder",
        type=str,
        default="./XoFTR-main-more/data/xinzhou_UVA/NIR",
    )
    parser.add_argument(
        "--R_folder",
        type=str,
        default="./XoFTR-main-more/data/xinzhou_UVA/VIS",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./results_all_methods/xinzhou_UVA",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint path. If omitted, use the default for the selected method.",
    )

    # 通用匹配参数。load_model 中不需要的参数会被忽略。
    parser.add_argument("--match_threshold", type=float, default=0.3)
    parser.add_argument("--fine_threshold", type=float, default=0.1)
    parser.add_argument("--thr", type=float, default=0.2)
    parser.add_argument("--resize", type=int, default=640)
    parser.add_argument("--max_keypoints", type=int, default=2048)
    parser.add_argument("--nms_radius", type=int, default=4)
    parser.add_argument("--sinkhorn_iterations", type=int, default=20)
    parser.add_argument("--match_filter", type=float, default=0.2)

    # 部分方法可能会访问这些参数。
    parser.add_argument("--model_type", type=str, default="full")
    parser.add_argument("--ckpt2", type=str, default="large")
    parser.add_argument("--border_rm", type=int, default=2)
    parser.add_argument("--coarse_thr", type=float, default=0.2)
    parser.add_argument("--fine_thr", type=float, default=0.1)
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--denser", action="store_true")

    # 统计与保存参数。
    parser.add_argument("--ransac_threshold", type=float, default=3.0)
    parser.add_argument("--min_matches", type=int, default=20)
    parser.add_argument("--min_inliers", type=int, default=20)
    parser.add_argument("--min_inlier_ratio", type=float, default=0.2)
    parser.add_argument(
        "--failure_rmse",
        type=float,
        default=5.0,
        help=(
            "Fixed RMSE penalty assigned when homography estimation fails "
            "or the number of RANSAC inliers is insufficient."
        ),
    )
    parser.add_argument("--max_save_figs", type=int, default=20)
    parser.add_argument(
        "--save_figs",
        action="store_true",
        help="Save matching figures. At most max_save_figs images are saved.",
    )
    parser.add_argument(
        "--read_color",
        action="store_true",
        help="Ask the loader to read color images when supported.",
    )

    args = parser.parse_args()

    if args.ckpt is None:
        args.ckpt = DEFAULT_CKPTS[args.method]

    return args


def natural_sort(file_list):
    return sorted(
        file_list,
        key=lambda x: [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r"([0-9]+)", x)
        ],
    )


def resize_image_and_points(img, points, target_long_edge=640):
    height, width = img.shape[:2]
    long_edge = max(height, width)

    if long_edge <= 0:
        return img, np.asarray(points).reshape(-1, 2)

    scale = target_long_edge / long_edge
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resized_img = cv2.resize(
        img,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )

    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    scaled_points = np.round(points * scale).astype(int)

    return resized_img, scaled_points


def draw_matches(
    img1_path,
    img2_path,
    keypoints1,
    keypoints2,
    inlier_mask,
    path,
    target_long_edge=640,
):
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

    inlier_mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)

    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    canvas = np.zeros(
        (max(height1, height2), width1 + width2, 3),
        dtype=np.uint8,
    )
    canvas[:height1, :width1] = img1
    canvas[:height2, width1:] = img2

    for pt1, pt2, is_inlier in zip(
        keypoints1,
        keypoints2,
        inlier_mask,
    ):
        p1 = tuple(map(int, pt1))
        p2 = (int(pt2[0] + width1), int(pt2[1]))
        color = (0, 255, 0) if is_inlier else (0, 0, 255)

        cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 2, color, -1)
        cv2.circle(canvas, p2, 2, color, -1)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, canvas)


def compute_ransac_statistics(
    pts1,
    pts2,
    threshold=3.0,
):
    """
    没有真实单应矩阵时，统计 RANSAC 几何内点率。

    注意：这里得到的是 RANSAC inlier ratio，
    不是 ground-truth correct match rate。
    """
    pts1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 2)
    pts2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 2)

    num_matches = len(pts1)

    empty = {
        "num_matches": num_matches,
        "num_inliers": 0,
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
        ransacReprojThreshold=threshold,
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

    valid = (
        np.abs(projected_h[:, 2])
        > np.finfo(np.float64).eps
    )
    projected = np.full_like(pts1, np.inf, dtype=np.float64)
    projected[valid] = (
        projected_h[valid, :2]
        / projected_h[valid, 2:3]
    )

    errors = np.linalg.norm(projected - pts2, axis=1)
    valid_inlier_mask = mask & np.isfinite(errors)

    num_inliers = int(valid_inlier_mask.sum())
    inlier_ratio = (
        num_inliers / num_matches
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
        "inlier_ratio": inlier_ratio,
        "rmse": rmse,
        "inlier_mask": valid_inlier_mask,
        "H": H,
    }


def build_matcher(args):
    """
    load_model(..., use_path=True) 应返回一个以图像路径为输入的匹配函数。

    等价调用形式：
        output_data = matcher(img0_path, img1_path)
    """
    matcher = load_model(
        args.method,
        args,
        use_path=True,
    )
    return matcher


def main(args):
# Set CUDA_VISIBLE_DEVICES in the shell when selecting GPUs.

    L_files = natural_sort([
        f for f in os.listdir(args.L_folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
        )
    ])
    R_files = natural_sort([
        f for f in os.listdir(args.R_folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
        )
    ])

    if len(L_files) != len(R_files):
        raise ValueError(
            f"L/R image counts differ: "
            f"{len(L_files)} vs {len(R_files)}"
        )

    output_dir = os.path.join(
        args.output_root,
        args.method,
    )
    os.makedirs(output_dir, exist_ok=True)

    matcher = build_matcher(args)

    total_matches = 0
    total_inliers = 0
    sum_inlier_ratio = 0.0
    total_rmse = 0.0
    successful_pairs = 0

    per_pair = []
    start = time.time()

    for idx, (L_file, R_file) in enumerate(
        zip(L_files, R_files)
    ):
        img0_path = os.path.join(args.L_folder, L_file)
        img1_path = os.path.join(args.R_folder, R_file)

        # use_path=True 时，统一直接传入路径。
        output_data = matcher(img0_path, img1_path)

        mkpts0 = np.asarray(
            output_data["mkpts0"],
            dtype=np.float64,
        ).reshape(-1, 2)
        mkpts1 = np.asarray(
            output_data["mkpts1"],
            dtype=np.float64,
        ).reshape(-1, 2)

        stats = compute_ransac_statistics(
            mkpts0,
            mkpts1,
            threshold=args.ransac_threshold,
        )

        num_matches = stats["num_matches"]
        num_inliers = stats["num_inliers"]
        inlier_ratio = stats["inlier_ratio"]
        rmse = stats["rmse"]

        total_matches += num_matches
        total_inliers += num_inliers
        sum_inlier_ratio += inlier_ratio

        # 保留原实验中的固定惩罚设计：
        # 1) 单应矩阵估计失败，rmse is None；
        # 2) RANSAC 内点数量不足。
        # 以上情况统一将 RMSE 记为 failure_rmse，默认 5.0。
        raw_rmse = rmse
        if rmse is None or num_inliers < args.min_inliers:
            rmse = float(args.failure_rmse)

        success = (
            num_matches >= args.min_matches
            and num_inliers > args.min_inliers
            and inlier_ratio > args.min_inlier_ratio
        )
        successful_pairs += int(success)
        total_rmse += rmse

        rmse_print = f"{rmse:.4f}"

        print(
            f"[{idx + 1}/{len(L_files)}] "
            f"{L_file} <-> {R_file} | "
            f"#Matches={num_matches}, "
            f"#Inliers={num_inliers}, "
            f"IR={100.0 * inlier_ratio:.2f}%, "
            f"RMSE={rmse_print}, "
            f"Success={success}"
        )

        per_pair.append({
            "image_index": idx + 1,
            "left_image": L_file,
            "right_image": R_file,
            "method": args.method,
            "num_matches": num_matches,
            "num_ransac_inliers": num_inliers,
            "ransac_inlier_ratio": inlier_ratio,
            "raw_inlier_rmse": raw_rmse,
            "RMSE": rmse,
            "rmse_penalized": bool(
                raw_rmse is None
                or num_inliers < args.min_inliers
            ),
            "successful_pair": bool(success),
        })

        if (
            args.save_figs
            and args.max_save_figs > 0
            and idx < args.max_save_figs
        ):
            output_path = os.path.join(
                output_dir,
                "match_figures",
                f"{idx + 1:04d}_{L_file}_{R_file}.png",
            )
            draw_matches(
                img0_path,
                img1_path,
                mkpts0,
                mkpts1,
                stats["inlier_mask"],
                output_path,
                target_long_edge=args.resize,
            )

    num_pairs = len(per_pair)
    pair_denom = max(num_pairs, 1)
    match_denom = max(total_matches, 1)

    overall = {
        "method": args.method,
        "ckpt": args.ckpt,
        "num_pairs": num_pairs,
        "avg_matches": total_matches / pair_denom,
        "avg_ransac_inliers": total_inliers / pair_denom,
        "avg_ransac_inlier_ratio": (
            sum_inlier_ratio / pair_denom
        ),
        "pooled_ransac_inlier_ratio": (
            total_inliers / match_denom
        ),
        "RMSE": total_rmse / pair_denom,
        "failure_rmse": args.failure_rmse,
        "successful_pairs": successful_pairs,
        "success_rate": successful_pairs / pair_denom,
        "ransac_threshold_px": args.ransac_threshold,
        "success_definition": {
            "min_matches": args.min_matches,
            "min_inliers_strictly_greater_than": args.min_inliers,
            "min_inlier_ratio_strictly_greater_than": args.min_inlier_ratio,
        },
        "rmse_definition": (
            "RMSE is computed from RANSAC homography inliers. "
            "Pairs with failed homography estimation or fewer than "
            f"{args.min_inliers} inliers are assigned the fixed penalty "
            f"{args.failure_rmse}."
        ),
        "metric_note": (
            "No ground-truth homography is available. "
            "The inlier ratio is computed from a homography "
            "estimated by RANSAC and measures geometric consistency."
        ),
    }

    json_data = {
        "per_image_matches": per_pair,
        "overall_statistics": overall,
    }

    json_path = os.path.join(
        output_dir,
        "pair2match.json",
    )
    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            json_data,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print("\n处理完毕")
    print(f"平均惩罚后 RMSE: {overall['RMSE']:.4f}")
    print(json.dumps(overall, indent=4, ensure_ascii=False))
    print(f"结果保存至: {json_path}")
    print(f"Elapsed time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main(parse_args())
