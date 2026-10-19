import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np

from src.xoftr import XoFTR_dino_resnet_panmatch_light_withfpn
from src.config.default import get_cfg_defaults
from src.utils.data_io import DataIOWrapper, lower_config


def natural_sort(file_list):
    """Sort filenames in natural numeric order."""
    return sorted(
        file_list,
        key=lambda x: [
            int(token) if token.isdigit() else token.lower()
            for token in re.split(r"([0-9]+)", x)
        ],
    )


def list_image_pairs(folder0, folder1):
    """
    Pair images by natural-sorted order.

    The two folders do not need to be geometrically aligned, but they must
    contain corresponding image pairs in the same natural order.
    """
    image_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
    }

    files0 = natural_sort([
        name for name in os.listdir(folder0)
        if Path(name).suffix.lower() in image_extensions
    ])
    files1 = natural_sort([
        name for name in os.listdir(folder1)
        if Path(name).suffix.lower() in image_extensions
    ])

    if len(files0) != len(files1):
        raise ValueError(
            f"The two folders contain different numbers of images: "
            f"{len(files0)} vs {len(files1)}."
        )

    return [
        (
            os.path.join(folder0, name0),
            os.path.join(folder1, name1),
            name0,
            name1,
        )
        for name0, name1 in zip(files0, files1)
    ]


def compute_ransac_statistics(
    mkpts0,
    mkpts1,
    ransac_threshold=3.0,
    confidence=0.999,
    max_iters=10000,
):
    """
    Compute geometric-consistency statistics without ground-truth homography.

    The returned inlier ratio is the proportion of matches accepted by
    RANSAC-based homography estimation. It is not a ground-truth correct-match
    rate.
    """
    mkpts0 = np.asarray(mkpts0, dtype=np.float64).reshape(-1, 2)
    mkpts1 = np.asarray(mkpts1, dtype=np.float64).reshape(-1, 2)

    num_matches = int(len(mkpts0))

    empty_result = {
        "num_matches": num_matches,
        "num_inliers": 0,
        "inlier_ratio": 0.0,
        "inlier_rmse": None,
        "inlier_mean_error": None,
        "inlier_median_error": None,
        "inlier_mask": np.zeros(num_matches, dtype=bool),
        "H_est": None,
    }

    if num_matches < 4 or len(mkpts1) != num_matches:
        return empty_result

    H_est, mask = cv2.findHomography(
        mkpts0,
        mkpts1,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=max_iters,
        confidence=confidence,
    )

    if H_est is None or mask is None:
        return empty_result

    if not np.isfinite(H_est).all() or np.linalg.matrix_rank(H_est) < 3:
        return empty_result

    inlier_mask = mask.ravel().astype(bool)
    num_inliers = int(inlier_mask.sum())

    if num_inliers == 0:
        empty_result["H_est"] = H_est
        return empty_result

    pts0_h = np.hstack([
        mkpts0,
        np.ones((num_matches, 1), dtype=np.float64),
    ])
    projected_h = (H_est @ pts0_h.T).T

    valid = (
        np.abs(projected_h[:, 2])
        > np.finfo(np.float64).eps
    )
    projected = np.full(
        (num_matches, 2),
        np.inf,
        dtype=np.float64,
    )
    projected[valid] = (
        projected_h[valid, :2]
        / projected_h[valid, 2:3]
    )

    errors = np.linalg.norm(projected - mkpts1, axis=1)
    valid_inliers = inlier_mask & np.isfinite(errors)

    if not np.any(valid_inliers):
        empty_result["H_est"] = H_est
        return empty_result

    inlier_errors = errors[valid_inliers]

    return {
        "num_matches": num_matches,
        "num_inliers": int(valid_inliers.sum()),
        "inlier_ratio": float(valid_inliers.mean()),
        "inlier_rmse": float(
            np.sqrt(np.mean(inlier_errors ** 2))
        ),
        "inlier_mean_error": float(np.mean(inlier_errors)),
        "inlier_median_error": float(np.median(inlier_errors)),
        "inlier_mask": valid_inliers,
        "H_est": H_est,
    }


def draw_matches(
    img0_path,
    img1_path,
    mkpts0,
    mkpts1,
    inlier_mask,
    output_path,
    text_lines,
):
    """Save green RANSAC inliers and red RANSAC outliers."""
    img0 = cv2.imread(img0_path, cv2.IMREAD_COLOR)
    img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)

    if img0 is None or img1 is None:
        raise FileNotFoundError(
            f"Cannot read image pair: {img0_path}, {img1_path}"
        )

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    canvas = np.zeros(
        (max(h0, h1), w0 + w1, 3),
        dtype=np.uint8,
    )
    canvas[:h0, :w0] = img0
    canvas[:h1, w0:w0 + w1] = img1

    mkpts0 = np.asarray(mkpts0).reshape(-1, 2)
    mkpts1 = np.asarray(mkpts1).reshape(-1, 2)
    inlier_mask = np.asarray(inlier_mask, dtype=bool)

    for p0, p1, is_inlier in zip(
        mkpts0,
        mkpts1,
        inlier_mask,
    ):
        pt0 = tuple(np.round(p0).astype(int))
        pt1 = (
            int(round(p1[0])) + w0,
            int(round(p1[1])),
        )

        color = (0, 255, 0) if is_inlier else (0, 0, 255)
        cv2.line(canvas, pt0, pt1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, pt0, 2, color, -1)
        cv2.circle(canvas, pt1, 2, color, -1)

    y = 30
    for line in text_lines:
        cv2.putText(
            canvas,
            str(line),
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, canvas)


def build_matcher(args):
    config = get_cfg_defaults(inference=True)
    config = lower_config(config)

    config["xoftr"]["match_coarse"]["thr"] = args.match_threshold
    config["xoftr"]["fine"]["thr"] = args.fine_threshold
    config["xoftr"]["fine"]["denser"] = args.denser

    config["test"]["img0_resize"] = args.resize
    config["test"]["img1_resize"] = args.resize

    model = XoFTR_dino_resnet_panmatch_light_withfpn(
        config=config["xoftr"]
    )
    matcher = DataIOWrapper(
        model,
        config=config["test"],
        ckpt=args.ckpt,
    )
    return matcher.eval()


def evaluate(args):
# Set CUDA_VISIBLE_DEVICES in the shell when selecting GPUs.

    pairs = list_image_pairs(args.folder0, args.folder1)
    if not pairs:
        raise RuntimeError("No image pairs were found.")

    matcher = build_matcher(args)

    pair_rows = []

    total_matches = 0
    total_inliers = 0
    sum_pair_inlier_ratio = 0.0
    valid_rmse = []
    valid_mean_error = []
    valid_median_error = []

    successful_pairs = 0
    start = time.time()

    for idx, (
        img0_path,
        img1_path,
        name0,
        name1,
    ) in enumerate(pairs):
        output = matcher.from_paths(
            img0_path,
            img1_path,
            read_color=args.read_color,
        )

        mkpts0 = np.asarray(output["mkpts0"]).reshape(-1, 2)
        mkpts1 = np.asarray(output["mkpts1"]).reshape(-1, 2)

        stats = compute_ransac_statistics(
            mkpts0,
            mkpts1,
            ransac_threshold=args.ransac_threshold,
            confidence=args.ransac_confidence,
            max_iters=args.ransac_max_iters,
        )

        num_matches = stats["num_matches"]
        num_inliers = stats["num_inliers"]
        inlier_ratio = stats["inlier_ratio"]

        total_matches += num_matches
        total_inliers += num_inliers
        sum_pair_inlier_ratio += inlier_ratio

        if stats["inlier_rmse"] is not None:
            valid_rmse.append(stats["inlier_rmse"])
        if stats["inlier_mean_error"] is not None:
            valid_mean_error.append(stats["inlier_mean_error"])
        if stats["inlier_median_error"] is not None:
            valid_median_error.append(
                stats["inlier_median_error"]
            )

        is_success = (
            num_matches >= args.min_matches
            and num_inliers >= args.min_inliers
            and inlier_ratio >= args.min_inlier_ratio
        )
        successful_pairs += int(is_success)

        row = {
            "pair_index": idx + 1,
            "image0": name0,
            "image1": name1,
            "num_matches": num_matches,
            "num_ransac_inliers": num_inliers,
            "ransac_inlier_ratio": inlier_ratio,
            "inlier_rmse": (
                stats["inlier_rmse"]
                if stats["inlier_rmse"] is not None
                else ""
            ),
            "inlier_mean_error": (
                stats["inlier_mean_error"]
                if stats["inlier_mean_error"] is not None
                else ""
            ),
            "inlier_median_error": (
                stats["inlier_median_error"]
                if stats["inlier_median_error"] is not None
                else ""
            ),
            "successful_pair": int(is_success),
        }
        pair_rows.append(row)

        rmse_text = (
            f"{stats['inlier_rmse']:.4f}"
            if stats["inlier_rmse"] is not None
            else "NA"
        )

        print(
            f"[{idx + 1}/{len(pairs)}] "
            f"{name0} <-> {name1}: "
            f"#Matches={num_matches}, "
            f"#Inliers={num_inliers}, "
            f"IR={100.0 * inlier_ratio:.2f}%, "
            f"Inlier-RMSE={rmse_text}, "
            f"Success={is_success}"
        )

        if (
            args.save_figs
            and args.max_save_figs > 0
            and idx < args.max_save_figs
        ):
            fig_path = os.path.join(
                args.output_dir,
                "match_figures",
                f"{idx + 1:04d}_{Path(name0).stem}_"
                f"{Path(name1).stem}.png",
            )
            draw_matches(
                img0_path,
                img1_path,
                mkpts0,
                mkpts1,
                stats["inlier_mask"],
                fig_path,
                text_lines=[
                    f"#Matches: {num_matches}",
                    (
                        f"RANSAC Inliers: "
                        f"{num_inliers}/{num_matches}"
                    ),
                    (
                        f"Inlier Ratio "
                        f"({args.ransac_threshold:g}px): "
                        f"{100.0 * inlier_ratio:.2f}%"
                    ),
                    f"Inlier RMSE: {rmse_text}px",
                ],
            )

    num_pairs = len(pair_rows)
    pair_denom = max(num_pairs, 1)
    match_denom = max(total_matches, 1)

    summary = {
        "num_pairs": num_pairs,
        "avg_matches": total_matches / pair_denom,
        "avg_ransac_inliers": total_inliers / pair_denom,
        "avg_ransac_inlier_ratio": (
            sum_pair_inlier_ratio / pair_denom
        ),
        "pooled_ransac_inlier_ratio": (
            total_inliers / match_denom
        ),
        "avg_inlier_rmse": (
            float(np.mean(valid_rmse))
            if valid_rmse
            else None
        ),
        "avg_inlier_mean_error": (
            float(np.mean(valid_mean_error))
            if valid_mean_error
            else None
        ),
        "avg_inlier_median_error": (
            float(np.mean(valid_median_error))
            if valid_median_error
            else None
        ),
        "successful_pairs": successful_pairs,
        "success_rate": successful_pairs / pair_denom,
        "ransac_threshold_px": args.ransac_threshold,
        "ransac_confidence": args.ransac_confidence,
        "ransac_max_iters": args.ransac_max_iters,
        "success_definition": {
            "min_matches": args.min_matches,
            "min_inliers": args.min_inliers,
            "min_inlier_ratio": args.min_inlier_ratio,
        },
        "metric_note": (
            "The reported inlier ratio is based on a homography "
            "estimated from the predicted correspondences using RANSAC. "
            "It measures geometric consistency and is not a "
            "ground-truth correct-match rate."
        ),
    }

    os.makedirs(args.output_dir, exist_ok=True)

    csv_path = os.path.join(
        args.output_dir,
        "pair_ransac_statistics.csv",
    )
    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(pair_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(pair_rows)

    json_path = os.path.join(
        args.output_dir,
        "summary_ransac_statistics.json",
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            indent=4,
            ensure_ascii=False,
        )

    elapsed = time.time() - start

    print("\n===== Dataset summary =====")
    print(f"Pairs: {num_pairs}")
    print(f"Avg. Matches: {summary['avg_matches']:.4f}")
    print(
        "Avg. RANSAC Inliers: "
        f"{summary['avg_ransac_inliers']:.4f}"
    )
    print(
        "Avg. RANSAC Inlier Ratio: "
        f"{100.0 * summary['avg_ransac_inlier_ratio']:.4f}%"
    )
    print(
        "Pooled RANSAC Inlier Ratio: "
        f"{100.0 * summary['pooled_ransac_inlier_ratio']:.4f}%"
    )
    print(
        "Avg. Inlier RMSE: "
        f"{summary['avg_inlier_rmse']}"
    )
    print(
        "Success Rate: "
        f"{100.0 * summary['success_rate']:.4f}% "
        f"({successful_pairs}/{num_pairs})"
    )
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Elapsed time: {elapsed:.2f}s")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate average match count and RANSAC inlier ratio "
            "for unaligned cross-modal image pairs without GT H."
        )
    )

    parser.add_argument(
        "--folder0",
        type=str,
        default="./data/op_sar/sar_R",
    )
    parser.add_argument(
        "--folder1",
        type=str,
        default="./data/op_sar/op",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=(
            "./logs/tb_logs/"
            "dinomatch/version_32/checkpoints/"
            "epoch=27-auc@5=0.206-auc@10=0.378-auc@20=0.554.ckpt"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results_op_sar_ransac_statistics",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="6",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=640,
    )
    parser.add_argument(
        "--match_threshold",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--fine_threshold",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--denser",
        action="store_true",
    )
    parser.add_argument(
        "--read_color",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--ransac_threshold",
        type=float,
        default=3.0,
        help="RANSAC homography reprojection threshold in pixels.",
    )
    parser.add_argument(
        "--ransac_confidence",
        type=float,
        default=0.999,
    )
    parser.add_argument(
        "--ransac_max_iters",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--min_matches",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--min_inliers",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--min_inlier_ratio",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--save_figs",
        action="store_true",
    )
    parser.add_argument(
        "--max_save_figs",
        type=int,
        default=10,
        help=(
            "Maximum number of figures saved in total; "
            "set 0 or omit --save_figs to save none."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
