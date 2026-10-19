from collections import defaultdict, OrderedDict
import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import warnings
import json
import time
import csv

from src.utils.metrics import (
    estimate_pose,
    relative_pose_error,
    error_auc,
    symmetric_epipolar_distance_numpy,
)
from src.utils.plotting import dynamic_alpha, error_colormap, make_matching_figure


# ============================================================================
# Loading functions
# ============================================================================
def load_xoftr(args):
    """Load the GeoSAM ablation model used by the original script."""
    from src.xoftr import (
        XoFTR_dino,
        XoFTR_dino_resnet,
        XoFTR_dino_resnet_panmatch,
        XoFTR_dino_resnet_panmatch_light,
        XoFTR_dino_resnet_panmatch_light_withfpn,
        XoFTR_dino_resnet_panmatch_without_fpn,
        GeoSAM_without_RopE,
        GeoSAM,
        GeoSAM_reverse_GGSA,
        GeoSAM_ablation_structure,
    )
    from src.config.default import get_cfg_defaults
    from src.utils.data_io import DataIOWrapper, lower_config

    config = get_cfg_defaults(inference=True)
    config = lower_config(config)
    config["xoftr"]["match_coarse"]["thr"] = args.match_threshold
    config["xoftr"]["fine"]["thr"] = args.fine_threshold

    ckpt = args.ckpt

    # Keep the same model as in your original ablation script.
    matcher = XoFTR_dino_resnet_panmatch_light_withfpn(config=config["xoftr"])
    matcher = DataIOWrapper(matcher, config=config["test"], ckpt=ckpt)

    return matcher.from_paths


# ============================================================================
# Dataset
# ============================================================================
def load_vis_tir_pairs_npz(npz_root, npz_list):
    """Load scene/image-pair information from the METU_VisTIR npz files."""
    with open(npz_list, "r") as f:
        npz_names = [name.split()[0] for name in f.readlines()]

    print(f"Parse {len(npz_names)} npz from {npz_list}.")

    total_pairs = 0
    scene_pairs = {}

    for name in npz_names:
        print(f"Loading {name}")
        scene_info = np.load(f"{npz_root}/{name}", allow_pickle=True)
        pairs = []

        for pair_info in scene_info["pair_infos"]:
            total_pairs += 1
            id0, id1 = pair_info

            im0 = scene_info["image_paths"][id0][0]
            im1 = scene_info["image_paths"][id1][1]

            K0 = scene_info["intrinsics"][id0][0].astype(np.float32)
            K1 = scene_info["intrinsics"][id1][1].astype(np.float32)

            dist0 = np.array(scene_info["distortion_coefs"][id0][0], dtype=float)
            dist1 = np.array(scene_info["distortion_coefs"][id1][1], dtype=float)

            T0 = scene_info["poses"][id0]
            T1 = scene_info["poses"][id1]
            T_0to1 = np.matmul(T1, np.linalg.inv(T0))

            pairs.append(
                {
                    "im0": im0,
                    "im1": im1,
                    "dist0": dist0,
                    "dist1": dist1,
                    "K0": K0,
                    "K1": K1,
                    "T_0to1": T_0to1,
                }
            )

        scene_pairs[name] = pairs

    print(f"Loaded {total_pairs} pairs.")
    return scene_pairs


# ============================================================================
# GT correspondence statistics
# ============================================================================
def compute_gt_match_statistics(
    mkpts0,
    mkpts1,
    T_0to1,
    K0,
    K1,
    epi_thr=1e-4,
):
    """
    Compute GT-based correspondence statistics over ALL predicted matches.

    Correctness is determined using the ground-truth essential matrix and
    symmetric epipolar distance. No RANSAC filtering is used for CMR.

    Returns
    -------
    num_matches : int
        Number of all predicted correspondences for this image pair.
    num_correct : int
        Number of correspondences with GT epipolar error < epi_thr.
    correct_match_rate : float
        num_correct / num_matches for this image pair. If no match is predicted,
        the value is 0.
    epi_errs : ndarray
        Per-match GT symmetric epipolar errors.
    correct_mask : ndarray[bool]
        Per-match correctness mask.
    """
    mkpts0 = np.asarray(mkpts0)
    mkpts1 = np.asarray(mkpts1)

    num_matches = len(mkpts0)

    if num_matches == 0:
        return {
            "num_matches": 0,
            "num_correct": 0,
            "correct_match_rate": 0.0,
            "epi_errs": np.empty((0,), dtype=np.float32),
            "correct_mask": np.empty((0,), dtype=bool),
        }

    # Essential matrix from GT relative pose: E = [t]_x R
    Tx = np.cross(np.eye(3), T_0to1[:3, 3])
    E_mat = Tx @ T_0to1[:3, :3]

    epi_errs = symmetric_epipolar_distance_numpy(
        mkpts0,
        mkpts1,
        E_mat,
        K0,
        K1,
    )
    epi_errs = np.asarray(epi_errs)

    correct_mask = epi_errs < epi_thr
    num_correct = int(correct_mask.sum())
    correct_match_rate = float(num_correct / num_matches)

    return {
        "num_matches": int(num_matches),
        "num_correct": num_correct,
        "correct_match_rate": correct_match_rate,
        "epi_errs": epi_errs,
        "correct_mask": correct_mask,
    }


# ============================================================================
# Visualization
# ============================================================================
def save_matching_figure(
    path,
    img0,
    img1,
    mkpts0,
    mkpts1,
    T_0to1,
    K0,
    K1,
    t_err=None,
    R_err=None,
    name=None,
    epi_thr=1e-4,
):
    """Save matching visualization colored by GT epipolar error."""
    match_stats = compute_gt_match_statistics(
        mkpts0,
        mkpts1,
        T_0to1,
        K0,
        K1,
        epi_thr=epi_thr,
    )

    num_matches = match_stats["num_matches"]
    num_correct = match_stats["num_correct"]
    cmr = match_stats["correct_match_rate"]
    epi_errs = match_stats["epi_errs"]

    text = [name] if name is not None else []
    text += [
        f"#Matches: {num_matches}",
        f"CMR@{epi_thr:.1e}: {100.0 * cmr:.2f}% ({num_correct}/{num_matches})",
    ]

    if R_err is not None and t_err is not None:
        text += [
            f"Delta R: {R_err:.2f} deg",
            f"Delta t: {t_err:.2f} deg",
        ]

    if num_matches > 0:
        alpha = dynamic_alpha(num_matches)
        color = error_colormap(epi_errs, epi_thr, alpha=alpha)
    else:
        color = np.empty((0, 4), dtype=np.float32)

    make_matching_figure(
        img0,
        img1,
        mkpts0,
        mkpts1,
        color,
        text=text,
        path=path,
        dpi=150,
    )


# ============================================================================
# Aggregation helpers
# ============================================================================
def aggregiate_scenes(scene_pose_auc, thresholds):
    """Average AUC results for cloudy_cloudy / cloudy_sunny groups."""
    temp_pose_auc = {}

    for npz_name in scene_pose_auc.keys():
        scene_name = npz_name.split("_scene")[0]
        temp_pose_auc[scene_name] = [
            np.zeros(len(thresholds), dtype=np.float32),
            0,
        ]

    for npz_name in scene_pose_auc.keys():
        scene_name = npz_name.split("_scene")[0]
        temp_pose_auc[scene_name][0] += scene_pose_auc[npz_name]
        temp_pose_auc[scene_name][1] += 1

    agg_pose_auc = {}
    for scene_name in temp_pose_auc.keys():
        agg_pose_auc[scene_name] = (
            temp_pose_auc[scene_name][0] / temp_pose_auc[scene_name][1]
        )

    return agg_pose_auc


def aggregate_match_statistics(scene_match_stats):
    """
    Aggregate scene-level matching statistics into cloudy_cloudy/cloudy_sunny.

    Aggregation is weighted by the number of evaluated pairs. Therefore:
      Avg. Matches          = sum_i M_i / N
      Avg. Correct Matches  = sum_i C_i / N
      CMR                   = (1/N) * sum_i (C_i / M_i)

    Note that CMR is a per-pair macro average, matching the reviewer's request
    for per-pair average statistics.
    """
    grouped = defaultdict(
        lambda: {
            "num_pairs": 0,
            "failed_pairs": 0,
            "sum_matches": 0.0,
            "sum_correct_matches": 0.0,
            "sum_correct_match_rate": 0.0,
            "sum_ransac_inlier_ratio": 0.0,
        }
    )

    for npz_name, stats in scene_match_stats.items():
        group_name = npz_name.split("_scene")[0]
        group = grouped[group_name]

        group["num_pairs"] += stats["num_pairs"]
        group["failed_pairs"] += stats["failed_pairs"]
        group["sum_matches"] += stats["sum_matches"]
        group["sum_correct_matches"] += stats["sum_correct_matches"]
        group["sum_correct_match_rate"] += stats["sum_correct_match_rate"]
        group["sum_ransac_inlier_ratio"] += stats["sum_ransac_inlier_ratio"]

    aggregated = {}

    for group_name, values in grouped.items():
        n = values["num_pairs"]
        denom = max(n, 1)

        aggregated[group_name] = {
            "num_pairs": int(n),
            "failed_pairs": int(values["failed_pairs"]),
            "avg_matches": float(values["sum_matches"] / denom),
            "avg_correct_matches": float(values["sum_correct_matches"] / denom),
            "avg_correct_match_rate": float(
                values["sum_correct_match_rate"] / denom
            ),
            "avg_ransac_inlier_ratio": float(
                values["sum_ransac_inlier_ratio"] / denom
            ),
        }

    return aggregated


def save_match_statistics_csv(csv_path, scene_match_stats, aggregate_match_stats):
    """Save scene-level and aggregate matching statistics to CSV."""
    fieldnames = [
        "level",
        "name",
        "num_pairs",
        "failed_pairs",
        "avg_matches",
        "avg_correct_matches",
        "cmr_percent",
        "avg_ransac_inlier_ratio_percent",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for name, stats in scene_match_stats.items():
            writer.writerow(
                {
                    "level": "scene",
                    "name": name,
                    "num_pairs": stats["num_pairs"],
                    "failed_pairs": stats["failed_pairs"],
                    "avg_matches": f'{stats["avg_matches"]:.6f}',
                    "avg_correct_matches": f'{stats["avg_correct_matches"]:.6f}',
                    "cmr_percent": f'{100.0 * stats["avg_correct_match_rate"]:.6f}',
                    "avg_ransac_inlier_ratio_percent": (
                        f'{100.0 * stats["avg_ransac_inlier_ratio"]:.6f}'
                    ),
                }
            )

        for name, stats in aggregate_match_stats.items():
            writer.writerow(
                {
                    "level": "aggregate",
                    "name": name,
                    "num_pairs": stats["num_pairs"],
                    "failed_pairs": stats["failed_pairs"],
                    "avg_matches": f'{stats["avg_matches"]:.6f}',
                    "avg_correct_matches": f'{stats["avg_correct_matches"]:.6f}',
                    "cmr_percent": f'{100.0 * stats["avg_correct_match_rate"]:.6f}',
                    "avg_ransac_inlier_ratio_percent": (
                        f'{100.0 * stats["avg_ransac_inlier_ratio"]:.6f}'
                    ),
                }
            )


# ============================================================================
# Evaluation
# ============================================================================
def eval_relapose(
    matcher,
    data_root,
    scene_pairs,
    ransac_thres,
    thresholds,
    save_figs,
    figures_dir=None,
    method=None,
    print_out=False,
    debug=False,
    epi_thr=1e-4,
    max_save_figs=10,
):
    scene_pose_auc = {}
    scene_match_stats = {}

    for scene_name in scene_pairs.keys():
        scene_dir = osp.join(figures_dir, scene_name.split(".")[0])
        if save_figs and not osp.exists(scene_dir):
            os.makedirs(scene_dir)

        pairs = scene_pairs[scene_name]
        statis = defaultdict(list)
        np.set_printoptions(precision=2)

        print("\nStart evaluation on VisTir\n")

        for i, pair in tqdm(
            enumerate(pairs),
            smoothing=0.1,
            total=len(pairs),
        ):
            if debug and i > 10:
                break

            T_0to1 = pair["T_0to1"]
            im0 = str(data_root / pair["im0"])
            im1 = str(data_root / pair["im1"])

            match_res = matcher(
                im0,
                im1,
                pair["K0"],
                pair["K1"],
                pair["dist0"],
                pair["dist1"],
            )

            # Compatible keypoint extraction.
            mkpts0 = np.asarray(
                match_res.get("mkpts0", match_res.get("keypoints0", []))
            )
            mkpts1 = np.asarray(
                match_res.get("mkpts1", match_res.get("keypoints1", []))
            )

            # Prefer intrinsics returned by the wrapper. If absent, use GT input K.
            new_K0 = match_res.get("new_K0", match_res.get("K0", pair["K0"]))
            new_K1 = match_res.get("new_K1", match_res.get("K1", pair["K1"]))

            # ------------------------------------------------------------------
            # Reviewer-requested statistics, computed on ALL predicted matches.
            # ------------------------------------------------------------------
            gt_stats = compute_gt_match_statistics(
                mkpts0,
                mkpts1,
                T_0to1,
                new_K0,
                new_K1,
                epi_thr=epi_thr,
            )

            statis["num_matches"].append(gt_stats["num_matches"])
            statis["num_correct"].append(gt_stats["num_correct"])
            statis["correct_match_rate"].append(
                gt_stats["correct_match_rate"]
            )

            # ------------------------------------------------------------------
            # Original relative-pose evaluation + RANSAC IR (kept separately).
            # ------------------------------------------------------------------
            ret = estimate_pose(
                mkpts0,
                mkpts1,
                new_K0,
                new_K1,
                thresh=ransac_thres,
            )

            if ret is None:
                R, t, inliers = None, None, None
                t_err, R_err = np.inf, np.inf

                statis["failed"].append(i)
                statis["R_errs"].append(R_err)
                statis["t_errs"].append(t_err)
                statis["ransac_inlier_ratio"].append(0.0)
            else:
                R, t, inliers = ret
                t_err, R_err = relative_pose_error(T_0to1, R, t)

                statis["R_errs"].append(R_err)
                statis["t_errs"].append(t_err)

                if len(mkpts0) > 0:
                    ransac_inlier_ratio = float(np.asarray(inliers).sum()) / len(
                        mkpts0
                    )
                else:
                    ransac_inlier_ratio = 0.0

                statis["ransac_inlier_ratio"].append(ransac_inlier_ratio)

            if print_out:
                print(
                    f"#Matches={gt_stats['num_matches']} "
                    f"#Correct={gt_stats['num_correct']} "
                    f"CMR@{epi_thr:.1e}="
                    f"{100.0 * gt_stats['correct_match_rate']:.2f}% "
                    f"RANSAC-IR="
                    f"{100.0 * statis['ransac_inlier_ratio'][-1]:.2f}% "
                    f"R={R_err:.3f}, t={t_err:.3f}"
                )

            if save_figs and max_save_figs > 0 and i < max_save_figs:
                img0_name = (
                    f"{'vis' if 'visible' in pair['im0'] else 'tir'}_"
                    f"{osp.basename(pair['im0']).split('.')[0]}"
                )
                img1_name = (
                    f"{'vis' if 'visible' in pair['im1'] else 'tir'}_"
                    f"{osp.basename(pair['im1']).split('.')[0]}"
                )
                fig_path = osp.join(scene_dir, f"{img0_name}_{img1_name}.jpg")

                save_matching_figure(
                    path=fig_path,
                    img0=(
                        match_res["img0_undistorted"]
                        if "img0_undistorted" in match_res
                        else match_res["img0"]
                    ),
                    img1=(
                        match_res["img1_undistorted"]
                        if "img1_undistorted" in match_res
                        else match_res["img1"]
                    ),
                    mkpts0=mkpts0,
                    mkpts1=mkpts1,
                    T_0to1=T_0to1,
                    K0=new_K0,
                    K1=new_K1,
                    t_err=t_err,
                    R_err=R_err,
                    name=method,
                    epi_thr=epi_thr,
                )

        # ----------------------------------------------------------------------
        # Scene-level correspondence statistics
        # ----------------------------------------------------------------------
        num_evaluated = len(statis["num_matches"])
        denom = max(num_evaluated, 1)

        avg_matches = float(np.sum(statis["num_matches"]) / denom)
        avg_correct_matches = float(np.sum(statis["num_correct"]) / denom)
        avg_correct_match_rate = float(
            np.sum(statis["correct_match_rate"]) / denom
        )
        avg_ransac_inlier_ratio = float(
            np.sum(statis["ransac_inlier_ratio"]) / denom
        )

        scene_match_stats[scene_name] = {
            "num_pairs": int(num_evaluated),
            "failed_pairs": int(len(statis["failed"])),
            "avg_matches": avg_matches,
            "avg_correct_matches": avg_correct_matches,
            "avg_correct_match_rate": avg_correct_match_rate,
            "avg_ransac_inlier_ratio": avg_ransac_inlier_ratio,
            # Keep sums for pair-weighted group aggregation.
            "sum_matches": float(np.sum(statis["num_matches"])),
            "sum_correct_matches": float(np.sum(statis["num_correct"])),
            "sum_correct_match_rate": float(
                np.sum(statis["correct_match_rate"])
            ),
            "sum_ransac_inlier_ratio": float(
                np.sum(statis["ransac_inlier_ratio"])
            ),
        }

        print(
            f"\nScene: {scene_name} "
            f"Evaluated: {num_evaluated} "
            f"Failed pose pairs: {len(statis['failed'])}"
        )
        print(
            f"  Avg. Matches: {avg_matches:.2f}\n"
            f"  Avg. Correct Matches: {avg_correct_matches:.2f}\n"
            f"  CMR@{epi_thr:.1e}: "
            f"{100.0 * avg_correct_match_rate:.2f}%\n"
            f"  Avg. RANSAC Inlier Ratio: "
            f"{100.0 * avg_ransac_inlier_ratio:.2f}%"
        )

        # Original pose AUC.
        pose_errors = np.max(
            np.stack([statis["R_errs"], statis["t_errs"]]),
            axis=0,
        )
        pose_auc = error_auc(pose_errors, thresholds)
        scene_pose_auc[scene_name] = 100 * np.array(
            [pose_auc[f"auc@{t}"] for t in thresholds]
        )
        print(f"{scene_name} {pose_auc}")

    # --------------------------------------------------------------------------
    # Aggregate scene results into cloudy_cloudy / cloudy_sunny groups.
    # --------------------------------------------------------------------------
    agg_pose_auc = aggregiate_scenes(scene_pose_auc, thresholds)
    aggregate_match_stats = aggregate_match_statistics(scene_match_stats)

    print("\n============================================================")
    print("Aggregated correspondence statistics")
    print("============================================================")

    for group_name, stats in aggregate_match_stats.items():
        print(
            f"[{group_name}] "
            f"Pairs={stats['num_pairs']}, "
            f"Failed={stats['failed_pairs']}, "
            f"Avg. Matches={stats['avg_matches']:.2f}, "
            f"Avg. Correct Matches={stats['avg_correct_matches']:.2f}, "
            f"CMR={100.0 * stats['avg_correct_match_rate']:.2f}%, "
            f"RANSAC-IR={100.0 * stats['avg_ransac_inlier_ratio']:.2f}%"
        )

    return (
        scene_pose_auc,
        agg_pose_auc,
        scene_match_stats,
        aggregate_match_stats,
    )


# ============================================================================
# Main benchmark wrapper
# ============================================================================
def test_relative_pose_vistir(
    data_root_dir,
    method="xoftr",
    exp_name="VisTIR",
    ransac_thres=1.5,
    print_out=False,
    save_dir="./results_relative_pose_geosam",
    save_figs=False,
    debug=False,
    args=None,
):
    print(f"[DEBUG] save_dir = {save_dir}, type = {type(save_dir)}")
    print(f"[DEBUG] method = {method}, type = {type(method)}")
    print(f"[DEBUG] save_dir is None: {save_dir is None}")
    print(f"[DEBUG] method is None: {method is None}")

    method_dir = osp.join(save_dir, method)
    if not osp.exists(method_dir):
        os.makedirs(method_dir)

    counter = 0
    path = osp.join(method_dir, f"{exp_name}" + "_{}")
    while osp.exists(path.format(counter)):
        counter += 1

    exp_dir = path.format(counter)
    os.mkdir(exp_dir)

    results_file = osp.join(exp_dir, "results.json")
    match_stats_csv = osp.join(exp_dir, "match_statistics.csv")
    figures_dir = osp.join(exp_dir, "match_figures")

    if save_figs and args.max_save_figs > 0:
        os.mkdir(figures_dir)

    # Dataset paths.
    npz_root = data_root_dir / "index/scene_info_test/"
    npz_list = data_root_dir / "index/val_test_list/test_list.txt"
    data_root = data_root_dir

    scene_pairs = load_vis_tir_pairs_npz(npz_root, npz_list)

    # This file currently contains only load_xoftr().
    matcher = eval(f"load_{method}")(args)

    thresholds = [5, 10, 20]

    (
        scene_pose_auc,
        agg_pose_auc,
        scene_match_stats,
        aggregate_match_stats,
    ) = eval_relapose(
        matcher,
        data_root,
        scene_pairs,
        ransac_thres=ransac_thres,
        thresholds=thresholds,
        save_figs=save_figs,
        figures_dir=figures_dir,
        method=method,
        print_out=print_out,
        debug=debug,
        epi_thr=args.epi_thr,
        max_save_figs=args.max_save_figs,
    )

    # Save all results.
    results = OrderedDict(
        {
            "method": method,
            "exp_name": exp_name,
            "ransac_thres": ransac_thres,
            "auc_thresholds": thresholds,
        }
    )

    results.update(
        {key: value for key, value in vars(args).items() if key not in results}
    )
    results.update({key: value.tolist() for key, value in agg_pose_auc.items()})
    results.update({key: value.tolist() for key, value in scene_pose_auc.items()})

    results["epi_thr"] = float(args.epi_thr)
    results["scene_match_statistics"] = scene_match_stats
    results["aggregate_match_statistics"] = aggregate_match_stats

    print(f"Results: {json.dumps(results, indent=4)}")

    with open(results_file, "w") as outfile:
        json.dump(results, outfile, indent=4)

    save_match_statistics_csv(
        match_stats_csv,
        scene_match_stats,
        aggregate_match_stats,
    )

    print(f"Results saved to {results_file}")
    print(f"Matching statistics saved to {match_stats_csv}")


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":

    def add_common_arguments(parser):
        parser.add_argument("--gpu", "-gpu", type=str, default="0")
        parser.add_argument("--exp_name", type=str, default="VisTIR")
        parser.add_argument("--method", type=str, default="xoftr")
        parser.add_argument(
            "--data_root_dir",
            type=str,
            default="./data/METU_VisTIR/",
        )
        parser.add_argument(
            "--save_dir",
            type=str,
            default=(
                "./isprs_ablation/"
                "results_relative_pose_geosam"
            ),
        )
        parser.add_argument("--ransac_thres", type=float, default=1.5)
        parser.add_argument(
            "--epi_thr",
            type=float,
            default=1e-4,
            help=(
                "GT symmetric epipolar-error threshold used to define a "
                "correct match for Avg. Correct Matches and CMR"
            ),
        )
        parser.add_argument("--print_out", action="store_true")
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--save_figs", action="store_true")
        parser.add_argument(
            "--max_save_figs",
            type=int,
            default=10,
            help="Maximum number of matching figures saved per scene; 0 disables saving",
        )
        parser.add_argument("--match_threshold", type=float, default=0.3)
        parser.add_argument("--fine_threshold", type=float, default=0.1)
        parser.add_argument(
            "--ckpt",
            type=str,
            default=(
                "./logs/tb_logs/dinomatch/version_32/checkpoints/epoch=27-auc@5=0.206-auc@10=0.378-auc@20=0.554.ckpt"

            ),
        )

    def add_xoftr_arguments(subparsers):
        subcommand = subparsers.add_parser("xoftr")
        add_common_arguments(subcommand)

    parser = argparse.ArgumentParser(description="Benchmark Relative Pose")
    add_common_arguments(parser)

    # Preserve the command structure of your original script.
    subparsers = parser.add_subparsers(dest="subcommand")
    add_xoftr_arguments(subparsers)

    args = parser.parse_args()

    # If the optional positional subcommand `xoftr` is used, the method remains xoftr.
    if getattr(args, "subcommand", None) == "xoftr":
        args.method = "xoftr"
# Set CUDA_VISIBLE_DEVICES in the shell when selecting GPUs.

    tt = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_relative_pose_vistir(
            Path(args.data_root_dir),
            args.method,
            args.exp_name,
            ransac_thres=args.ransac_thres,
            print_out=args.print_out,
            save_dir=args.save_dir,
            save_figs=args.save_figs,
            debug=args.debug,
            args=args,
        )

    print(f"Elapsed time: {time.time() - tt}")
