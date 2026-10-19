import os

# ============================================================
# GPU
# ============================================================
# Set CUDA_VISIBLE_DEVICES in the shell when selecting GPUs.

import sys
import torch
import numpy as np


from src.xoftr import (
    XoFTR_dino_resnet_panmatch_light_withfpn
)

from src.config.default import get_cfg_defaults

from src.utils.data_io import lower_config


# ============================================================
# 1. 修改这里：Standard CA checkpoint
# ============================================================

CKPT = (
    "./logs/tb_logs/dinomatch/version_51_Standard_Cross_Attention/checkpoints/epoch=20-clean.ckpt"
)


# ============================================================
# 输出报告
# ============================================================

REPORT_FILE = "inspect_standard_CA_checkpoint.txt"


# ============================================================
# 是否真正把 checkpoint 权重加载进模型
#
# True:
#   检查完成后，将所有兼容参数加载进入当前模型，
#   并再次报告 load_state_dict 信息。
#
# 推荐保持 True
# ============================================================

LOAD_COMPATIBLE_WEIGHTS = True


# ============================================================
# Tee:
# 同时输出到终端和 txt
# ============================================================

class Tee:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(
            filename,
            "w",
            encoding="utf-8"
        )

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = Tee(REPORT_FILE)


# ============================================================
# Build Standard CA model
#
# IMPORTANT:
#
# 当前你的网络代码应该满足：
#
# structure_extractor = "none"
# rope = False
#
# 也就是：
#
# w/o Structural Extraction & 2D RoPE
#
# ============================================================

def build_model():

    config = get_cfg_defaults(
        inference=True
    )

    config = lower_config(
        config
    )

    # 与你正式实验保持一致
    config["xoftr"]["match_coarse"]["thr"] = 0.3
    config["xoftr"]["fine"]["thr"] = 0.1
    config["xoftr"]["fine"]["denser"] = False

    model = (
        XoFTR_dino_resnet_panmatch_light_withfpn(
            config=config["xoftr"]
        )
    )

    return model


# ============================================================
# 获取 checkpoint state_dict
# ============================================================

def get_checkpoint_state_dict(
        ckpt_path
):

    print("\n")
    print("=" * 100)
    print("STEP 1: LOAD CHECKPOINT")
    print("=" * 100)

    print(
        "Checkpoint path:"
    )
    print(
        ckpt_path
    )

    if not os.path.exists(
        ckpt_path
    ):
        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{ckpt_path}"
        )

    ckpt = torch.load(
        ckpt_path,
        map_location="cpu"
    )

    print(
        "\nCheckpoint object type:"
    )

    print(
        type(ckpt)
    )

    if isinstance(
        ckpt,
        dict
    ):

        print(
            "\nCheckpoint top-level keys:"
        )

        for k in ckpt.keys():
            print(
                "   ",
                k
            )

    # Lightning checkpoint
    if (
        isinstance(ckpt, dict)
        and
        "state_dict" in ckpt
    ):

        state_dict = (
            ckpt["state_dict"]
        )

        print(
            "\nUsing checkpoint['state_dict']."
        )

    else:

        state_dict = ckpt

        print(
            "\nUsing checkpoint itself "
            "as state_dict."
        )

    print(
        "\nNumber of checkpoint tensors:",
        len(state_dict)
    )

    return (
        ckpt,
        state_dict
    )


# ============================================================
# 打印前几个 keys
# ============================================================

def print_sample_keys(
        state_dict,
        model_state
):

    print("\n")
    print("=" * 100)
    print("STEP 2: CHECK KEY NAMES")
    print("=" * 100)

    print(
        "\nFirst 30 CHECKPOINT keys:"
    )

    for i, k in enumerate(
        state_dict.keys()
    ):

        if i >= 30:
            break

        print(
            f"[{i:02d}] {k}"
        )

    print(
        "\nFirst 30 CURRENT MODEL keys:"
    )

    for i, k in enumerate(
        model_state.keys()
    ):

        if i >= 30:
            break

        print(
            f"[{i:02d}] {k}"
        )


# ============================================================
# Prefix清理
# ============================================================

def remove_prefix(
        key,
        prefix
):

    if (
        prefix
        and
        key.startswith(prefix)
    ):

        return key[
            len(prefix):
        ]

    return key


# ============================================================
# 自动寻找 checkpoint key prefix
# ============================================================

def find_best_prefix(
        state_dict,
        model_state
):

    print("\n")
    print("=" * 100)
    print("STEP 3: FIND BEST CHECKPOINT PREFIX")
    print("=" * 100)

    model_keys = set(
        model_state.keys()
    )

    candidate_prefixes = [
        "",
        "model.",
        "matcher.",
        "module.",
        "xoftr.",
        "model.model.",
        "model.matcher.",
        "matcher.model.",
        "module.model.",
        "module.matcher.",
    ]

    best_prefix = ""
    best_matches = -1

    for prefix in candidate_prefixes:

        matched = 0

        for k in state_dict.keys():

            nk = remove_prefix(
                k,
                prefix
            )

            if nk in model_keys:
                matched += 1

        print(
            f"prefix="
            f"{repr(prefix):20s} "
            f"matched={matched}"
        )

        if matched > best_matches:

            best_matches = matched
            best_prefix = prefix

    print("\nBest prefix:")
    print(
        repr(best_prefix)
    )

    print(
        "Matched keys with best prefix:",
        best_matches
    )

    return best_prefix


# ============================================================
# Clean state dict
# ============================================================

def clean_state_dict(
        state_dict,
        prefix
):

    cleaned = {}

    for k, v in state_dict.items():

        nk = remove_prefix(
            k,
            prefix
        )

        cleaned[nk] = v

    return cleaned


# ============================================================
# Compare model/checkpoint
# ============================================================

def compare_state_dicts(
        model_state,
        cleaned_state
):

    print("\n")
    print("=" * 100)
    print("STEP 4: MODEL / CHECKPOINT COMPATIBILITY")
    print("=" * 100)

    matched = []
    missing = []
    unexpected = []
    shape_mismatch = []

    # --------------------------------------------------------
    # model -> checkpoint
    # --------------------------------------------------------

    for k, model_tensor in (
        model_state.items()
    ):

        if k not in cleaned_state:

            missing.append(
                k
            )

        else:

            ckpt_tensor = (
                cleaned_state[k]
            )

            if (
                tuple(model_tensor.shape)
                !=
                tuple(ckpt_tensor.shape)
            ):

                shape_mismatch.append(
                    (
                        k,
                        tuple(
                            model_tensor.shape
                        ),
                        tuple(
                            ckpt_tensor.shape
                        )
                    )
                )

            else:

                matched.append(
                    k
                )

    # --------------------------------------------------------
    # checkpoint -> model
    # --------------------------------------------------------

    for k in cleaned_state.keys():

        if k not in model_state:

            unexpected.append(
                k
            )

    print(
        "\nCurrent model tensors:",
        len(model_state)
    )

    print(
        "Checkpoint tensors:",
        len(cleaned_state)
    )

    print(
        "Matched:",
        len(matched)
    )

    print(
        "Missing:",
        len(missing)
    )

    print(
        "Unexpected:",
        len(unexpected)
    )

    print(
        "Shape mismatch:",
        len(shape_mismatch)
    )

    matched_ratio = (
        100.0
        *
        len(matched)
        /
        max(
            len(model_state),
            1
        )
    )

    print(
        "\nMatched ratio:"
    )

    print(
        f"{matched_ratio:.2f}%"
    )

    # --------------------------------------------------------
    # missing
    # --------------------------------------------------------

    if len(missing) > 0:

        print("\n")
        print("-" * 100)
        print("MISSING KEYS")
        print("-" * 100)

        for k in missing:
            print(
                k
            )

    else:

        print(
            "\nMissing keys: NONE"
        )

    # --------------------------------------------------------
    # unexpected
    # --------------------------------------------------------

    if len(unexpected) > 0:

        print("\n")
        print("-" * 100)
        print("UNEXPECTED KEYS")
        print("-" * 100)

        for k in unexpected:
            print(
                k
            )

    else:

        print(
            "\nUnexpected keys: NONE"
        )

    # --------------------------------------------------------
    # shape mismatch
    # --------------------------------------------------------

    if len(shape_mismatch) > 0:

        print("\n")
        print("-" * 100)
        print("SHAPE MISMATCH")
        print("-" * 100)

        for (
            k,
            model_shape,
            ckpt_shape
        ) in shape_mismatch:

            print(
                f"{k}"
            )

            print(
                f"    model : "
                f"{model_shape}"
            )

            print(
                f"    ckpt  : "
                f"{ckpt_shape}"
            )

    else:

        print(
            "\nShape mismatch: NONE"
        )

    return (
        matched,
        missing,
        unexpected,
        shape_mismatch
    )


# ============================================================
# 检查当前 ablation 配置
# ============================================================

def inspect_ablation_configuration(
        model
):

    print("\n")
    print("=" * 100)
    print("STEP 5: CHECK ABLATION CONFIGURATION")
    print("=" * 100)

    extractor_modules = []
    rope_modules = []

    # --------------------------------------------------------
    # Structural extractor
    # --------------------------------------------------------

    for name, module in (
        model.named_modules()
    ):

        if hasattr(
            module,
            "extractor_type"
        ):

            extractor_modules.append(
                (
                    name,
                    module.extractor_type
                )
            )

    print(
        "\nStructural extractor modules:"
    )

    if len(
        extractor_modules
    ) == 0:

        print(
            "WARNING: "
            "No module with "
            "'extractor_type' found!"
        )

    for name, value in (
        extractor_modules
    ):

        print(
            f"{name}"
        )

        print(
            f"    extractor_type = "
            f"{value}"
        )

    # --------------------------------------------------------
    # RoPE
    # --------------------------------------------------------

    for name, module in (
        model.named_modules()
    ):

        # 避免打印其他不相关 rope 属性
        if (
            module.__class__.__name__
            ==
            "AG_RoPE_EncoderLayer"
        ):

            rope_modules.append(
                (
                    name,
                    module.rope,
                    module.agg_size0,
                    module.agg_size1
                )
            )

    print(
        "\nAG_RoPE_EncoderLayer modules:"
    )

    if len(
        rope_modules
    ) == 0:

        print(
            "WARNING: "
            "No AG_RoPE_EncoderLayer found!"
        )

    for (
        name,
        rope,
        agg0,
        agg1
    ) in rope_modules:

        print(
            f"{name}"
        )

        print(
            f"    rope      = "
            f"{rope}"
        )

        print(
            f"    agg_size0 = "
            f"{agg0}"
        )

        print(
            f"    agg_size1 = "
            f"{agg1}"
        )

    # --------------------------------------------------------
    # 自动判断
    # --------------------------------------------------------

    print("\n")
    print("-" * 100)
    print("ABLATION CONFIG CHECK")
    print("-" * 100)

    extractor_ok = True

    if len(
        extractor_modules
    ) == 0:

        extractor_ok = False

    for _, extractor_type in (
        extractor_modules
    ):

        if extractor_type != "none":
            extractor_ok = False

    rope_ok = True

    if len(
        rope_modules
    ) != 4:

        rope_ok = False

    for _, rope, _, _ in (
        rope_modules
    ):

        if rope is not False:
            rope_ok = False

    if extractor_ok:
        print(
            "[PASS] Structural Extraction "
            "is disabled."
        )
    else:
        print(
            "[WARNING] Structural Extraction "
            "configuration is NOT fully 'none'."
        )

    if rope_ok:
        print(
            "[PASS] All four GGSA layers "
            "have rope=False."
        )
    else:
        print(
            "[WARNING] RoPE configuration "
            "is not the expected 4 x False."
        )

    if (
        extractor_ok
        and
        rope_ok
    ):

        print(
            "\n[PASS] Current model corresponds to:"
        )

        print(
            "w/o Structural Extraction "
            "& 2D RoPE"
        )

    else:

        print(
            "\n[WARNING] Current architecture "
            "does NOT exactly correspond to "
            "the intended Standard CA ablation."
        )


# ============================================================
# 参数统计函数
# ============================================================

def tensor_statistics(
        tensor
):

    x = (
        tensor
        .detach()
        .float()
        .cpu()
    )

    n = x.numel()

    if n == 0:

        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "norm": np.nan,
            "zero_ratio": np.nan,
            "nan": False,
            "inf": False,
        }

    mean = (
        x.mean().item()
    )

    # 单元素tensor避免std warning
    if n > 1:

        std = (
            x.std(
                unbiased=False
            ).item()
        )

    else:

        std = 0.0

    min_v = (
        x.min().item()
    )

    max_v = (
        x.max().item()
    )

    norm = (
        x.norm().item()
    )

    zero_ratio = (
        (x == 0)
        .float()
        .mean()
        .item()
    )

    has_nan = bool(
        torch.isnan(x).any()
    )

    has_inf = bool(
        torch.isinf(x).any()
    )

    return {
        "mean": mean,
        "std": std,
        "min": min_v,
        "max": max_v,
        "norm": norm,
        "zero_ratio": zero_ratio,
        "nan": has_nan,
        "inf": has_inf,
    }


# ============================================================
# 打印关键 checkpoint 参数
# ============================================================

def inspect_important_parameters(
        cleaned_state,
        model_state
):

    print("\n")
    print("=" * 100)
    print("STEP 6: INSPECT IMPORTANT PARAMETERS")
    print("=" * 100)

    keywords = [

        # GGSA
        "resize",

        "q_proj",
        "k_proj",
        "v_proj",
        "merge",
        "aggregate",

        # Structural branch
        "masg_extractor",
        "cnn_encoder",

        # FPN
        "cnn_decoder",
        "fuseconv",
        "light_proj",

        # saliency
        "saliency",

    ]

    num_printed = 0

    bad_params = []

    print(
        "\nFormat:"
    )

    print(
        "name | shape | mean | std | "
        "norm | zero_ratio"
    )

    print(
        "-" * 150
    )

    for name, tensor in (
        cleaned_state.items()
    ):

        if name not in model_state:
            continue

        if not any(
            key in name
            for key in keywords
        ):
            continue

        if not torch.is_tensor(
            tensor
        ):
            continue

        stats = (
            tensor_statistics(
                tensor
            )
        )

        print(
            f"{name:100s} "
            f"| "
            f"{str(tuple(tensor.shape)):20s} "
            f"| "
            f"mean={stats['mean']: .6e} "
            f"| "
            f"std={stats['std']: .6e} "
            f"| "
            f"norm={stats['norm']: .6e} "
            f"| "
            f"zero={stats['zero_ratio'] * 100:6.2f}%"
        )

        num_printed += 1

        # ----------------------------------------------------
        # 自动标记危险权重
        # ----------------------------------------------------

        if (
            stats["nan"]
            or
            stats["inf"]
            or
            stats["norm"] == 0
        ):

            bad_params.append(
                (
                    name,
                    stats
                )
            )

    print(
        "\nNumber of important parameters inspected:",
        num_printed
    )

    print("\n")
    print("-" * 100)
    print("PARAMETER HEALTH CHECK")
    print("-" * 100)

    if len(
        bad_params
    ) == 0:

        print(
            "[PASS] No NaN / Inf / "
            "completely-zero important parameters found."
        )

    else:

        print(
            "[WARNING] Suspicious parameters found:"
        )

        for name, stats in (
            bad_params
        ):

            print(
                name,
                stats
            )


# ============================================================
# 专门打印四层 Q/K/V
# ============================================================

def inspect_qkv(
        cleaned_state
):

    print("\n")
    print("=" * 100)
    print("STEP 7: CHECK FOUR GGSA Q/K/V LAYERS")
    print("=" * 100)

    for i in range(4):

        print(
            f"\n----- GGSA Layer {i} -----"
        )

        suffixes = [
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "merge.weight",
            "aggregate.weight",
        ]

        for suffix in suffixes:

            candidates = []

            # 不假定完整前缀，按字符串找
            target = (
                f"resize.{i}."
                f"{suffix}"
            )

            for k in (
                cleaned_state.keys()
            ):

                if target in k:
                    candidates.append(
                        k
                    )

            if len(
                candidates
            ) == 0:

                # 第4层 agg_size=1，
                # aggregate 是 Identity，
                # 所以没有 aggregate.weight 是正常的
                if (
                    suffix
                    ==
                    "aggregate.weight"
                    and
                    i == 3
                ):

                    print(
                        f"{suffix:20s}: "
                        f"NOT PRESENT "
                        f"(expected: "
                        f"agg_size0=1 -> Identity)"
                    )

                else:

                    print(
                        f"{suffix:20s}: "
                        f"NOT FOUND"
                    )

                continue

            for k in candidates:

                tensor = (
                    cleaned_state[k]
                )

                stats = (
                    tensor_statistics(
                        tensor
                    )
                )

                print(
                    f"{suffix:20s}: "
                    f"shape="
                    f"{str(tuple(tensor.shape)):18s} "
                    f"std="
                    f"{stats['std']:.6e} "
                    f"norm="
                    f"{stats['norm']:.6e} "
                    f"zero="
                    f"{stats['zero_ratio']*100:.2f}%"
                )


# ============================================================
# 参数量
# ============================================================

def count_parameters(
        model
):

    print("\n")
    print("=" * 100)
    print("STEP 8: PARAMETER COUNT")
    print("=" * 100)

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    frozen = (
        total
        -
        trainable
    )

    print(
        f"\nTotal registered params : "
        f"{total:,}"
    )

    print(
        f"Trainable params        : "
        f"{trainable:,}"
    )

    print(
        f"Frozen registered params: "
        f"{frozen:,}"
    )

    print(
        "\nNOTE:"
    )

    print(
        "If your frozen DINO backbone is stored "
        "inside a normal Python list instead of "
        "nn.ModuleList, its parameters will not "
        "be included in this registered parameter count."
    )


# ============================================================
# 真正加载 compatible weights
# ============================================================

def load_compatible_weights(
        model,
        cleaned_state
):

    print("\n")
    print("=" * 100)
    print("STEP 9: LOAD COMPATIBLE CHECKPOINT WEIGHTS")
    print("=" * 100)

    current_state = (
        model.state_dict()
    )

    compatible_state = {}

    for k, v in (
        cleaned_state.items()
    ):

        if k not in current_state:
            continue

        if (
            tuple(v.shape)
            !=
            tuple(
                current_state[k].shape
            )
        ):
            continue

        compatible_state[k] = v

    print(
        "\nCompatible tensors to load:",
        len(compatible_state)
    )

    msg = (
        model.load_state_dict(
            compatible_state,
            strict=False
        )
    )

    print(
        "\nPyTorch load_state_dict:"
    )

    print(
        "\nmissing_keys:"
    )

    if len(
        msg.missing_keys
    ) == 0:

        print(
            "NONE"
        )

    else:

        for k in (
            msg.missing_keys
        ):
            print(
                k
            )

    print(
        "\nunexpected_keys:"
    )

    if len(
        msg.unexpected_keys
    ) == 0:

        print(
            "NONE"
        )

    else:

        for k in (
            msg.unexpected_keys
        ):
            print(
                k
            )


# ============================================================
# 检查加载之后参数是否真的一致
# ============================================================

def verify_loaded_weights(
        model,
        cleaned_state
):

    print("\n")
    print("=" * 100)
    print("STEP 10: VERIFY LOADED WEIGHTS")
    print("=" * 100)

    model_state = (
        model.state_dict()
    )

    checked = 0
    exact_match = 0
    different = []

    for k, ckpt_tensor in (
        cleaned_state.items()
    ):

        if k not in model_state:
            continue

        if (
            tuple(
                ckpt_tensor.shape
            )
            !=
            tuple(
                model_state[k].shape
            )
        ):

            continue

        checked += 1

        a = (
            model_state[k]
            .detach()
            .cpu()
        )

        b = (
            ckpt_tensor
            .detach()
            .cpu()
        )

        if torch.equal(
            a,
            b
        ):

            exact_match += 1

        else:

            max_diff = (
                (
                    a.float()
                    -
                    b.float()
                )
                .abs()
                .max()
                .item()
            )

            different.append(
                (
                    k,
                    max_diff
                )
            )

    print(
        "\nChecked tensors:",
        checked
    )

    print(
        "Exactly equal:",
        exact_match
    )

    print(
        "Different:",
        len(different)
    )

    if len(
        different
    ) == 0:

        print(
            "\n[PASS] All compatible checkpoint "
            "weights were loaded exactly."
        )

    else:

        print(
            "\n[WARNING] Some tensors differ "
            "after loading:"
        )

        for k, diff in (
            different[:50]
        ):

            print(
                f"{k}: "
                f"max_abs_diff="
                f"{diff:.8e}"
            )


# ============================================================
# Checkpoint training metadata
# ============================================================

def inspect_checkpoint_metadata(
        ckpt
):

    print("\n")
    print("=" * 100)
    print("STEP 11: CHECKPOINT TRAINING METADATA")
    print("=" * 100)

    if not isinstance(
        ckpt,
        dict
    ):

        print(
            "Checkpoint is not a dict."
        )

        return

    keys_of_interest = [

        "epoch",
        "global_step",
        "hyper_parameters",
        "loops",
        "callbacks",

    ]

    for k in keys_of_interest:

        if k in ckpt:

            print(
                f"\n[{k}]"
            )

            # callbacks/loops可能很长
            if k in [
                "loops",
                "callbacks"
            ]:

                print(
                    type(
                        ckpt[k]
                    )
                )

            else:

                print(
                    ckpt[k]
                )


# ============================================================
# Final summary
# ============================================================

def print_final_summary(
        matched,
        missing,
        unexpected,
        mismatch,
        model_state
):

    print("\n")
    print("=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    ratio = (
        100.0
        *
        len(matched)
        /
        max(
            len(model_state),
            1
        )
    )

    print(
        f"\nMatched ratio : "
        f"{ratio:.2f}%"
    )

    print(
        f"Missing       : "
        f"{len(missing)}"
    )

    print(
        f"Unexpected    : "
        f"{len(unexpected)}"
    )

    print(
        f"Shape mismatch: "
        f"{len(mismatch)}"
    )

    print("\nInterpretation:")

    if (
        ratio >= 99.0
        and
        len(mismatch) == 0
    ):

        print(
            "[GOOD] Checkpoint is highly compatible "
            "with the current Standard CA model."
        )

    elif (
        ratio >= 95.0
        and
        len(mismatch) == 0
    ):

        print(
            "[CHECK] Most parameters match, "
            "but inspect missing/unexpected keys."
        )

    else:

        print(
            "[WARNING] Checkpoint/model compatibility "
            "is questionable."
        )

    print(
        "\nFor your ablation, the ideal result is:"
    )

    print(
        "  1. Matched ratio close to 100%"
    )

    print(
        "  2. Shape mismatch = 0"
    )

    print(
        "  3. No unexpected missing Q/K/V/CNN/FPN weights"
    )

    print(
        "  4. extractor_type = none"
    )

    print(
        "  5. Four AG_RoPE_EncoderLayer modules "
        "all have rope=False"
    )

    print(
        "  6. Q/K/V weights have non-zero std/norm"
    )

    print(
        "  7. No NaN / Inf parameters"
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 100
    )

    print(
        "STANDARD CA CHECKPOINT SANITY CHECK"
    )

    print(
        "Ablation: "
        "w/o Structural Extraction & 2D RoPE"
    )

    print(
        "=" * 100
    )

    # --------------------------------------------------------
    # Build current model
    # --------------------------------------------------------

    print(
        "\nBuilding current model..."
    )

    model = build_model()

    model.eval()

    print(
        "Model built."
    )

    model_state = (
        model.state_dict()
    )

    # --------------------------------------------------------
    # checkpoint
    # --------------------------------------------------------

    ckpt, state_dict = (
        get_checkpoint_state_dict(
            CKPT
        )
    )

    # --------------------------------------------------------
    # keys
    # --------------------------------------------------------

    print_sample_keys(
        state_dict,
        model_state
    )

    # --------------------------------------------------------
    # prefix
    # --------------------------------------------------------

    best_prefix = (
        find_best_prefix(
            state_dict,
            model_state
        )
    )

    cleaned_state = (
        clean_state_dict(
            state_dict,
            best_prefix
        )
    )

    # --------------------------------------------------------
    # compatibility
    # --------------------------------------------------------

    (
        matched,
        missing,
        unexpected,
        mismatch
    ) = compare_state_dicts(
        model_state,
        cleaned_state
    )

    # --------------------------------------------------------
    # actual architecture flags
    # --------------------------------------------------------

    inspect_ablation_configuration(
        model
    )

    # --------------------------------------------------------
    # critical parameter statistics
    # --------------------------------------------------------

    inspect_important_parameters(
        cleaned_state,
        model_state
    )

    # --------------------------------------------------------
    # 4 layers QKV
    # --------------------------------------------------------

    inspect_qkv(
        cleaned_state
    )

    # --------------------------------------------------------
    # parameter count
    # --------------------------------------------------------

    count_parameters(
        model
    )

    # --------------------------------------------------------
    # load checkpoint
    # --------------------------------------------------------

    if LOAD_COMPATIBLE_WEIGHTS:

        load_compatible_weights(
            model,
            cleaned_state
        )

        verify_loaded_weights(
            model,
            cleaned_state
        )

    # --------------------------------------------------------
    # metadata
    # --------------------------------------------------------

    inspect_checkpoint_metadata(
        ckpt
    )

    # --------------------------------------------------------
    # summary
    # --------------------------------------------------------

    print_final_summary(
        matched,
        missing,
        unexpected,
        mismatch,
        model_state
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "Inspection finished."
    )

    print(
        "Report saved to:"
    )

    print(
        os.path.abspath(
            REPORT_FILE
        )
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":

    main()