import os
import torch


INPUT_CKPT = (
    "./logs/tb_logs/"
    "dinomatch/version_51_Standard_Cross_Attention/checkpoints/"
    "epoch=20-auc@5=0.181-auc@10=0.341-auc@20=0.520.ckpt"
)

OUTPUT_CKPT = (
    "./logs/tb_logs/"
    "dinomatch/version_51_Standard_Cross_Attention/checkpoints/"
    "epoch=20-clean.ckpt"
)


def remove_prefix(key):
    prefixes = [
        "matcher.",
        "module.",
        "model.",
    ]

    new_key = key

    # 循环处理，兼容 matcher.module.xxx 等情况
    changed = True

    while changed:
        changed = False

        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                changed = True

    return new_key


def main():
    if not os.path.isfile(INPUT_CKPT):
        raise FileNotFoundError(
            f"找不到输入权重：{INPUT_CKPT}"
        )

    checkpoint = torch.load(
        INPUT_CKPT,
        map_location="cpu",
    )

    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    clean_state_dict = {}

    renamed_count = 0

    for key, value in state_dict.items():
        new_key = remove_prefix(key)

        if new_key != key:
            renamed_count += 1

        if new_key in clean_state_dict:
            raise RuntimeError(
                f"清理前缀后出现重复参数名：{new_key}"
            )

        clean_state_dict[new_key] = value

    # 保留原checkpoint中的其他信息，只替换state_dict
    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        output_checkpoint = checkpoint.copy()
        output_checkpoint["state_dict"] = clean_state_dict
    else:
        output_checkpoint = {
            "state_dict": clean_state_dict
        }

    os.makedirs(
        os.path.dirname(OUTPUT_CKPT),
        exist_ok=True,
    )

    torch.save(
        output_checkpoint,
        OUTPUT_CKPT,
    )

    print("=" * 70)
    print(f"Input checkpoint : {INPUT_CKPT}")
    print(f"Output checkpoint: {OUTPUT_CKPT}")
    print(f"Total parameters : {len(state_dict)}")
    print(f"Renamed keys     : {renamed_count}")

    old_examples = list(state_dict.keys())[:5]
    new_examples = list(clean_state_dict.keys())[:5]

    print("\nBefore:")
    for key in old_examples:
        print(f"  {key}")

    print("\nAfter:")
    for key in new_examples:
        print(f"  {key}")

    print("\nCheckpoint conversion completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()