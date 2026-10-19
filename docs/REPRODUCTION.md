# Reproduction notes

## 1. Environment

Create the environment from the repository root:

```bash
conda env create -f environment.yaml
conda activate geosam
```

The reference setup is Linux + CUDA 11.8 + PyTorch 2.0.1. Set `PYTHONPATH` to the repository root when running scripts from another working directory:

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

## 2. Paths and weights

The repository does not contain datasets, logs, or checkpoints. Keep datasets in `data/` and checkpoints in `weights/` or pass absolute paths through command-line arguments. DINOv3 is loaded from a local Torch Hub checkout; update the relative paths in `src/xoftr/backbone/` if your checkout uses a different location.

## 3. METU-VisTIR evaluation

Place the METU-VisTIR image folders and index files under `data/METU_VisTIR/` and run:

```bash
python test_relative_pose.py xoftr \
  --data_root_dir data/METU_VisTIR \
  --ckpt /path/to/geosam.ckpt \
  --save_dir results/metu_vistir \
  --save_figs
```

The command evaluates relative pose at AUC thresholds 5, 10, and 20 degrees. Use `--debug` for a short smoke run and omit `--save_figs` when only metrics are required.

## 4. Training

The main configuration combines MegaDepth training pairs with METU-VisTIR validation:

```bash
python train.py \
  configs/data/megadepth_vistir_trainval_640.py \
  configs/xoftr/outdoor/visible_thermal.py \
  --gpus 4 --num_nodes 1 --accelerator ddp \
  --batch_size 2 --num_workers 16 --max_epochs 30
```

For a single-GPU development run, use `--gpus 1 --accelerator gpu` and a small `--limit_train_batches` value. Training logs and checkpoints are written below `logs/`, which is ignored by Git.
