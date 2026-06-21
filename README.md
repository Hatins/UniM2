# UniM2: Unsupervised Multimodal Semantic Segmentation

This repository contains the code for **UniM2** on the **UMSS** task.

The codebase has been simplified for the released workflow and keeps only the
datasets used in the paper:

- NYU Depth V2 (`nyu`)
- MFNet (`mfnet`)
- MCubeS (`mcubes`)

Datasets, checkpoints, and DINOv3 pretrained weights are assumed to be downloaded already.
The default configs use project-relative paths:

```text
data/NYU_Depth
data/MFNet
data/MCUBES
pretrained/dinov3_*.pth
save_checkpoints/<dataset>/*.ckpt
```

You can override any path from the command line, for example
`pytorch_data_dir=/path/to/NYU_Depth` or `pretrained_weights=/path/to/dinov3.pth`.

## Environment

```bash
conda env create -f environment.yml
conda activate UMSS
```

## Data Preparation

If cropped data has not been generated yet, run:

```bash
python src/crop_datasets.py --config-name train_config_nyu.yml
```

Then precompute nearest neighbors for contrastive positive samples:

```bash
python src/precompute_knns.py --config-name train_config_nyu.yml
```

Swap `train_config_nyu.yml` for `train_config_mfnet.yml` or
`train_config_mcubes.yml` as needed.

## Hyperparameter Search

The recommended workflow is to search hyperparameters first:

```bash
python src/hyperparameter_search.py \
  --config_name train_config_nyu.yml \
  --max_steps 7500 \
  --n_trials 200
```

Search results are written to `optuna_results/`.

## Training

After choosing hyperparameters, update the matching config file and train:

```bash
python src/train_segmentation.py --config-name train_config_nyu.yml
```

## Evaluation

```bash
python src/eval_segmentation.py --config-name eval_config.yml
```

Set `model_paths` and `pytorch_data_dir` in `src/configs/eval_config.yml` for
the checkpoint and dataset you want to evaluate. Evaluation uses CRF
post-processing by default; pass `run_crf=false` to report raw probe outputs.
