# UMSS:Towards Unsupervised Multimodal Semantic Segmentation

<p align="center">
  <img src="Figures/framework.png" alt="UniM2 framework" width="75%">
</p>

<p align="center">
  <b>Official code for UniM2 on the UMSS task.</b>
</p>

<p align="center">
  <a href="#environments"><b>Environments</b></a> |
  <a href="#datasets"><b>Datasets</b></a> |
  <a href="#data-preparation"><b>Data Preparation</b></a> |
  <a href="#hyperparameter-search"><b>Hyperparameter Search</b></a> |
  <a href="#training"><b>Training</b></a> |
  <a href="#evaluation"><b>Evaluation</b></a>
</p>

This repository has been simplified for the released workflow. It keeps the
three datasets used in the paper: **NYU Depth V2**, **MFNet**, and **MCubeS**.

Datasets, checkpoints, and DINOv3 pretrained weights are not included in the
repository. The default configs use project-relative paths:

```text
data/
pretrained/dinov3_*.pth
save_checkpoints/<dataset>/*.ckpt
```

Any path can be overridden from the command line, for example
`pytorch_data_dir=/path/to/NYU_Depth` or `pretrained_weights=/path/to/dinov3.pth`.

## Environments

UniM2 uses a single Conda environment named `UMSS`.

```bash
conda env create -f environment.yml
conda activate UMSS
```

## Datasets

Please download the prepared dataset archives from our OneDrive links and place
them under `data/` as shown below. If your datasets live elsewhere, keep the
same internal folder structure and pass `pytorch_data_dir=/your/path`.

| Dataset | Download | Modalities Used | Config Key | Expected Root |
| :-- | :-- | :-- | :-- | :-- |
| **[NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)** | **[OneDrive](https://entuedu-my.sharepoint.com/:u:/r/personal/haitian003_e_ntu_edu_sg/Documents/Project-Datasets-and-Checkpoints/UMSS/Datasets/NYU_Depth.zip?csf=1&web=1&e=7A4yoL)** | RGB + HHA/depth | `dataset_name: nyu` | `data/NYU_Depth/nyu` |
| **[MFNet](https://github.com/haqishen/MFNet-pytorch)** | **[OneDrive](https://entuedu-my.sharepoint.com/:u:/r/personal/haitian003_e_ntu_edu_sg/Documents/Project-Datasets-and-Checkpoints/UMSS/Datasets/MFNet.zip?csf=1&web=1&e=qgCWpa)** | RGB + thermal | `dataset_name: mfnet` | `data/MFNet/mfnet` |
| **[MCubeS](https://github.com/kyotovision-public/multimodal-material-segmentation)** | **[OneDrive](https://entuedu-my.sharepoint.com/:u:/r/personal/haitian003_e_ntu_edu_sg/Documents/Project-Datasets-and-Checkpoints/UMSS/Datasets/MCUBES.zip?csf=1&web=1&e=85hZ7y)** | RGB + AoLP/DoLP/NIR | `dataset_name: mcubes` | `data/MCUBES/MCubeS` |

The expected project layout is:

```text
data/
|-- NYU_Depth/
|   `-- nyu/
|       |-- RGB/
|       |-- HHA/
|       |-- Labels/
|       |-- train.txt
|       `-- val.txt
|-- MFNet/
|   `-- mfnet/
|       |-- images/
|       |-- labels/
|       |-- train.txt
|       `-- test.txt
`-- MCUBES/
    `-- MCubeS/
        |-- polL_color/
        |-- polL_aolp_sin/
        |-- polL_aolp_cos/
        |-- polL_dolp/
        |-- NIR_warped/
        |-- GT/
        `-- list_folder/
```

MFNet stores RGB and thermal data in one 4-channel PNG. Keep the original
`images/*.png` files; UniM2 reads RGB from the first three channels and thermal
from the fourth channel.

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
the checkpoint and dataset you want to evaluate. Raw evaluation is used by
default; set `run_crf=true` to enable CRF post-processing.
