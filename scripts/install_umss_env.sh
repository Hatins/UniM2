#!/usr/bin/env bash
set -euo pipefail

# UniM2 / UMSS environment installer.
# Default target: CUDA 12.8 PyTorch wheels for recent NVIDIA GPUs.
#
# Usage:
#   bash scripts/install_umss_env.sh
#
# Optional overrides:
#   ENV_NAME=UMSS FORCE_RECREATE=1 bash scripts/install_umss_env.sh
#   CUDA_WHEEL=cu128 TORCH_VERSION=2.11.0 bash scripts/install_umss_env.sh

ENV_NAME="${ENV_NAME:-UMSS}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
FORCE_RECREATE="${FORCE_RECREATE:-0}"

CUDA_WHEEL="${CUDA_WHEEL:-cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.26.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0}"
TORCH_INDEX_URL="https://download.pytorch.org/whl/${CUDA_WHEEL}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not available. Please install Miniconda/Anaconda first." >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is required to install pydensecrf from source." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  if [[ "${FORCE_RECREATE}" == "1" ]]; then
    echo "Removing existing conda environment: ${ENV_NAME}"
    conda env remove -n "${ENV_NAME}" -y
  else
    echo "ERROR: conda environment '${ENV_NAME}' already exists." >&2
    echo "Set FORCE_RECREATE=1 to remove and recreate it:" >&2
    echo "  FORCE_RECREATE=1 bash scripts/install_umss_env.sh" >&2
    exit 1
  fi
fi

echo "Creating conda environment: ${ENV_NAME}"
conda create -n "${ENV_NAME}" -y -c conda-forge \
  "python=${PYTHON_VERSION}" \
  "numpy=1.26.4" \
  "scipy=1.11.4" \
  pillow \
  tqdm \
  pip

echo "Installing PyTorch (${CUDA_WHEEL})"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip setuptools wheel
conda run -n "${ENV_NAME}" python -m pip install \
  --index-url "${TORCH_INDEX_URL}" \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}"

echo "Installing UniM2 Python dependencies"
conda run -n "${ENV_NAME}" python -m pip install \
  "hydra-core==1.3.2" \
  "omegaconf==2.3.0" \
  "optuna==4.4.0" \
  "pytorch-lightning==2.5.2" \
  "torchmetrics==1.7.4" \
  "wandb==0.19.11" \
  "opencv-python-headless==4.8.1.78" \
  "Cython<3"

# pydensecrf's PyPI sdist is fragile with modern build isolation/Cython.
# Installing from the upstream git repository with Cython<3 is more reliable.
echo "Installing pydensecrf"
conda run -n "${ENV_NAME}" python -m pip install \
  --no-build-isolation \
  git+https://github.com/lucasb-eyer/pydensecrf.git

# Keep NumPy pinned after all pip installs. Some OpenCV wheels may otherwise
# pull NumPy 2.x, which can break SciPy/torchmetrics in this project.
conda run -n "${ENV_NAME}" python -m pip install --force-reinstall \
  "numpy==1.26.4" \
  "opencv-python-headless==4.8.1.78"

echo "Validating environment"
conda run -n "${ENV_NAME}" python -m pip check
conda run -n "${ENV_NAME}" python - <<'PY'
import cv2
import hydra
import numpy
import optuna
import pytorch_lightning
import scipy
import torch
import torchmetrics
import wandb
import pydensecrf

print("UMSS environment import check passed")
print(f"numpy={numpy.__version__}")
print(f"scipy={scipy.__version__}")
print(f"torch={torch.__version__}, cuda={torch.version.cuda}, cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}, capability={torch.cuda.get_device_capability(0)}")
PY

echo "Done. Activate with: conda activate ${ENV_NAME}"
