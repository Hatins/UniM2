import os
from os.path import join

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import ContrastiveSegDataset
from modules import Dinov3Featurizer, LambdaLayer
from utils import get_transform, prep_args, resolve_path


def get_feats(model, loader):
    all_feats = []
    for batch in tqdm(loader, desc="Extracting features"):
        img = batch["rgb"].cuda(non_blocking=True)
        feats = F.normalize(model(img).mean([2, 3]), dim=1)
        all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0).contiguous()


def compute_knns(normed_feats, num_neighbors=30, n_batches=128, chunk_size=10000):
    all_nns = []
    step = max(1, normed_feats.shape[0] // n_batches)
    for start in tqdm(range(0, normed_feats.shape[0], step), desc="Computing KNNs"):
        batch_feats = normed_feats[start:start + step].cuda()
        similarities = []
        for chunk_start in range(0, normed_feats.shape[0], chunk_size):
            chunk_feats = normed_feats[chunk_start:chunk_start + chunk_size].cuda()
            similarities.append(torch.einsum("nf,mf->nm", batch_feats, chunk_feats).cpu())
            del chunk_feats
            torch.cuda.empty_cache()
        pairwise_sims = torch.cat(similarities, dim=1)
        all_nns.append(torch.topk(pairwise_sims, num_neighbors)[1])
        del pairwise_sims, batch_feats, similarities
        torch.cuda.empty_cache()
    return torch.cat(all_nns, dim=0)


@hydra.main(version_base="1.1", config_path="configs", config_name="train_config_nyu.yml")
def my_app(cfg: DictConfig) -> None:
    cfg.pytorch_data_dir = resolve_path(cfg.pytorch_data_dir)
    if cfg.get("pretrained_weights", None):
        cfg.pretrained_weights = resolve_path(cfg.pretrained_weights)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.get("gpu_ids", "0")
    print(OmegaConf.to_yaml(cfg))
    os.makedirs(join(cfg.pytorch_data_dir, "nns"), exist_ok=True)
    seed_everything(seed=0)

    if cfg.arch != "dino" or cfg.get("dino_version", "v3") != "v3":
        raise ValueError("UniM2/UMSS KNN precompute only supports DINOv3.")

    feature_model = torch.nn.Sequential(
        Dinov3Featurizer(20, cfg),
        LambdaLayer(lambda outputs: outputs[0]),
    ).cuda()
    if torch.cuda.device_count() > 1:
        feature_model = torch.nn.DataParallel(feature_model)

    dino_version = cfg.get("dino_version", "v3")
    patch_size = cfg.get("dino_patch_size", 16)
    model_info = f"{cfg.model_type}_dino{dino_version}_p{patch_size}"

    for image_set in ("val", "train"):
        cache_path = join(
            cfg.pytorch_data_dir,
            "nns",
            f"nns_{model_info}_{cfg.dataset_name}_{image_set}_{cfg.crop_type}_{cfg.res}.npz",
        )
        if os.path.exists(cache_path):
            print(f"{cache_path} already exists, skipping.")
            continue

        dataset = ContrastiveSegDataset(
            pytorch_data_dir=cfg.pytorch_data_dir,
            dataset_name=cfg.dataset_name,
            crop_type=cfg.crop_type,
            image_set=image_set,
            transform=get_transform(cfg.res, False, "center"),
            target_transform=get_transform(cfg.res, True, "center"),
            cfg=cfg,
        )
        loader = DataLoader(dataset, 128, shuffle=False, num_workers=cfg.num_workers, pin_memory=False)

        with torch.no_grad():
            normed_feats = get_feats(feature_model, loader)
            nearest_neighbors = compute_knns(normed_feats)

        np.savez_compressed(
            cache_path,
            nns=nearest_neighbors.numpy(),
            dino_version=dino_version,
            model_type=cfg.model_type,
            patch_size=patch_size,
            resolution=cfg.res,
            dataset=cfg.dataset_name,
            image_set=image_set,
            crop_type=str(cfg.crop_type),
        )
        print(f"Saved {cache_path}")


if __name__ == "__main__":
    prep_args()
    my_app()
