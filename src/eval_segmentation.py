import json
import multiprocessing as mp
import os
from os.path import join

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from crf import dense_crf
from data import ContrastiveSegDataset
from train_segmentation import LitUnsupervisedSegmenter
from utils import flexible_collate, get_transform, prep_for_plot, resolve_path


try:
    torch.multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass
torch.multiprocessing.set_sharing_strategy("file_system")


def _batch_images(batch, modals):
    if len(modals) == 1:
        return batch[modals[0]].cuda(non_blocking=True)
    return {modal: batch[modal].cuda(non_blocking=True) for modal in modals}


def _crf_images(batch, modals):
    guide_modal = "rgb" if "rgb" in batch else modals[0]
    return batch[guide_modal]


def _apply_crf(args):
    return dense_crf(args[0], args[1])


def _batched_crf(pool, image_tensor, prob_tensor):
    outputs = pool.map(_apply_crf, zip(image_tensor.detach().cpu(), prob_tensor.detach().cpu()))
    return torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in outputs], dim=0)


def resolve_runtime_paths(cfg):
    cfg.output_root = resolve_path(cfg.get("output_root", "."))
    cfg.pytorch_data_dir = resolve_path(cfg.pytorch_data_dir)
    if cfg.get("result_dir", None):
        cfg.result_dir = resolve_path(cfg.result_dir)
    cfg.model_paths = [resolve_path(path) for path in cfg.model_paths]
    return cfg


def load_segmenter_from_checkpoint(model_path):
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")

    hparams = checkpoint.get("hyper_parameters", {})
    checkpoint_cfg = hparams.get("cfg")
    n_classes = hparams.get("n_classes")
    if checkpoint_cfg is None or n_classes is None:
        return LitUnsupervisedSegmenter.load_from_checkpoint(model_path, map_location="cpu")

    if isinstance(checkpoint_cfg, DictConfig):
        checkpoint_cfg = OmegaConf.to_container(checkpoint_cfg, resolve=True)
    checkpoint_cfg = OmegaConf.create(checkpoint_cfg)
    checkpoint_cfg.pretrained_weights = None
    return LitUnsupervisedSegmenter.load_from_checkpoint(
        model_path,
        map_location="cpu",
        n_classes=n_classes,
        cfg=checkpoint_cfg,
    )


def _save_prediction_triplet(result_dir, idx, image, label, pred, label_cmap):
    os.makedirs(result_dir, exist_ok=True)
    label_vis = label.clone()
    label_vis[label_vis == 255] = 0
    pred_vis = pred.clone()
    pred_vis[pred_vis == 255] = 0

    img_np = (prep_for_plot(image.cpu()) * 255).numpy().astype(np.uint8)
    label_np = label_cmap[label_vis.cpu()].astype(np.uint8)
    pred_np = label_cmap[pred_vis.cpu()].astype(np.uint8)
    Image.fromarray(np.concatenate([img_np, label_np, pred_np], axis=1)).save(
        join(result_dir, f"{idx:04d}.png")
    )


@hydra.main(version_base="1.1", config_path="configs", config_name="eval_config.yml")
def my_app(cfg: DictConfig) -> None:
    cfg = resolve_runtime_paths(cfg)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.get("gpu_ids", "0")
    result_root = cfg.get("result_dir", join(cfg.output_root, "results", cfg.experiment_name))
    os.makedirs(result_root, exist_ok=True)

    all_results = {}
    for model_path in cfg.model_paths:
        print(f"Evaluating {model_path}")
        model = load_segmenter_from_checkpoint(model_path)
        model.eval().cuda()
        print(OmegaConf.to_yaml(model.cfg))

        eval_crop_type = cfg.get("eval_crop_type", model.cfg.get("val_crop_type", None))
        eval_res = cfg.get("res", model.cfg.get("val_res", 320))
        dataset = ContrastiveSegDataset(
            pytorch_data_dir=cfg.get("pytorch_data_dir", model.cfg.pytorch_data_dir),
            dataset_name=model.cfg.dataset_name,
            crop_type=eval_crop_type,
            image_set=cfg.get("image_set", "val"),
            transform=get_transform(eval_res, False, "center"),
            target_transform=get_transform(eval_res, True, "center"),
            cfg=model.cfg,
            mask=True,
        )
        loader = DataLoader(
            dataset,
            cfg.get("batch_size", 16),
            shuffle=False,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=True,
            collate_fn=flexible_collate,
        )

        use_crf = cfg.get("run_crf", cfg.get("use_crf", False))
        pool_workers = cfg.get("crf_num_workers", cfg.get("num_workers", 4))
        vis_limit = cfg.get("num_vis_images", 0)
        saved_vis = 0
        cached_vis = []
        pool_workers = max(1, int(pool_workers))
        pool = mp.get_context("spawn").Pool(pool_workers) if use_crf else None
        try:
            for batch_idx, batch in enumerate(tqdm(loader)):
                with torch.no_grad():
                    img = _batch_images(batch, model.cfg.modals)
                    label = batch["label"].cuda(non_blocking=True)
                    _, code, _, _, _ = model.net(img)
                    code = F.interpolate(code, label.shape[-2:], mode="bilinear", align_corners=False)

                    linear_logits = model.linear_probe(code)
                    _, cluster_probs = model.cluster_probe(code, None)
                    linear_preds = linear_logits.argmax(1)
                    cluster_preds = cluster_probs.argmax(1)

                    if use_crf:
                        guide_images = _crf_images(batch, model.cfg.modals)
                        linear_probs = torch.log_softmax(linear_logits, dim=1)
                        cluster_probs = model.cluster_probe(code, 2, log_probs=True)
                        linear_preds = _batched_crf(pool, guide_images, linear_probs).argmax(1).cuda()
                        cluster_preds = _batched_crf(pool, guide_images, cluster_probs).argmax(1).cuda()

                    model.test_linear_metrics.update(linear_preds, label)
                    model.test_cluster_metrics.update(cluster_preds, label)

                    if saved_vis < vis_limit:
                        img_for_save = batch["rgb"] if "rgb" in batch else batch[model.cfg.modals[0]]
                        for i in range(cluster_preds.shape[0]):
                            if saved_vis >= vis_limit:
                                break
                            cached_vis.append((
                                saved_vis,
                                img_for_save[i].cpu(),
                                label[i].cpu(),
                                cluster_preds[i].cpu(),
                            ))
                            saved_vis += 1
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        metrics = {
            **model.test_linear_metrics.compute(),
            **model.test_cluster_metrics.compute(),
        }
        model.test_linear_metrics.reset()
        model.test_cluster_metrics.reset()
        print(json.dumps(metrics, indent=2, sort_keys=True))

        run_name = os.path.splitext(os.path.basename(model_path))[0]
        vis_dir = join(result_root, run_name)
        for idx, image, label, pred in cached_vis:
            mapped = model.test_cluster_metrics.map_clusters(pred)
            _save_prediction_triplet(vis_dir, idx, image, label, mapped, model.label_cmap)

        all_results[model_path] = metrics

    with open(join(result_root, "metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    my_app()
