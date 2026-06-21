import os
import sys
import warnings
from datetime import datetime
from os.path import join

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import transforms as T

from data import (
    ContrastiveSegDataset,
    MCubeSDataset,
    MFNetDataset,
    NYUDataset,
    create_mcubes_colormap,
    create_mfnet_colormap,
    create_nyu_4class_colormap,
    create_nyu_colormap,
)
from modules import ClusterLookup, ContrastiveCorrelationLoss, Dinov3Featurizer, norm
from utils import UnsupervisedMetrics, get_transform, one_hot_feats, prep_args, resolve_path


warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.meshgrid.*")
torch.multiprocessing.set_sharing_strategy("file_system")


def get_class_labels(dataset_name, num_classes=None):
    if dataset_name == "nyu":
        return NYUDataset.CLASSES_4 if num_classes == 4 else NYUDataset.CLASSES_13
    if dataset_name == "mfnet":
        return MFNetDataset.CLASSES
    if dataset_name == "mcubes":
        return MCubeSDataset.CLASSES
    raise ValueError(f"Unsupported dataset: {dataset_name}")


class LitUnsupervisedSegmenter(pl.LightningModule):
    PARAM_KEYS = (
        "pos_inter_weight",
        "pos_intra_weight",
        "neg_inter_weight",
        "neg_inter_shift",
        "pos_inter_shift",
        "pos_intra_shift",
        "weight",
    )

    def __init__(self, n_classes, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_classes = n_classes
        dim = n_classes if not cfg.continuous else cfg.dim

        if cfg.arch != "dino" or cfg.get("dino_version", "v3") != "v3":
            raise ValueError("UniM2/UMSS open-source path only supports DINOv3 (arch=dino, dino_version=v3).")

        self.net = Dinov3Featurizer(dim, cfg)
        self.train_cluster_probe = ClusterLookup(dim, n_classes)
        self.cluster_probe = ClusterLookup(dim, n_classes + cfg.extra_clusters)
        self.linear_probe = nn.Conv2d(dim, n_classes, kernel_size=1)
        self.decoder = nn.Conv2d(dim, self.net.n_feats, kernel_size=1)

        self.cluster_metrics = UnsupervisedMetrics("test/cluster/", n_classes, cfg.extra_clusters, True)
        self.linear_metrics = UnsupervisedMetrics("test/linear/", n_classes, 0, False)
        self.test_cluster_metrics = UnsupervisedMetrics("final/cluster/", n_classes, cfg.extra_clusters, True)
        self.test_linear_metrics = UnsupervisedMetrics("final/linear/", n_classes, 0, False)
        self.linear_probe_loss_fn = nn.CrossEntropyLoss(ignore_index=255)

        self.contrastive_corr_loss_fns = {}
        for modal in cfg.modals:
            modal_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            for key in self.PARAM_KEYS[:-1]:
                setattr(modal_cfg, key, cfg.get(f"{modal}_{key}", cfg.get(key)))
            self.contrastive_corr_loss_fns[modal] = ContrastiveCorrelationLoss(modal_cfg)
            for param in self.contrastive_corr_loss_fns[modal].parameters():
                param.requires_grad = False

        self.label_cmap = self._make_colormap()
        self.automatic_optimization = False
        self.save_hyperparameters()

    def _make_colormap(self):
        if self.cfg.dataset_name == "nyu":
            return create_nyu_4class_colormap() if self.cfg.get("nyu_num_classes", 13) == 4 else create_nyu_colormap()
        if self.cfg.dataset_name == "mfnet":
            return create_mfnet_colormap()
        if self.cfg.dataset_name == "mcubes":
            return create_mcubes_colormap()
        raise ValueError(f"Unsupported dataset: {self.cfg.dataset_name}")

    def forward(self, x):
        return self.net(x)[1]

    def _batch_images(self, batch, positive=False):
        suffix = "_pos" if positive else ""
        if len(self.cfg.modals) == 1:
            return batch[f"{self.cfg.modals[0]}{suffix}"]
        return {modal: batch[f"{modal}{suffix}"] for modal in self.cfg.modals}

    def _modal_code(self, modal_name, code, code_pos, code_for_modal, code_for_modal_pos):
        rgb_soft = self.cfg.get("rgb_use_soft_alignment", False)
        nonrgb_soft = self.cfg.get("nonrgb_use_soft_alignment", True)
        use_soft = rgb_soft if modal_name == "rgb" else nonrgb_soft
        if use_soft and code_for_modal is not None:
            return code_for_modal, code_for_modal_pos if code_for_modal_pos is not None else code_pos
        return code, code_pos

    def training_step(self, batch, batch_idx):
        self.train()
        net_optim, linear_probe_optim, cluster_probe_optim = self.optimizers()
        net_optim.zero_grad()
        linear_probe_optim.zero_grad()
        cluster_probe_optim.zero_grad()

        img = self._batch_images(batch)
        img_pos = self._batch_images(batch, positive=True)
        label = batch["label"]
        label_pos = batch["label_pos"]
        salience = batch["mask"].to(torch.float32).squeeze(1) if self.cfg.use_salience else None
        salience_pos = batch["mask_pos"].to(torch.float32).squeeze(1) if self.cfg.use_salience else None

        feats, code, modal_feats_raw, _, code_for_modal = self.net(img)
        feats_pos, code_pos, modal_feats_pos, _, code_for_modal_pos = self.net(img_pos)

        if self.cfg.use_true_labels:
            signal = one_hot_feats(label + 1, self.n_classes + 1)
            signal_pos = one_hot_feats(label_pos + 1, self.n_classes + 1)
        else:
            signal = feats
            signal_pos = feats_pos

        loss = 0.0
        log_args = dict(sync_dist=False, rank_zero_only=True)
        if self.cfg.correspondence_weight > 0:
            corr_loss = self._contrastive_loss(
                signal, signal_pos, salience, salience_pos,
                code, code_pos, modal_feats_raw, modal_feats_pos,
                code_for_modal, code_for_modal_pos, log_args,
            )
            loss = loss + self.cfg.correspondence_weight * corr_loss

        if self.cfg.rec_weight > 0:
            rec_feats = self.decoder(code)
            rec_loss = -(norm(rec_feats) * norm(feats)).sum(1).mean()
            self.log("loss/rec", rec_loss, **log_args)
            loss = loss + self.cfg.rec_weight * rec_loss

        detached_code = code.detach().clone()
        linear_logits = self.linear_probe(detached_code)
        linear_logits = F.interpolate(linear_logits, label.shape[-2:], mode="bilinear", align_corners=False)
        linear_logits = linear_logits.permute(0, 2, 3, 1).reshape(-1, self.n_classes)
        linear_loss = self.linear_probe_loss_fn(linear_logits, label.reshape(-1))
        loss = loss + linear_loss
        self.log("loss/linear", linear_loss, **log_args)

        cluster_loss, _ = self.cluster_probe(detached_code, None)
        loss = loss + cluster_loss
        self.log("loss/cluster", cluster_loss, **log_args)
        self.log("loss/total", loss, **log_args)

        self.manual_backward(loss)
        net_optim.step()
        cluster_probe_optim.step()
        linear_probe_optim.step()

        if self.cfg.reset_probe_steps is not None and self.global_step == self.cfg.reset_probe_steps:
            self.linear_probe.reset_parameters()
            self.cluster_probe.reset_parameters()
            self.trainer.optimizers[1] = torch.optim.Adam(self.linear_probe.parameters(), lr=5e-3)
            self.trainer.optimizers[2] = torch.optim.Adam(self.cluster_probe.parameters(), lr=5e-3)

        if "OPTUNA_EXPERIMENT_NAME" in os.environ and self.global_step % 30 == 0:
            print(f"Training: global_step={self.global_step}", flush=True)
        return loss

    def _contrastive_loss(self, signal, signal_pos, salience, salience_pos, code, code_pos,
                          modal_feats_raw, modal_feats_pos, code_for_modal, code_for_modal_pos, log_args):
        pos_intra_total = 0.0
        pos_inter_total = 0.0
        neg_inter_total = 0.0

        if modal_feats_raw is None:
            modal_name = self.cfg.modals[0]
            loss_fn = self.contrastive_corr_loss_fns[modal_name]
            modal_inputs = [(modal_name, signal, signal_pos, code, code_pos)]
        else:
            modal_inputs = []
            for modal_name, modal_feat, modal_feat_pos in zip(self.cfg.modals, modal_feats_raw, modal_feats_pos):
                code_to_use, code_pos_to_use = self._modal_code(
                    modal_name, code, code_pos, code_for_modal, code_for_modal_pos
                )
                modal_inputs.append((modal_name, modal_feat, modal_feat_pos, code_to_use, code_pos_to_use))

        for modal_name, feat, feat_pos, code_to_use, code_pos_to_use in modal_inputs:
            loss_fn = self.contrastive_corr_loss_fns[modal_name]
            pos_intra, _, pos_inter, _, neg_inter, _ = loss_fn(
                feat, feat_pos, salience, salience_pos, code_to_use, code_pos_to_use,
                is_rgb=(modal_name == "rgb"),
            )
            pos_intra = pos_intra.mean()
            pos_inter = pos_inter.mean()
            neg_inter = neg_inter.mean()

            pos_inter_w = self.cfg.get(f"{modal_name}_pos_inter_weight", 0.6313)
            pos_intra_w = self.cfg.get(f"{modal_name}_pos_intra_weight", 0.7939)
            neg_inter_w = self.cfg.get(f"{modal_name}_neg_inter_weight", 0.8754)
            modal_w = self.cfg.get(f"{modal_name}_weight", 1.0 / len(self.cfg.modals))

            self.log(f"loss/pos_intra_{modal_name}", pos_intra, **log_args)
            self.log(f"loss/pos_inter_{modal_name}", pos_inter, **log_args)
            self.log(f"loss/neg_inter_{modal_name}", neg_inter, **log_args)
            self.log(f"weight/{modal_name}_weight", modal_w, **log_args)

            pos_intra_total = pos_intra_total + modal_w * pos_intra_w * pos_intra
            pos_inter_total = pos_inter_total + modal_w * pos_inter_w * pos_inter
            neg_inter_total = neg_inter_total + modal_w * neg_inter_w * neg_inter

        self.log("loss/pos_intra_total", pos_intra_total, **log_args)
        self.log("loss/pos_inter_total", pos_inter_total, **log_args)
        self.log("loss/neg_inter_total", neg_inter_total, **log_args)
        return pos_intra_total + pos_inter_total + neg_inter_total

    def on_train_start(self):
        self.logger.log_hyperparams(dict(self.cfg))

    def validation_step(self, batch, batch_idx):
        self.eval()
        img = self._batch_images(batch)
        label = batch["label"]
        with torch.no_grad():
            _, code, _, _, _ = self.net(img)
            code = F.interpolate(code, label.shape[-2:], mode="bilinear", align_corners=False)

            linear_preds = self.linear_probe(code).argmax(1)
            self.linear_metrics.update(linear_preds, label)

            _, cluster_preds = self.cluster_probe(code, None)
            self.cluster_metrics.update(cluster_preds.argmax(1), label)

    def on_validation_epoch_end(self):
        with torch.no_grad():
            metrics = {**self.linear_metrics.compute(), **self.cluster_metrics.compute()}
            if self.global_step > 2:
                self.log_dict(metrics)
                if "OPTUNA_EXPERIMENT_NAME" in os.environ and self.trainer.is_global_zero:
                    miou = metrics.get("test/cluster/mIoU", 0.0)
                    acc = metrics.get("test/cluster/Accuracy", 0.0)
                    print(f"OPTUNA_METRIC: step={self.global_step}, mIoU={miou:.4f}, Accuracy={acc:.4f}", flush=True)
            self.linear_metrics.reset()
            self.cluster_metrics.reset()
            self.train()

    def configure_optimizers(self):
        main_params = list(self.net.parameters())
        if self.cfg.rec_weight > 0:
            main_params.extend(self.decoder.parameters())
        return (
            torch.optim.Adam(main_params, lr=self.cfg.lr),
            torch.optim.Adam(self.linear_probe.parameters(), lr=5e-3),
            torch.optim.Adam(self.cluster_probe.parameters(), lr=5e-3),
        )


def apply_optuna_overrides(cfg):
    OmegaConf.set_struct(cfg, False)
    for modal in cfg.modals:
        prefix = modal.upper()
        env_to_key = {
            f"OPTUNA_{prefix}_POS_INTER_WEIGHT": f"{modal}_pos_inter_weight",
            f"OPTUNA_{prefix}_POS_INTRA_WEIGHT": f"{modal}_pos_intra_weight",
            f"OPTUNA_{prefix}_NEG_INTER_WEIGHT": f"{modal}_neg_inter_weight",
            f"OPTUNA_{prefix}_NEG_INTER_SHIFT": f"{modal}_neg_inter_shift",
            f"OPTUNA_{prefix}_POS_INTER_SHIFT": f"{modal}_pos_inter_shift",
            f"OPTUNA_{prefix}_POS_INTRA_SHIFT": f"{modal}_pos_intra_shift",
            f"OPTUNA_{prefix}_WEIGHT": f"{modal}_weight",
        }
        for env_name, cfg_key in env_to_key.items():
            if env_name in os.environ:
                setattr(cfg, cfg_key, float(os.environ[env_name]))

    scalar_overrides = {
        "OPTUNA_MAX_STEPS": ("max_steps", int),
        "OPTUNA_EVAL_EVERY_N_EPOCHS": ("eval_every_n_epochs", int),
        "OPTUNA_EXPERIMENT_NAME": ("experiment_name", str),
        "OPTUNA_DIM": ("dim", int),
        "OPTUNA_LR": ("lr", float),
    }
    for env_name, (cfg_key, caster) in scalar_overrides.items():
        if env_name in os.environ:
            setattr(cfg, cfg_key, caster(os.environ[env_name]))
    return cfg


def resolve_runtime_paths(cfg):
    cfg.output_root = resolve_path(cfg.get("output_root", "."))
    cfg.pytorch_data_dir = resolve_path(cfg.pytorch_data_dir)
    if cfg.get("pretrained_weights", None):
        cfg.pretrained_weights = resolve_path(cfg.pretrained_weights)
    return cfg


def build_loaders(cfg, dataloader_seed):
    geometric_transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(size=cfg.res, scale=(0.8, 1.0)),
    ])
    photometric_transforms = T.Compose([
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        T.RandomGrayscale(0.2),
        T.RandomApply([T.GaussianBlur((5, 5))]),
    ])

    train_dataset = ContrastiveSegDataset(
        pytorch_data_dir=cfg.pytorch_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=cfg.crop_type,
        image_set="train",
        transform=get_transform(cfg.res, False, cfg.loader_crop_type),
        target_transform=get_transform(cfg.res, True, cfg.loader_crop_type),
        cfg=cfg,
        aug_geometric_transform=geometric_transforms,
        aug_photometric_transform=photometric_transforms,
        num_neighbors=cfg.num_neighbors,
        mask=True,
        pos_images=True,
        pos_labels=True,
    )

    val_dataset = ContrastiveSegDataset(
        pytorch_data_dir=cfg.pytorch_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=cfg.get("val_crop_type", None),
        image_set="val",
        transform=get_transform(cfg.get("val_res", 320), False, "center"),
        target_transform=get_transform(cfg.get("val_res", 320), True, "center"),
        cfg=cfg,
        mask=True,
    )

    generator = torch.Generator()
    generator.manual_seed(dataloader_seed)
    train_loader = DataLoader(
        train_dataset,
        cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        cfg.get("val_batch_size", cfg.batch_size),
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )
    return train_dataset, train_loader, val_loader


@hydra.main(version_base="1.1", config_path="configs", config_name="train_config_nyu.yml")
def my_app(cfg: DictConfig) -> None:
    cfg = apply_optuna_overrides(cfg)
    cfg = resolve_runtime_paths(cfg)
    gpu_ids = cfg.get("gpu_ids", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    logical_gpu_ids = list(range(len(gpu_ids.split(","))))

    print(OmegaConf.to_yaml(cfg))
    for path in (join(cfg.output_root, "data"), join(cfg.output_root, "logs"), join(cfg.output_root, "checkpoints")):
        os.makedirs(path, exist_ok=True)

    seed_everything(seed=100)
    dataloader_seed = int(os.environ.get("OPTUNA_DATALOADER_SEED", cfg.get("dataloader_seed", 18)))
    print(f"Using global seed: 100, dataloader seed: {dataloader_seed}")

    train_dataset, train_loader, val_loader = build_loaders(cfg, dataloader_seed)
    model = LitUnsupervisedSegmenter(train_dataset.n_classes, cfg)

    logger = WandbLogger(
        name=f"{cfg.log_dir}_{cfg.experiment_name}",
        project=cfg.get("project_name", "UniM2"),
        entity=cfg.get("entity", None),
        save_dir=cfg.output_root,
        group=cfg.get("group", None),
        offline=cfg.get("wandb_offline", False),
    )

    if "OPTUNA_EVAL_EVERY_N_EPOCHS" in os.environ:
        gpu_args = dict(devices=logical_gpu_ids, accelerator="gpu", check_val_every_n_epoch=cfg.eval_every_n_epochs)
    else:
        gpu_args = dict(devices=logical_gpu_ids, accelerator="gpu", val_check_interval=cfg.get("val_freq", 10))
    if len(logical_gpu_ids) > 1:
        gpu_args["strategy"] = "ddp_find_unused_parameters_true"

    run_name = f"{cfg.dataset_name}_{cfg.experiment_name}_date_{datetime.now().strftime('%b%d_%H-%M-%S')}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=join(cfg.output_root, "checkpoints", cfg.log_dir, run_name),
        filename="step={step}-mIoU={test/cluster/mIoU:.2f}",
        save_top_k=3,
        monitor="test/cluster/mIoU",
        mode="max",
        auto_insert_metric_name=False,
        save_on_train_epoch_end=False,
    )

    trainer = Trainer(
        log_every_n_steps=cfg.scalar_log_freq,
        logger=logger,
        max_steps=cfg.max_steps,
        max_epochs=-1,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        **gpu_args,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    prep_args()
    my_app()
