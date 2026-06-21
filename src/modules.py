import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from dinov3.hub.backbones import dinov3_vitb16, dinov3_vitl16, dinov3_vits16
    try:
        from dinov3.hub.backbones import dinov3_vits16plus
        DINOV3_VITSPLUS_AVAILABLE = True
    except ImportError:
        DINOV3_VITSPLUS_AVAILABLE = False
    DINOV3_AVAILABLE = True
except ImportError as exc:
    DINOV3_AVAILABLE = False
    DINOV3_VITSPLUS_AVAILABLE = False
    DINOV3_IMPORT_ERROR = exc


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class ModalityNetwork(nn.Module):
    def __init__(self, n_feats, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        self.conv1 = nn.Conv2d(n_feats, n_feats // 2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_feats // 2, n_feats, kernel_size=1)

    def forward(self, x):
        out = self.conv2(self.relu(self.conv1(x)))
        return out + x if self.use_residual else out


class Dinov3Featurizer(nn.Module):
    def __init__(self, dim, cfg):
        super().__init__()
        if not DINOV3_AVAILABLE:
            raise ImportError(f"DINOv3 is not available: {DINOV3_IMPORT_ERROR}")

        self.cfg = cfg
        self.dim = dim
        self.patch_size = 16
        self.feat_type = cfg.get("dino_feat_type", "ori")
        self.modals = list(cfg.get("modals", ["rgb"]))
        self.n_modals = len(self.modals)

        arch_map = {
            "vit_small": dinov3_vits16,
            "vit_base": dinov3_vitb16,
            "vit_large": dinov3_vitl16,
        }
        if DINOV3_VITSPLUS_AVAILABLE:
            arch_map["vit_small_plus"] = dinov3_vits16plus

        arch = cfg.model_type
        if arch not in arch_map:
            raise ValueError(f"Unknown DINOv3 model_type '{arch}'. Available: {sorted(arch_map)}")

        self.model = arch_map[arch](pretrained=False)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval().cuda()

        self.dropout = nn.Dropout2d(p=0.1)
        self._load_pretrained_weights(cfg.get("pretrained_weights", None))

        base_dim = 384 if arch in ("vit_small", "vit_small_plus") else 768 if arch == "vit_base" else 1024
        self.n_feats = base_dim * 3 if self.feat_type == "fea" else base_dim

        if self.n_modals > 1:
            use_residual = cfg.get("use_modality_residual", True)
            self.modality_networks = nn.ModuleDict({
                modal: ModalityNetwork(self.n_feats, use_residual=use_residual)
                for modal in self.modals
            })
            self.fusion_method = cfg.get("modality_fusion_method", "sum")
            self.fusion_output_dim = cfg.get("modality_fusion_output_dim", self.n_feats)
            self.fusion_conv = self._build_fusion_conv(cfg) if self.fusion_method == "conv" else None
            self.normalize_before_fusion = cfg.get("normalize_before_fusion", False)
            self.normalization_type = cfg.get("normalization_type", "standardize")
        else:
            self.modality_networks = None
            self.fusion_method = "sum"
            self.fusion_output_dim = self.n_feats
            self.fusion_conv = None
            self.normalize_before_fusion = False
            self.normalization_type = None

        self.proj_type = cfg.projection_type
        self.cluster1 = self.make_clusterer(self.fusion_output_dim)
        if self.proj_type == "nonlinear":
            self.cluster2 = self.make_nonlinear_clusterer(self.fusion_output_dim)

        if self.n_modals > 1 and cfg.get("use_modal_projection_head", True):
            self.modal_projection_head = nn.Sequential(
                nn.Conv2d(self.dim, self.dim, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.dim, self.dim, kernel_size=1),
            )
            print("  Created modal_projection_head for non-RGB soft alignment")
        else:
            self.modal_projection_head = None

    def _load_pretrained_weights(self, weight_path):
        if weight_path is None:
            print("Warning: no DINOv3 pretrained weights provided")
            return

        state_dict = torch.load(weight_path, map_location="cpu")
        if "teacher" in state_dict:
            state_dict = state_dict["teacher"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = self.model.load_state_dict(state_dict, strict=False)
        print(f"Pretrained weights found at {weight_path} and loaded with msg: {msg}")

    def _build_fusion_conv(self, cfg):
        if cfg.get("dataset_name", "") == "mcubes":
            layers = []
            for i in range(self.n_modals):
                in_ch = self.n_feats * (self.n_modals - i)
                out_ch = self.n_feats * (self.n_modals - i - 1) if i < self.n_modals - 1 else self.fusion_output_dim
                layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=1))
                if i < self.n_modals - 1:
                    layers.append(nn.BatchNorm2d(out_ch))
                    layers.append(nn.ReLU(inplace=True))
            print("  Conv fusion (MCubeS progressive)")
            return nn.Sequential(*layers)

        input_dim = self.n_feats * self.n_modals
        hidden_dim = (input_dim + self.fusion_output_dim) // 2
        print(f"  Conv fusion: {input_dim} -> {hidden_dim} -> {self.fusion_output_dim}")
        return nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, self.fusion_output_dim, kernel_size=1),
        )

    def make_clusterer(self, in_channels):
        return nn.Sequential(nn.Conv2d(in_channels, self.dim, kernel_size=1))

    def make_nonlinear_clusterer(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, self.dim, kernel_size=1),
        )

    def _normalize_features(self, x):
        if not self.normalize_before_fusion:
            return x
        if self.normalization_type == "l2":
            return F.normalize(x, p=2, dim=1, eps=1e-8)
        if self.normalization_type == "standardize":
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            std = x.std(dim=(1, 2, 3), keepdim=True)
            return (x - mean) / (std + 1e-8)
        if self.normalization_type == "minmax":
            flat = x.reshape(x.size(0), -1)
            x_min = flat.min(dim=1, keepdim=True)[0].reshape(x.size(0), 1, 1, 1)
            x_max = flat.max(dim=1, keepdim=True)[0].reshape(x.size(0), 1, 1, 1)
            return (x - x_min) / (x_max - x_min + 1e-8)
        raise ValueError(f"Unknown normalization_type '{self.normalization_type}'")

    def forward(self, img, n=1, return_class_feat=False):
        self.model.eval()

        if isinstance(img, dict):
            raw_feats = [self._extract_dinov3_features(img[modal], return_class_feat) for modal in self.modals]
            feats = [
                self.modality_networks[modal](feat)
                for modal, feat in zip(self.modals, raw_feats)
            ] if (not return_class_feat and self.modality_networks is not None) else list(raw_feats)
        else:
            raw_feats = [self._extract_dinov3_features(img, return_class_feat)]
            feats = list(raw_feats)

        if return_class_feat:
            return feats[0] if len(feats) == 1 else torch.stack(feats, dim=0).mean(dim=0)

        image_feat = self._fuse(feats, raw_feats)
        if self.proj_type is not None:
            code = self.cluster1(self.dropout(image_feat))
            if self.proj_type == "nonlinear":
                code = code + self.cluster2(self.dropout(image_feat))
        else:
            code = image_feat

        code_for_modal = self.modal_projection_head(code) if self.modal_projection_head is not None else None
        modal_feats_raw = raw_feats if len(raw_feats) > 1 else None
        modal_feats_transformed = feats if len(feats) > 1 and self.modality_networks is not None else None
        return (
            self.dropout(image_feat) if self.cfg.dropout else image_feat,
            code,
            modal_feats_raw,
            modal_feats_transformed,
            code_for_modal,
        )

    def _fuse(self, feats, raw_feats):
        if len(raw_feats) == 1:
            return raw_feats[0]

        feats = [self._normalize_features(feat) for feat in feats] if self.normalize_before_fusion else feats
        if self.fusion_method == "sum":
            return sum(feats)
        if self.fusion_method == "mean":
            return torch.stack(feats, dim=0).mean(dim=0)
        if self.fusion_method == "max":
            return torch.stack(feats, dim=0).max(dim=0)[0]
        if self.fusion_method == "conv":
            return self.fusion_conv(torch.cat(feats, dim=1))
        raise ValueError(f"Unknown modality_fusion_method '{self.fusion_method}'")

    def _extract_dinov3_features(self, img, return_class_feat=False):
        with torch.no_grad():
            assert img.shape[2] % self.patch_size == 0
            assert img.shape[3] % self.patch_size == 0
            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size

            if self.feat_type == "fea":
                outputs = self.model.get_intermediate_layers(
                    img, n=3, reshape=True, return_class_token=return_class_feat, norm=True
                )
                if return_class_feat:
                    _, cls_tokens = zip(*outputs)
                    return cls_tokens[-1].unsqueeze(-1).unsqueeze(-1)
                return torch.cat([outputs[-3], outputs[-2], outputs[-1]], dim=1)

            output = self.model.forward_features(img)
            if return_class_feat:
                return output["x_norm_clstoken"].unsqueeze(-1).unsqueeze(-1)
            patch_tokens = output["x_norm_patchtokens"]
            batch, tokens, dim = patch_tokens.shape
            assert tokens == feat_h * feat_w, "DINOv3 token count mismatch"
            return patch_tokens.permute(0, 2, 1).reshape(batch, dim, feat_h, feat_w)


class ClusterLookup(nn.Module):
    def __init__(self, dim, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.dim = dim
        self.clusters = nn.Parameter(torch.randn(n_classes, dim))

    def reset_parameters(self):
        with torch.no_grad():
            self.clusters.copy_(torch.randn(self.n_classes, self.dim, device=self.clusters.device))

    def forward(self, x, alpha, log_probs=False):
        normed_clusters = F.normalize(self.clusters, dim=1)
        normed_features = F.normalize(x, dim=1)
        inner_products = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)
        if alpha is None:
            cluster_probs = F.one_hot(torch.argmax(inner_products, dim=1), self.clusters.shape[0])
            cluster_probs = cluster_probs.permute(0, 3, 1, 2).to(torch.float32)
        else:
            cluster_probs = F.softmax(inner_products * alpha, dim=1)
        cluster_loss = -(cluster_probs * inner_products).sum(1).mean()
        return F.log_softmax(inner_products * alpha, dim=1) if log_probs else (cluster_loss, cluster_probs)


def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)


def tensor_correlation(a, b):
    return torch.einsum("nchw,ncij->nhwij", a, b)


def sample(t, coords):
    return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode="border", align_corners=True)


@torch.jit.script
def super_perm(size: int, device: torch.device):
    perm = torch.randperm(size, device=device, dtype=torch.long)
    perm[perm == torch.arange(size, device=device)] += 1
    return perm % size


def sample_nonzero_locations(t, target_size):
    nonzeros = torch.nonzero(t)
    coords = torch.zeros(target_size, dtype=nonzeros.dtype, device=nonzeros.device)
    n = target_size[1] * target_size[2]
    for i in range(t.shape[0]):
        selected = nonzeros[nonzeros[:, 0] == i]
        if selected.shape[0] == 0:
            selected_coords = torch.randint(t.shape[1], size=(n, 2), device=nonzeros.device)
        else:
            selected_coords = selected[torch.randint(len(selected), size=(n,), device=nonzeros.device), 1:]
        coords[i] = selected_coords.reshape(target_size[1], target_size[2], 2)
    coords = coords.to(torch.float32) / t.shape[1]
    coords = coords * 2 - 1
    return torch.flip(coords, dims=[-1])


class ContrastiveCorrelationLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def helper(self, f1, f2, c1, c2, shift, weight_map1=None, weight_map2=None):
        with torch.no_grad():
            fd = tensor_correlation(norm(f1), norm(f2))
            if self.cfg.pointwise:
                old_mean = fd.mean()
                fd -= fd.mean([3, 4], keepdim=True)
                fd = fd - fd.mean() + old_mean

        cd = tensor_correlation(norm(c1), norm(c2))
        min_val = 0.0 if self.cfg.zero_clamp else -9999.0
        if self.cfg.stabalize:
            loss = -cd.clamp(min_val, 0.8) * (fd - shift)
        else:
            loss = -cd.clamp(min_val) * (fd - shift)

        if weight_map1 is not None and weight_map2 is not None:
            weight_i = weight_map1.unsqueeze(-1).unsqueeze(-1)
            weight_j = weight_map2.unsqueeze(1).unsqueeze(1)
            loss = loss * weight_i * weight_j
        return loss, cd

    def forward(self, orig_feats, orig_feats_pos, orig_salience, orig_salience_pos,
                orig_code, orig_code_pos, is_rgb=True):
        use_modality_weight = getattr(self.cfg, "use_modality_weight", False)
        rgb_use_weight = getattr(self.cfg, "rgb_use_weight", False)
        use_weight = (rgb_use_weight if is_rgb else True) if use_modality_weight else False

        weight_map_full = None
        weight_map_full_pos = None
        if use_weight:
            feat_importance = orig_feats.abs().mean(dim=1)
            feat_importance_pos = orig_feats_pos.abs().mean(dim=1)
            activation = getattr(self.cfg, "weight_activation", "sigmoid")
            if activation == "sigmoid":
                weight_map_full = torch.sigmoid(feat_importance)
                weight_map_full_pos = torch.sigmoid(feat_importance_pos)
            elif activation == "tanh":
                weight_map_full = torch.tanh(feat_importance)
                weight_map_full_pos = torch.tanh(feat_importance_pos)
            else:
                raise ValueError(f"Unknown weight_activation '{activation}'")

        coord_shape = [orig_feats.shape[0], self.cfg.feature_samples, self.cfg.feature_samples, 2]
        if self.cfg.use_salience:
            coords1_nonzero = sample_nonzero_locations(orig_salience, coord_shape)
            coords2_nonzero = sample_nonzero_locations(orig_salience_pos, coord_shape)
            coords1_reg = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
            coords2_reg = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
            mask = (torch.rand(coord_shape[:-1], device=orig_feats.device) > 0.1).unsqueeze(-1).to(torch.float32)
            coords1 = coords1_nonzero * mask + coords1_reg * (1 - mask)
            coords2 = coords2_nonzero * mask + coords2_reg * (1 - mask)
        else:
            coords1 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
            coords2 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1

        feats = sample(orig_feats, coords1)
        code = sample(orig_code, coords1)
        feats_pos = sample(orig_feats_pos, coords2)
        code_pos = sample(orig_code_pos, coords2)

        weight_map1 = sample(weight_map_full.unsqueeze(1), coords1).squeeze(1) if use_weight else None
        weight_map_pos = sample(weight_map_full_pos.unsqueeze(1), coords2).squeeze(1) if use_weight else None

        pos_intra_loss, pos_intra_cd = self.helper(
            feats, feats, code, code, self.cfg.pos_intra_shift, weight_map1, weight_map1
        )
        pos_inter_loss, pos_inter_cd = self.helper(
            feats, feats_pos, code, code_pos, self.cfg.pos_inter_shift, weight_map1, weight_map_pos
        )

        neg_losses = []
        neg_cds = []
        for _ in range(self.cfg.neg_samples):
            perm_neg = super_perm(orig_feats.shape[0], orig_feats.device)
            feats_neg = sample(orig_feats[perm_neg], coords2)
            code_neg = sample(orig_code[perm_neg], coords2)
            weight_map_neg = None
            if use_weight:
                weight_map_neg = sample(weight_map_full[perm_neg].unsqueeze(1), coords2).squeeze(1)
            neg_loss, neg_cd = self.helper(
                feats, feats_neg, code, code_neg, self.cfg.neg_inter_shift, weight_map1, weight_map_neg
            )
            neg_losses.append(neg_loss)
            neg_cds.append(neg_cd)

        return (
            pos_intra_loss.mean(),
            pos_intra_cd,
            pos_inter_loss.mean(),
            pos_inter_cd,
            torch.cat(neg_losses, axis=0),
            torch.cat(neg_cds, axis=0),
        )
