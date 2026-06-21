import os
import random
from os.path import join

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import io
from torchvision.transforms.functional import to_pil_image


IGNORE_INDEX = 255


def create_nyu_colormap():
    colors = np.ones((256, 3), dtype=np.uint8) * 128
    palette = [
        [107, 142, 35], [220, 220, 0], [70, 130, 180], [128, 64, 128],
        [152, 251, 152], [244, 35, 232], [190, 153, 153], [250, 170, 30],
        [255, 0, 0], [220, 20, 60], [0, 0, 142], [70, 70, 70],
        [153, 153, 153],
    ]
    for i, color in enumerate(palette):
        colors[i] = color
    colors[IGNORE_INDEX] = [0, 0, 0]
    return colors


def create_nyu_4class_colormap():
    colors = np.ones((256, 3), dtype=np.uint8) * 128
    colors[0] = [70, 70, 70]
    colors[1] = [220, 20, 60]
    colors[2] = [190, 153, 153]
    colors[3] = [128, 128, 128]
    colors[IGNORE_INDEX] = [0, 0, 0]
    return colors


def create_mfnet_colormap():
    colors = np.ones((256, 3), dtype=np.uint8) * 128
    palette = [
        [64, 0, 128], [64, 64, 0], [0, 128, 192], [0, 0, 192],
        [128, 128, 0], [64, 64, 128], [192, 128, 128], [192, 64, 0],
    ]
    for i, color in enumerate(palette):
        colors[i] = color
    colors[IGNORE_INDEX] = [0, 0, 0]
    return colors


def create_mcubes_colormap():
    colors = np.ones((256, 3), dtype=np.uint8) * 128
    palette = [
        [44, 160, 44], [31, 119, 180], [255, 127, 14], [214, 39, 40],
        [140, 86, 75], [127, 127, 127], [188, 189, 34], [255, 152, 150],
        [23, 190, 207], [174, 199, 232], [196, 156, 148], [197, 176, 213],
        [247, 182, 210], [199, 199, 199], [219, 219, 141], [158, 218, 229],
        [57, 59, 121], [107, 110, 207], [156, 158, 222], [99, 121, 57],
    ]
    for i, color in enumerate(palette):
        colors[i] = color
    colors[IGNORE_INDEX] = [0, 0, 0]
    return colors


def _sync_transforms(modals_data, label, transform, target_transform):
    seed = np.random.randint(2147483647)
    transformed = {}
    for modal, img in modals_data.items():
        random.seed(seed)
        torch.manual_seed(seed)
        transformed[modal] = transform(img)

    random.seed(seed)
    torch.manual_seed(seed)
    label = target_transform(label).squeeze(0)
    return transformed, label


def _read_image_as_rgb(path):
    img = io.read_image(path)
    if img.shape[0] >= 3:
        img = img[:3]
    elif img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    return to_pil_image(img)


class NYUDataset(Dataset):
    CLASSES_13 = [
        "bed", "books", "ceiling", "chair", "floor", "furniture", "objects",
        "picture", "sofa", "table", "tv", "wall", "window",
    ]
    CLASSES_4 = ["structure", "furniture", "objects", "void"]

    MAPPING_40_TO_13 = [
        11, 4, 5, 0, 3, 8, 9, 11, 12, 5, 7, 5, 12, 9, 5, 12, 5, 6, 6, 4,
        6, 2, 1, 5, 10, 6, 6, 6, 6, 6, 6, 5, 6, 6, 6, 6, 6, 6, 5, 6,
    ]
    MAPPING_40_TO_4 = [
        0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 2, 1, 0, 1, 1, 0, 1, 2, 2, 0,
        2, 0, 2, 1, 2, 2, 2, 0, 2, 2, 2, 1, 2, 2, 2, 2, 2, 0, 1, 2,
    ]

    def __init__(self, root, image_set, transform, target_transform, modals=None, num_classes=13):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.modals = modals or ["rgb"]
        self.n_classes = num_classes
        self.mapping = self.MAPPING_40_TO_4 if num_classes == 4 else self.MAPPING_40_TO_13
        if num_classes not in (4, 13):
            raise ValueError("NYU supports 4 or 13 classes.")
        self.files = self._read_split(image_set)
        print(f"Found {len(self.files)} {image_set} images for NYU ({num_classes} classes).")

    def _read_split(self, split_name):
        if split_name in ("val", "test"):
            candidates = [join(self.root, "val.txt"), join(self.root, "test.txt")]
            split_file = next((p for p in candidates if os.path.exists(p)), None)
            if split_file is None:
                raise FileNotFoundError(f"Missing val.txt/test.txt in {self.root}")
        else:
            split_file = join(self.root, f"{split_name}.txt")
        with open(split_file) as f:
            return [line.strip().split()[0] for line in f if line.strip()]

    def _load_modal(self, item_name, modal):
        item_name = str(item_name).zfill(4)
        if modal == "rgb":
            for ext in ("jpg", "png"):
                path = join(self.root, "RGB", f"{item_name}.{ext}")
                if os.path.exists(path):
                    return Image.open(path).convert("RGB").transpose(Image.TRANSPOSE)
            raise FileNotFoundError(f"RGB file not found for NYU sample {item_name}")
        if modal == "depth":
            return _read_image_as_rgb(join(self.root, "HHA", f"{item_name}.png"))
        raise NotImplementedError(f"NYU does not support modal '{modal}'")

    def __getitem__(self, index):
        item_name = str(self.files[index]).zfill(4)
        modals_data = {modal: self._load_modal(item_name, modal) for modal in self.modals}
        label_path = join(self.root, "Labels", f"{item_name}.png")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")

        label = to_pil_image(io.read_image(label_path)[0:1])
        transformed, label = _sync_transforms(modals_data, label, self.transform, self.target_transform)

        background = label == 0
        label = label - 1
        label[background] = IGNORE_INDEX

        mapped = torch.full_like(label, IGNORE_INDEX)
        valid = label != IGNORE_INDEX
        mapping = torch.tensor(self.mapping, dtype=label.dtype)
        mapped[valid] = mapping[label[valid].long()]
        mask = (mapped != IGNORE_INDEX).to(torch.float32)
        return transformed, mapped, mask

    def __len__(self):
        return len(self.files)


class MFNetDataset(Dataset):
    CLASSES = ["car", "person", "bike", "curve", "car_stop", "guardrail", "color_cone", "bump"]

    def __init__(self, root, image_set, transform, target_transform, modals=None):
        self.root = root
        self.split = "test" if image_set == "val" else image_set
        self.transform = transform
        self.target_transform = target_transform
        self.modals = modals or ["rgb"]
        self.n_classes = len(self.CLASSES)
        self.files = self._read_split(self.split)
        print(f"Found {len(self.files)} {self.split} images for MFNet.")

    def _read_split(self, split_name):
        split_file = join(self.root, f"{split_name}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        with open(split_file) as f:
            return [line.strip().split()[0] for line in f if line.strip()]

    def _load_modal(self, item_name, modal):
        is_flip = item_name.endswith("_flip")
        base_name = item_name[:-5] if is_flip else item_name
        path = join(self.root, "images", f"{base_name}.png")
        img = io.read_image(path)
        if img.shape[0] != 4:
            raise ValueError(f"Expected RGBA MFNet image, got {img.shape[0]} channels: {path}")

        if modal == "rgb":
            out = to_pil_image(img[:3])
        elif modal in ("thermal", "depth"):
            out = to_pil_image(img[3:4].repeat(3, 1, 1))
        else:
            raise NotImplementedError(f"MFNet does not support modal '{modal}'")
        return out.transpose(Image.FLIP_LEFT_RIGHT) if is_flip else out

    def __getitem__(self, index):
        item_name = self.files[index]
        is_flip = item_name.endswith("_flip")
        base_name = item_name[:-5] if is_flip else item_name
        modals_data = {modal: self._load_modal(item_name, modal) for modal in self.modals}

        label_path = join(self.root, "labels", f"{base_name}.png")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
        label = to_pil_image(io.read_image(label_path)[0:1])
        if is_flip:
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        label_arr = np.array(label)
        unlabeled = label_arr == 0
        label_arr = label_arr - 1
        label_arr[unlabeled] = IGNORE_INDEX
        label = Image.fromarray(label_arr.astype(np.uint8))

        transformed, label = _sync_transforms(modals_data, label, self.transform, self.target_transform)
        mask = (label != IGNORE_INDEX).to(torch.float32)
        return transformed, label, mask

    def __len__(self):
        return len(self.files)


class MCubeSDataset(Dataset):
    CLASSES = [
        "asphalt", "concrete", "metal", "road_marking", "fabric", "glass", "plaster",
        "plastic", "rubber", "sand", "gravel", "ceramic", "cobblestone", "brick",
        "grass", "wood", "leaf", "water", "human", "sky",
    ]
    LEFT_OFFSET = 192

    def __init__(self, root, image_set, transform, target_transform, modals=None):
        self.root = join(root, "MCubeS")
        self.split = "test" if image_set in ("val", "test") else image_set
        self.transform = transform
        self.target_transform = target_transform
        self.modals = modals or ["rgb"]
        self.n_classes = len(self.CLASSES)
        self.files = self._read_split(self.split)
        print(f"Found {len(self.files)} {image_set} images for MCubeS.")

    def _read_split(self, split_name):
        split_file = join(self.root, "list_folder", f"{split_name}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        with open(split_file) as f:
            return [line.strip() for line in f if line.strip()]

    @staticmethod
    def _normalize_to_uint8(arr):
        a_min, a_max = float(arr.min()), float(arr.max())
        if a_max - a_min <= 1e-6:
            return np.zeros_like(arr, dtype=np.uint8)
        return np.clip((arr - a_min) / (a_max - a_min) * 255.0, 0, 255).astype(np.uint8)

    def _load_modal(self, item_name, modal):
        if modal == "rgb":
            path = join(self.root, "polL_color", item_name + ".png")
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"RGB image not found: {path}")
            img = img[:, self.LEFT_OFFSET:, ::-1].copy()
            if img.dtype == np.uint16:
                img = (img >> 8).astype(np.uint8)
            return Image.fromarray(img.astype(np.uint8))

        if modal == "nir":
            path = join(self.root, "NIR_warped", item_name + ".png")
            nir = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if nir is None:
                raise FileNotFoundError(f"NIR image not found: {path}")
            nir = nir[:, self.LEFT_OFFSET:]
            if nir.dtype == np.uint16:
                nir = (nir >> 8).astype(np.uint8)
            return Image.fromarray(np.stack([nir, nir, nir], axis=2))

        if modal == "aolp":
            sin_path = join(self.root, "polL_aolp_sin", item_name + ".npy")
            cos_path = join(self.root, "polL_aolp_cos", item_name + ".npy")
            sin_arr = np.load(sin_path)[:, self.LEFT_OFFSET:]
            cos_arr = np.load(cos_path)[:, self.LEFT_OFFSET:]
            sin_u8 = self._normalize_to_uint8(sin_arr)
            cos_u8 = self._normalize_to_uint8(cos_arr)
            return Image.fromarray(np.stack([sin_u8, cos_u8, sin_u8], axis=2))

        if modal == "dolp":
            path = join(self.root, "polL_dolp", item_name + ".npy")
            dolp = self._normalize_to_uint8(np.load(path)[:, self.LEFT_OFFSET:])
            return Image.fromarray(np.stack([dolp, dolp, dolp], axis=2))

        raise NotImplementedError(f"MCubeS does not support modal '{modal}'")

    def __getitem__(self, index):
        item_name = self.files[index]
        modals_data = {modal: self._load_modal(item_name, modal) for modal in self.modals}

        label_path = join(self.root, "GT", item_name + ".png")
        label_arr = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if label_arr is None:
            raise FileNotFoundError(f"Label not found: {label_path}")
        label = Image.fromarray(label_arr[:, self.LEFT_OFFSET:])

        transformed, label = _sync_transforms(modals_data, label, self.transform, self.target_transform)
        mask = (label != IGNORE_INDEX).to(torch.float32)
        return transformed, label, mask

    def __len__(self):
        return len(self.files)


class CroppedDataset(Dataset):
    NYU_40_TO_13 = NYUDataset.MAPPING_40_TO_13
    NYU_40_TO_4 = NYUDataset.MAPPING_40_TO_4

    def __init__(self, root, dataset_name, crop_type, crop_ratio, image_set, transform,
                 target_transform, modals=None, num_classes=None):
        self.dataset_name = dataset_name
        self.modals = modals or ["rgb"]
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = num_classes or {"nyu": 13, "mfnet": 8, "mcubes": 20}[dataset_name]

        base = join(root, "cropped", f"{dataset_name}_{crop_type}_crop_{crop_ratio}")
        self.modal_dirs = {}
        for modal in self.modals:
            img_dir = join(base, modal, "img", image_set)
            label_dir = join(base, modal, "label", image_set)
            if not os.path.exists(img_dir):
                raise FileNotFoundError(f"Modal directory not found: {img_dir}")
            image_files = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
            self.modal_dirs[modal] = {"img_dir": img_dir, "label_dir": label_dir, "count": len(image_files)}

        counts = {v["count"] for v in self.modal_dirs.values()}
        if len(counts) != 1:
            raise ValueError(f"Mismatched cropped image counts: {counts}")
        self.num_images = counts.pop()

    def _restore_label(self, label):
        if self.dataset_name == "nyu":
            background = label == 0
            label = label - 1
            label[background] = IGNORE_INDEX
            mapping = self.NYU_40_TO_4 if self.num_classes == 4 else self.NYU_40_TO_13
            mapping = torch.tensor(mapping, dtype=label.dtype)
            mapped = torch.full_like(label, IGNORE_INDEX)
            valid = label != IGNORE_INDEX
            mapped[valid] = mapping[label[valid].long()]
            return mapped

        if self.dataset_name == "mfnet":
            unlabeled = label == 0
            label = label - 1
            label[unlabeled] = IGNORE_INDEX
            return label

        if self.dataset_name == "mcubes":
            ignore = label == 0
            label = label - 1
            label[ignore] = IGNORE_INDEX
            return label

        raise ValueError(f"Unsupported cropped dataset: {self.dataset_name}")

    def __getitem__(self, index):
        modals_data = {}
        target = None
        for modal in self.modals:
            dirs = self.modal_dirs[modal]
            modals_data[modal] = Image.open(join(dirs["img_dir"], f"{index}.jpg")).convert("RGB")
            if target is None:
                target = Image.open(join(dirs["label_dir"], f"{index}.png"))

        transformed, label = _sync_transforms(modals_data, target, self.transform, self.target_transform)
        label = self._restore_label(label)
        mask = (label != IGNORE_INDEX).to(torch.float32)
        return transformed, label.squeeze(0), mask

    def __len__(self):
        return self.num_images


class ContrastiveSegDataset(Dataset):
    DATASET_META = {
        "nyu": (13, NYUDataset, "nyu"),
        "mfnet": (8, MFNetDataset, "mfnet"),
        "mcubes": (20, MCubeSDataset, None),
    }

    def __init__(self, pytorch_data_dir, dataset_name, crop_type, image_set, transform,
                 target_transform, cfg, aug_geometric_transform=None,
                 aug_photometric_transform=None, num_neighbors=5, compute_knns=False,
                 mask=False, pos_labels=False, pos_images=False, extra_transform=None,
                 model_type_override=None):
        if dataset_name not in self.DATASET_META:
            raise ValueError(f"Unsupported dataset '{dataset_name}'. Use one of {sorted(self.DATASET_META)}.")

        self.num_neighbors = num_neighbors
        self.dataset_name = dataset_name
        self.mask = mask
        self.pos_labels = pos_labels
        self.pos_images = pos_images
        self.extra_transform = extra_transform
        self.aug_geometric_transform = aug_geometric_transform
        self.aug_photometric_transform = aug_photometric_transform

        modals = cfg.get("modals", ["rgb"])
        default_classes, dataset_cls, root_suffix = self.DATASET_META[dataset_name]
        self.n_classes = cfg.get("nyu_num_classes", default_classes) if dataset_name == "nyu" else default_classes

        if crop_type is not None:
            self.dataset = CroppedDataset(
                root=pytorch_data_dir,
                dataset_name=dataset_name,
                crop_type=cfg.crop_type,
                crop_ratio=cfg.crop_ratio,
                image_set=image_set,
                transform=transform,
                target_transform=target_transform,
                modals=modals,
                num_classes=self.n_classes,
            )
        else:
            root = join(pytorch_data_dir, root_suffix) if root_suffix else pytorch_data_dir
            kwargs = {"num_classes": self.n_classes} if dataset_name == "nyu" else {}
            self.dataset = dataset_cls(root, image_set, transform, target_transform, modals=modals, **kwargs)

        if pos_labels or pos_images:
            self.nns = self._load_knns(
                pytorch_data_dir, cfg, image_set, crop_type, compute_knns, model_type_override
            )
            assert len(self.dataset) == self.nns.shape[0]

    def _load_knns(self, pytorch_data_dir, cfg, image_set, crop_type, compute_knns, model_type_override):
        model_type = model_type_override or cfg.model_type
        dino_version = cfg.get("dino_version", "v3")
        patch_size = cfg.get("dino_patch_size", 16)
        model_info = f"{model_type}_dino{dino_version}_p{patch_size}"
        path = join(
            pytorch_data_dir,
            "nns",
            f"nns_{model_info}_{self.dataset_name}_{image_set}_{crop_type}_{cfg.res}.npz",
        )
        if not os.path.exists(path) or compute_knns:
            raise ValueError(f"Could not find KNN cache {path}. Run src/precompute_knns.py first.")

        loaded = np.load(path)
        if "dino_version" in loaded:
            print(
                f"Loaded KNN: DINO {loaded['dino_version']}, "
                f"{loaded['model_type']}, patch_size={int(loaded['patch_size'])}"
            )
        return loaded["nns"]

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _set_seed(seed):
        random.seed(seed)
        torch.manual_seed(seed)

    def __getitem__(self, ind):
        modals_data, label, mask = self.dataset[ind]

        if self.pos_images or self.pos_labels:
            pos_idx = self.nns[ind][torch.randint(low=1, high=self.num_neighbors + 1, size=[]).item()]
            modals_data_pos, label_pos, mask_pos = self.dataset[pos_idx]

        seed = np.random.randint(2147483647)
        self._set_seed(seed)

        first_img = next(iter(modals_data.values()))
        coord_entries = torch.meshgrid(
            torch.linspace(-1, 1, first_img.shape[1]),
            torch.linspace(-1, 1, first_img.shape[2]),
            indexing="ij",
        )
        coord = torch.cat([t.unsqueeze(0) for t in coord_entries], 0)
        extra_transform = self.extra_transform or (lambda _i, x: x)

        ret = {"ind": ind}
        for modal, img in modals_data.items():
            ret[modal] = extra_transform(ind, img)
        first_modal = "rgb" if "rgb" in ret else next(iter(modals_data))
        ret["img"] = ret[first_modal]
        ret["label"] = extra_transform(ind, label)

        if self.pos_images:
            for modal, img in modals_data_pos.items():
                ret[f"{modal}_pos"] = extra_transform(ind, img)
            first_pos = "rgb" if "rgb" in modals_data_pos else next(iter(modals_data_pos))
            ret["img_pos"] = ret[f"{first_pos}_pos"]
            ret["ind_pos"] = pos_idx

        if self.mask:
            ret["mask"] = mask

        if self.pos_labels:
            ret["label_pos"] = extra_transform(ind, label_pos)
            ret["mask_pos"] = mask_pos

        if self.aug_photometric_transform is not None:
            first_modal = next(iter(modals_data))
            img_aug = self.aug_photometric_transform(self.aug_geometric_transform(modals_data[first_modal]))
            self._set_seed(seed)
            coord_aug = self.aug_geometric_transform(coord)
            ret["img_aug"] = img_aug
            ret["coord_aug"] = coord_aug.permute(1, 2, 0)

        return ret
