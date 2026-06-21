import os
from data import ContrastiveSegDataset
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from torchvision.transforms.functional import five_crop, crop
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
from os.path import join
from torchvision import transforms as T
from utils import ToTargetTensor, prep_args, resolve_path

# Helper function to get image size (replaces deprecated _get_image_size)
def _get_image_size(img):
    if hasattr(img, 'shape'):  # torch.Tensor
        return img.shape[-1], img.shape[-2]  # width, height
    else:  # PIL Image
        return img.size  # width, height


def _random_crops(img, size, seed, n):
    """Crop the given image into four corners and the central crop.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).

    Returns:
       tuple: tuple (tl, tr, bl, br, center)
                Corresponding top left, top right, bottom left, bottom right and center crop.
    """
    if isinstance(size, int):
        size = (int(size), int(size))
    elif isinstance(size, (tuple, list)) and len(size) == 1:
        size = (size[0], size[0])

    if len(size) != 2:
        raise ValueError("Please provide only two dimensions (h, w) for size.")

    image_width, image_height = _get_image_size(img)
    crop_height, crop_width = size
    if crop_width > image_width or crop_height > image_height:
        msg = "Requested crop size {} is bigger than input size {}"
        raise ValueError(msg.format(size, (image_height, image_width)))

    images = []
    for i in range(n):
        seed1 = hash((seed, i, 0))
        seed2 = hash((seed, i, 1))
        crop_height, crop_width = int(crop_height), int(crop_width)

        top = seed1 % (image_height - crop_height)
        left = seed2 % (image_width - crop_width)
        images.append(crop(img, top, left, crop_height, crop_width))

    return images


class RandomCropComputer(Dataset):

    def _get_size(self, img):
        if len(img.shape) == 3:
            return [int(img.shape[1] * self.crop_ratio), int(img.shape[2] * self.crop_ratio)]
        elif len(img.shape) == 2:
            return [int(img.shape[0] * self.crop_ratio), int(img.shape[1] * self.crop_ratio)]
        else:
            raise ValueError("Bad image shape {}".format(img.shape))

    def random_crops(self, i, img):
        return _random_crops(img, self._get_size(img), i, 5)

    def five_crops(self, i, img):
        return five_crop(img, self._get_size(img))

    def __init__(self, cfg, dataset_name, img_set, crop_type, crop_ratio):
        self.pytorch_data_dir = cfg.pytorch_data_dir
        self.crop_ratio = crop_ratio
        self.dataset_name = dataset_name
        self.crop_type = crop_type
        self.img_set = img_set
        self.cfg = cfg
        self.modals = cfg.get('modals', ['rgb'])

        self.save_dirs = {}
        for modal in self.modals:
            base_save_dir = join(
                cfg.pytorch_data_dir, "cropped", "{}_{}_crop_{}".format(dataset_name, crop_type, crop_ratio))
            modal_save_dir = join(base_save_dir, modal)
            self.save_dirs[modal] = {
                'img_dir': join(modal_save_dir, "img", img_set),
                'label_dir': join(modal_save_dir, "label", img_set)
            }
            os.makedirs(self.save_dirs[modal]['img_dir'], exist_ok=True)
            os.makedirs(self.save_dirs[modal]['label_dir'], exist_ok=True)

        if crop_type == "random":
            cropper = lambda i, x: self.random_crops(i, x)
        elif crop_type == "five":
            cropper = lambda i, x: self.five_crops(i, x)
        else:
            raise ValueError('Unknown crop type {}'.format(crop_type))

        self.dataset = ContrastiveSegDataset(
            cfg.pytorch_data_dir,
            dataset_name,
            None,
            img_set,
            T.ToTensor(),
            ToTargetTensor(),
            cfg=cfg,
            num_neighbors=cfg.num_neighbors,
            pos_labels=False,
            pos_images=False,
            mask=False,
            aug_geometric_transform=None,
            aug_photometric_transform=None,
            extra_transform=cropper
        )

    def __getitem__(self, item):
        batch = self.dataset[item]
        label_crops = batch['label']

        for modal in self.modals:
            img_crops = batch[modal]

            for crop_num, (img, label) in enumerate(zip(img_crops, label_crops)):
                img_num = item * 5 + crop_num

                img_arr = img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                img_path = join(self.save_dirs[modal]['img_dir'], "{}.jpg".format(img_num))
                Image.fromarray(img_arr).save(img_path, 'JPEG')

                label_arr = (label + 1).unsqueeze(0).permute(1, 2, 0).to('cpu', torch.uint8).numpy().squeeze(-1)
                label_path = join(self.save_dirs[modal]['label_dir'], "{}.png".format(img_num))
                Image.fromarray(label_arr).save(label_path, 'PNG')

        return True

    def __len__(self):
        return len(self.dataset)


@hydra.main(version_base="1.1", config_path="configs", config_name="train_config_nyu.yml")
def my_app(cfg: DictConfig) -> None:
    cfg.pytorch_data_dir = resolve_path(cfg.pytorch_data_dir)
    print(OmegaConf.to_yaml(cfg))
    seed_everything(seed=0, workers=True)

    dataset_names = [cfg.dataset_name]
    img_sets = ["train", "val"]
    crop_types = [cfg.crop_type]
    crop_ratios = [cfg.crop_ratio]

    for crop_ratio in crop_ratios:
        for crop_type in crop_types:
            for dataset_name in dataset_names:
                for img_set in img_sets:
                    print(f"\nProcessing {dataset_name} {img_set} with {crop_type} crop ratio {crop_ratio}")
                    dataset = RandomCropComputer(cfg, dataset_name, img_set, crop_type, crop_ratio)
                    loader = DataLoader(dataset, 1, shuffle=False, num_workers=cfg.num_workers, collate_fn=lambda l: l)
                    for _ in tqdm(loader):
                        pass
                    base_save_dir = os.path.join(
                        cfg.pytorch_data_dir, "cropped",
                        "{}_{}_crop_{}".format(dataset_name, crop_type, crop_ratio))
                    print(f"Saved to {base_save_dir}")


if __name__ == "__main__":
    prep_args()
    my_app()
