# DINOv3 Pretrained Weights

Place downloaded DINOv3 pretrained `.pth` files in this directory.

The released configs expect files such as:

```text
pretrained/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
```

You can also keep weights elsewhere and override `pretrained_weights` in the
corresponding config file. Do not commit large `.pth` files to this repository.
