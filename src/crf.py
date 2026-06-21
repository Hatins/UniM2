import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF

from utils import unnorm


MAX_ITER = 10
POS_W = 3
POS_XY_STD = 1
BILATERAL_W = 4
BILATERAL_XY_STD = 67
BILATERAL_RGB_STD = 3


def dense_crf(image_tensor: torch.FloatTensor, output_logits: torch.FloatTensor):
    try:
        import pydensecrf.densecrf as dcrf
        import pydensecrf.utils as crf_utils
    except ImportError as exc:
        raise ImportError("run_crf=True requires pydensecrf. Please install pydensecrf first.") from exc

    image = np.array(VF.to_pil_image(unnorm(image_tensor)))[:, :, ::-1]
    height, width = image.shape[:2]
    image = np.ascontiguousarray(image)

    output_logits = F.interpolate(
        output_logits.unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    output_probs = F.softmax(output_logits, dim=0).cpu().numpy()

    num_classes = output_probs.shape[0]
    unary = crf_utils.unary_from_softmax(output_probs)
    unary = np.ascontiguousarray(unary)

    crf = dcrf.DenseCRF2D(width, height, num_classes)
    crf.setUnaryEnergy(unary)
    crf.addPairwiseGaussian(sxy=POS_XY_STD, compat=POS_W)
    crf.addPairwiseBilateral(
        sxy=BILATERAL_XY_STD,
        srgb=BILATERAL_RGB_STD,
        rgbim=image,
        compat=BILATERAL_W,
    )

    refined = crf.inference(MAX_ITER)
    return np.array(refined).reshape((num_classes, height, width))
