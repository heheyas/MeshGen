import torch
from PIL import Image
from rembg import remove, new_session
from kiui.op import recenter
import numpy as np
from transformers import pipeline
from .birefnet import run_model as remove_bg_birefnet

_rembg_session = None
_bria_model = None


def get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        _rembg_session = new_session()

    return _rembg_session


def get_bria_model():
    global _bria_model
    if _bria_model is None:
        _bria_model = pipeline(
            "image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True
        )

    return _bria_model


def remove_bg_rembg(image, **kwargs):
    return remove(image, **kwargs)


def remove_bg_briarmbg(image, **kwargs):
    model = get_bria_model()
    return model(image)


def preprocess_image(
    image: Image.Image,
    size: int | tuple[int] = 512,
    border_ratio: None | float = None,
    remove_bg: bool = False,
    ignore_alpha: bool = False,
    alpha_matting: bool = True,
    backend="bria",
):
    rembg_session = get_rembg_session()

    if border_ratio > 0:
        if image.mode != "RGBA" or ignore_alpha:
            image = image.convert("RGB")
            if backend == "rembg":
                carved_image = remove_bg_rembg(
                    image, alpha_matting=alpha_matting, session=rembg_session
                )  # [H, W, 4]
            elif backend == "bria":
                carved_image = remove_bg_briarmbg(image)
            elif backend == "birefnet":
                carved_image = remove_bg_birefnet(image)
            else:
                raise ValueError(f"Unknown backend: {backend}")
            carved_image = np.asarray(carved_image)
        else:
            image = np.asarray(image)
            carved_image = image
        mask = carved_image[..., -1] > 0
        image = recenter(carved_image, mask, border_ratio=border_ratio)
        image = image.astype(np.float32) / 255.0

        if remove_bg:
            if image.shape[-1] == 4:
                image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
        else:
            image = image
        image = Image.fromarray((image * 255).astype(np.uint8))
    # else:
    #     # raise ValueError("border_ratio must be set currently")
    #     pass

    if isinstance(size, int):
        size = (size, size)

    image = image.resize(size)

    return image
