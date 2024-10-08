import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

_model = None


def run_model(image):

    model, transform = get_model()

    image_size = image.size
    input_images = transform(image).unsqueeze(0).to("cuda")
    # Prediction
    with torch.no_grad():
        preds = model(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)

    return image


def get_model():
    global _model

    if _model is not None:
        return _model

    birefnet = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True
    )
    birefnet.to("cuda")
    transform_image = transforms.Compose(
        [
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    _model = birefnet, transform_image

    return _model
