import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN


class RealESRGANUpscaler(torch.nn.Module):
    def __init__(self, scale: int = 4):
        super().__init__()
        model_path = f"weights/RealESRGAN_x{scale}.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RealESRGAN(self.device, scale=scale)
        self.model.load_weights(model_path, download=True)

    def forward(self, image):
        return np.asarray(self.model.predict(image))


class ControlNetUpscaler(torch.nn.Module):
    def __init__(self):
        pass
