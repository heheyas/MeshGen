import torch
from transformers import AutoImageProcessor, AutoModel


class FrozenDINOv2Encoder(torch.nn.Module):
    def __init__(self, model_name="facebook/dinov2-giant", do_rescale=True):
        super().__init__()
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(
            model_name, do_rescale=do_rescale
        )
        self.model = AutoModel.from_pretrained(model_name)

        self.output_dim = self.model.config.hidden_size

        self.freeze()
        self.eval()

    def freeze(self):
        print(f"======== Freezing DinoWrapper ========")
        self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        input = self.processor(images=x, return_tensors="pt")
        input["pixel_values"] = input["pixel_values"].to(self.device)
        outputs = self.model(**input)

        return outputs.last_hidden_state
