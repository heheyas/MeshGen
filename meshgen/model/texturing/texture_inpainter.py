import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from torchvision.transforms import v2
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchvision.utils import make_grid, save_image
from einops import rearrange
from PIL import Image
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from meshgen.util import instantiate_from_config
from meshgen.utils.io import write_image
from meshgen.modules.mesh.textured_mesh import TexturedMesh
import diffusers
from diffusers import (
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    StableDiffusionPipeline,
    StableDiffusionXLControlNetPipeline,
    DDIMScheduler,
    ImagePipelineOutput,
    StableDiffusionControlNetInpaintPipeline,
)
from diffusers.pipelines.controlnet.pipeline_controlnet import *
from transformers import CLIPVisionModelWithProjection
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def get_device():
    return torch.device(f"cuda:{get_rank()}")


class TextureLoraStableDiffusion(pl.LightningModule):
    def __init__(self, base_model, lora_rank=8, drop_prop=0.1, ckpt_path=None):
        super().__init__()
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            base_model, safety_checker=None
        )
        self.unet = self.pipeline.unet
        unet_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        self.drop_prob = drop_prop
        self.unet.add_adapter(unet_lora_config)

        self.lora_layers = filter(lambda p: p.requires_grad, self.unet.parameters())

        train_sched = DDPMScheduler.from_config(self.pipeline.scheduler.config)
        self.train_scheduler = train_sched
        self.num_timesteps = 1000
        self.validation_step_outputs = []

        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location="cpu")
            if "state_dict" in sd:
                sd = sd["state_dict"]
            self.load_state_dict(sd)

    def on_fit_start(self):
        device = get_device()
        self.pipeline.to(device)
        if self.global_rank == 0:
            os.makedirs(os.path.join(self.logdir, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.logdir, "images_val"), exist_ok=True)

    @torch.no_grad()
    def encode_target_images(self, images):
        dtype = next(self.pipeline.vae.parameters()).dtype
        latents = self.pipeline.vae.encode(images.to(dtype=dtype)).latent_dist.sample()
        latents = latents * self.pipeline.vae.config.scaling_factor

        return latents

    @torch.no_grad()
    def encode_prompts(self, prompts):
        if hasattr(self.pipeline, "encode_prompt"):
            encoder_hidden_states = self.pipeline.encode_prompt(
                prompts, self.device, 1, False
            )[0]
        else:
            encoder_hidden_states = self.pipeline._encode_prompt(
                prompts, self.device, 1, False
            )

        return encoder_hidden_states

    def training_step(self, batch, batch_idx):
        target, prompt = batch["target"], batch["prompt"]
        prompt = [p if np.random.rand() > self.drop_prob else "" for p in prompt]
        latents = self.encode_target_images(target)
        B = latents.shape[0]
        t = torch.randint(0, self.num_timesteps, size=(B,)).long().to(self.device)
        noise = torch.randn_like(latents)
        noisy_latents = self.train_scheduler.add_noise(latents, noise, t)

        encoder_hidden_states = self.encode_prompts(prompt)

        model_pred = self.unet(
            noisy_latents, t, encoder_hidden_states, return_dict=False
        )[0]

        if self.train_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.train_scheduler.config.prediction_type == "v_prediction":
            target = self.train_scheduler.get_velocity(latents, noise, t)
        else:
            raise ValueError(
                f"Unknown prediction type {self.train_scheduler.config.prediction_type}"
            )
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        loss_dict = {"train/loss": loss}

        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        target, prompts = batch["target"], batch["prompt"]
        target_pils = [to_pil_image(tar * 0.5 + 0.5) for tar in target]

        outputs = []
        for target_pil, prompt in zip(target_pils, prompts):
            output = self.pipeline(prompt=prompt, num_inference_steps=75).images[0]
            output = to_tensor(output).unsqueeze(0)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0).to(self.device)
        images = torch.cat([target * 0.5 + 0.5, outputs], dim=-2)
        self.validation_step_outputs.append(images.cpu())

    @torch.no_grad()
    def on_validation_epoch_end(self):
        images = torch.cat(self.validation_step_outputs, dim=0)

        all_images = self.all_gather(images)
        all_images = rearrange(all_images, "r b c h w -> (r b) c h w")

        if self.global_rank == 0:
            grid = make_grid(all_images, nrow=8, normalize=True, value_range=(0, 1))
            save_image(
                grid,
                os.path.join(
                    self.logdir, "images_val", f"val_{self.global_step:07d}.png"
                ),
            )

        self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        lr = self.learning_rate
        return torch.optim.AdamW(self.lora_layers, lr=lr)

    def to(self, *args, **kwargs):
        self.pipeline.to(*args, **kwargs)
        return super().to(*args, **kwargs)


class TextureInpainter(pl.LightningModule):
    def __init__(
        self,
        base_model,
        controlnet_channels=9,
        drop_prob=0.1,
        lora_config=None,
        lora_ft_ckpt=None,
        ckpt_path=None,
    ):
        super().__init__()
        self.pipeline = StableDiffusionPipeline.from_pretrained(base_model)
        self.unet = self.pipeline.unet

        if lora_config is not None:
            unet_lora_config = instantiate_from_config(lora_config)
            self.unet.add_adapter(unet_lora_config)
            sd = torch.load(lora_ft_ckpt, map_location="cpu")
            if "state_dict" in sd:
                sd = sd["state_dict"]
            true_sd = {k[5:]: v for k, v in sd.items() if k.startswith("unet")}
            self.unet.load_state_dict(true_sd)
            # self.unet = self.unet.merge_and_unload()
            # self.unet.fuse_lora()
            self.pipeline.fuse_lora()
            self.pipeline.unload_lora_weights()

        self.controlnet = diffusers.ControlNetModel.from_unet(
            self.unet, conditioning_channels=controlnet_channels
        )
        self.drop_prob = drop_prob

        for n, p in self.unet.named_parameters():
            p.requires_grad = False

        train_sched = DDPMScheduler.from_config(self.pipeline.scheduler.config)
        self.train_scheduler = train_sched
        self.num_timesteps = 1000

        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location="cpu")
            if "state_dict" in sd:
                sd = sd["state_dict"]
            self.load_state_dict(sd)

        # self.validation_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        #     base_model,
        #     unet=self.unet,
        #     controlnet=self.controlnet,
        # )
        self.validation_pipeline = (
            StableDiffusionControlNetInpaintPipeline.from_pretrained(
                base_model,
                unet=self.unet,
                controlnet=self.controlnet,
                safety_checker=None,
            )
        )
        # self.validation_pipeline.set_progress_bar_config(disable=True)
        self.validation_step_outputs = []

    def on_fit_start(self):
        device = get_device()
        self.pipeline.to(device)
        if self.global_rank == 0:
            os.makedirs(os.path.join(self.logdir, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.logdir, "images_val"), exist_ok=True)

    @torch.no_grad()
    def encode_target_images(self, images):
        dtype = next(self.pipeline.vae.parameters()).dtype
        latents = self.pipeline.vae.encode(images.to(dtype=dtype)).latent_dist.sample()
        latents = latents * self.pipeline.vae.config.scaling_factor

        return latents

    @torch.no_grad()
    def encode_prompts(self, prompts):
        if hasattr(self.pipeline, "encode_prompt"):
            encoder_hidden_states = self.pipeline.encode_prompt(
                prompts, self.device, 1, False
            )[0]
        else:
            encoder_hidden_states = self.pipeline._encode_prompt(
                prompts, self.device, 1, False
            )

        return encoder_hidden_states

    def training_step(self, batch, batch_idx):
        target, control, prompt = batch["target"], batch["control"], batch["prompt"]
        prompt = [p if np.random.rand() > self.drop_prob else "" for p in prompt]
        latents = self.encode_target_images(target)
        B = latents.shape[0]
        t = torch.randint(0, self.num_timesteps, size=(B,)).long().to(self.device)
        noise = torch.randn_like(latents)
        noisy_latents = self.train_scheduler.add_noise(latents, noise, t)

        encoder_hidden_states = self.encode_prompts(prompt)

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            noisy_latents,
            t,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=control,
            return_dict=False,
        )

        model_pred = self.unet(
            noisy_latents,
            t,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=[
                sample for sample in down_block_res_samples
            ],
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]

        if self.train_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.train_scheduler.config.prediction_type == "v_prediction":
            target = self.train_scheduler.get_velocity(latents, noise, t)
        else:
            raise ValueError(
                f"Unknown prediction type {self.train_scheduler.config.prediction_type}"
            )
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        loss_dict = {"train/loss": loss}

        self.log_dict(
            loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False,
        )
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr_abs", lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def make_inpaint_condition(self, image, image_mask):
        image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

        assert (
            image.shape[0:1] == image_mask.shape[0:1]
        ), "image and image_mask must have the same image size"
        image[image_mask > 0.5] = -1.0  # set as masked pixel
        image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return image

    def fill_image(self, image, image_mask, inpaintRadius=3):
        image = np.array(image.convert("RGB"))
        image_mask = (np.array(image_mask.convert("L"))).astype(np.uint8)
        filled_image = cv2.inpaint(image, image_mask, inpaintRadius, cv2.INPAINT_TELEA)

        res_img = Image.fromarray(np.clip(filled_image, 0, 255).astype(np.uint8))
        # res_img.save("trash/filled_image.png")
        # breakpoint()
        return res_img

    def validation_step(self, batch, batch_idx):
        target, controls, prompts = batch["target"], batch["control"], batch["prompt"]
        masks = batch["mask"]
        target_pils = [to_pil_image(tar * 0.5 + 0.5) for tar in target]

        outputs = []
        for idx, (target_pil, control, prompt) in enumerate(
            zip(target_pils, controls, prompts)
        ):
            image = to_pil_image(control[6:])
            mask = to_pil_image(masks[idx])
            image = self.fill_image(image, mask)
            output = self.validation_pipeline(
                prompt=prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=75,
                control_image=controls[idx : idx + 1],
                strength=1.0,
            ).images[0]
            output = to_tensor(output).unsqueeze(0)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0).to(self.device)
        images = torch.cat([target * 0.5 + 0.5, outputs], dim=-2)
        control_imgs = controls[:, 6:]
        images = torch.cat(
            [images, control_imgs, controls[:, :3], controls[:, 3:6]], dim=-2
        )
        self.validation_step_outputs.append(images.cpu())

    def test_step(self, batch, batch_idx):
        target, controls, prompts = batch["target"], batch["control"], batch["prompt"]
        masks = batch["mask"]
        target_pils = [to_pil_image(tar * 0.5 + 0.5) for tar in target]

        outputs = []
        for idx, (target_pil, control, prompt) in enumerate(
            zip(target_pils, controls, prompts)
        ):
            image = to_pil_image(control[6:])
            mask = to_pil_image(masks[idx])
            image = self.fill_image(image, mask)
            output = self.validation_pipeline(
                prompt=prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=75,
                control_image=controls[idx : idx + 1],
                strength=1.0,
            ).images[0]
            output = to_tensor(output).unsqueeze(0)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=0).to(self.device)
        images = torch.cat([target * 0.5 + 0.5, outputs], dim=-2)
        control_imgs = controls[:, 6:]
        images = torch.cat(
            [images, control_imgs, controls[:, :3], controls[:, 3:6]], dim=-2
        )
        self.validation_step_outputs.append(images.cpu())

    @torch.no_grad()
    def on_validation_epoch_end(self):
        images = torch.cat(self.validation_step_outputs, dim=0)

        all_images = self.all_gather(images)
        all_images = rearrange(all_images, "r b c h w -> (r b) c h w")

        if self.global_rank == 0:
            grid = make_grid(all_images, nrow=8, normalize=True, value_range=(0, 1))
            save_image(
                grid,
                os.path.join(
                    self.logdir, "images_val", f"val_{self.global_step:07d}.png"
                ),
            )

        self.validation_step_outputs.clear()  # free memory

    def configure_optimizers(self):
        lr = self.learning_rate
        return torch.optim.AdamW(self.controlnet.parameters(), lr=lr)

    def forward(self, *args, **kwargs):
        return self.validation_pipeline(*args, **kwargs).images

    def to(self, *args, **kwargs):
        self.pipeline.to(*args, **kwargs)
        self.validation_pipeline.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def inference(
        self,
        prompt,
        image,
        mask,
        UV_pos,
        UV_normal,
        num_inference_steps=30,
        strength=1.0,
        controlnet_conditioning_scale=1.0,
        **kwargs,
    ):
        if isinstance(image, torch.Tensor):
            image = to_pil_image(image)
        if isinstance(mask, torch.Tensor):
            mask = to_pil_image(mask)

        if isinstance(UV_pos, Image.Image):
            UV_pos = to_tensor(UV_pos)

        if isinstance(UV_normal, Image.Image):
            UV_normal = to_tensor(UV_normal)

        image = self.fill_image(image, mask)
        # image.save("trash/filled_image.png")
        masked_image = self.make_inpaint_condition(image, mask).to(UV_pos)[0]

        latent_mask = to_tensor(mask.convert("L")).unsqueeze(0)
        latent_mask = (
            F.interpolate(latent_mask, (64, 64), mode="nearest")
            .to(self.pipeline.device)
            .to(self.pipeline.dtype)
        )
        latent_mask = 1 - latent_mask
        # write_image("trash/latent_mask.png", latent_mask[0], format="chw")
        # breakpoint()
        original_latent = self.encode_target_images(
            to_tensor(image)[None].to(self.pipeline.dtype).to(self.pipeline.device)
        )

        def latent_mask_callback(pipeline, i, t, callback_kwargs):
            latents = callback_kwargs["latents"]
            noisy_original_latent = pipeline.scheduler.add_noise(
                original_latent, torch.randn_like(original_latent), t
            )
            # breakpoint()

            new_latents = latents * latent_mask + noisy_original_latent * (
                1 - latent_mask
            )

            return {"latents": new_latents.to(pipeline.dtype)}

        control = torch.cat([UV_pos, UV_normal, masked_image], dim=0)
        width, height = image.size
        output = self.validation_pipeline(
            prompt=prompt,
            height=height,
            width=width,
            image=image,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
            control_image=control[None],
            strength=strength,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            # callback_on_step_end=latent_mask_callback,
            negative_prompt="low quality, noisy image, over-exposed, shadow",
            latents=torch.randn_like(original_latent),
            **kwargs,
        ).images[0]

        return output
