import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from torchvision.transforms import v2
from torchvision.transforms.functional import to_tensor
from torchvision.utils import make_grid, save_image
from einops import rearrange
from PIL import Image

from meshgen.util import instantiate_from_config
from meshgen.utils.io import write_image
from meshgen.modules.mesh.textured_mesh import TexturedMesh
from diffusers import (
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
)
from .pipeline import (
    RefOnlyNoisedUNet,
    DepthControlUNet,
    NormalDepthControlUNet,
    NormalControlUNet,
)

from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    DDIMScheduler,
    ImagePipelineOutput,
)
from diffusers.pipelines.controlnet.pipeline_controlnet import *
from transformers import CLIPVisionModelWithProjection
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .utils import step_tex


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = (
        guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    )
    return noise_cfg


def to_rgb_image(maybe_rgba: Image.Image):
    if maybe_rgba.mode == "RGB":
        return maybe_rgba
    elif maybe_rgba.mode == "RGBA":
        rgba = maybe_rgba
        img = np.random.randint(
            255, 256, size=[rgba.size[1], rgba.size[0], 3], dtype=np.uint8
        )
        img = Image.fromarray(img, "RGB")
        img.paste(rgba, mask=rgba.getchannel("A"))
        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)


def scale_latents(latents):
    latents = (latents - 0.22) * 0.75
    return latents


def unscale_latents(latents):
    latents = latents / 0.75 + 0.22
    return latents


def scale_image(image):
    image = image * 0.5 / 0.8
    return image


def unscale_image(image):
    image = image / 0.5 * 0.8
    return image


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


def scale_latents(latents):
    latents = (latents - 0.22) * 0.75
    return latents


def unscale_latents(latents):
    latents = latents / 0.75 + 0.22
    return latents


def scale_image(image):
    image = image * 0.5 / 0.8
    return image


def unscale_image(image):
    image = image / 0.5 * 0.8
    return image


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class MVJointControlNet(pl.LightningModule):
    def __init__(
        self,
        stable_diffusion_config,
        control_type="normal",
        drop_cond_prob=0.1,
        unet_ckpt_path=None,
        ckpt_path=None,
        conditioning_scale=1.0,
        scheduler_type="euler",
    ):
        super(MVJointControlNet, self).__init__()

        self.drop_cond_prob = drop_cond_prob
        self.control_type = control_type

        self.register_schedule()

        # init modules
        pipeline = DiffusionPipeline.from_pretrained(**stable_diffusion_config)
        # pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        #     pipeline.scheduler.config, timestep_spacing="trailing"
        # )

        if scheduler_type == "ddim":
            pipeline.scheduler = DDIMScheduler.from_config(
                pipeline.scheduler.config, timestep_spacing="trailing"
            )
        elif scheduler_type == "euler":
            pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                pipeline.scheduler.config, timestep_spacing="trailing"
            )
        elif scheduler_type == "ddpm":
            pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        self.pipeline = pipeline
        self.pipe = self.pipeline

        train_sched = DDPMScheduler.from_config(self.pipeline.scheduler.config)

        if unet_ckpt_path is not None:
            sd = torch.load(unet_ckpt_path, map_location="cpu")
            if "state_dict" in sd:
                sd = sd["state_dict"]
            pipeline.unet.load_state_dict(sd)

        if isinstance(self.pipeline.unet, UNet2DConditionModel):
            self.pipeline.unet = RefOnlyNoisedUNet(
                self.pipeline.unet, train_sched, self.pipeline.scheduler
            )

        self.train_scheduler = train_sched  # use ddpm scheduler during training

        type2cnet = {
            "normal": NormalControlUNet,
            "depth": DepthControlUNet,
            "normal_depth": NormalDepthControlUNet,
        }
        self.control_type = control_type

        pipeline.unet = type2cnet[control_type](
            pipeline.unet, conditioning_scale=conditioning_scale
        )

        self.unet = pipeline.unet

        for n, p in self.unet.unet.named_parameters():
            p.requires_grad = False

        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location="cpu")
            if "state_dict" in sd:
                sd = sd["state_dict"]
            self.load_state_dict(sd)

        # validation output buffer
        self.validation_step_outputs = []

        bg = torch.ones(1, 3, 960, 640, dtype=self.pipeline.dtype, device=self.device)
        bg_latent = self.encode_target_images(bg)
        self.register_buffer("bg_latent", bg_latent)

    def change_scheduler(self, scheduler_type):
        pipeline = self.pipeline
        if scheduler_type == "ddim":
            pipeline.scheduler = DDIMScheduler.from_config(
                pipeline.scheduler.config, timestep_spacing="trailing"
            )
            # pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        elif scheduler_type == "euler":
            pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                pipeline.scheduler.config, timestep_spacing="trailing"
            )
        elif scheduler_type == "ddpm":
            pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def register_schedule(self):
        self.num_timesteps = 1000

        # replace scaled_linear schedule with linear schedule as Zero123++
        beta_start = 0.00085
        beta_end = 0.0120
        betas = torch.linspace(beta_start, beta_end, 1000, dtype=torch.float32)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=torch.float64), alphas_cumprod[:-1]], 0
        )

        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas_cumprod", alphas_cumprod.float())
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev.float())

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).float())
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod).float()
        )

        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod).float()
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1).float()
        )

    def on_fit_start(self):
        device = get_device()
        self.pipeline.to(device)
        if self.global_rank == 0:
            os.makedirs(os.path.join(self.logdir, "images"), exist_ok=True)
            os.makedirs(os.path.join(self.logdir, "images_val"), exist_ok=True)

    def prepare_batch_data(self, batch):
        # prepare stable diffusion input
        cond_imgs = batch["cond_imgs"]  # (B, C, H, W)
        cond_imgs = cond_imgs.to(self.device)

        # random resize the condition image
        cond_size = np.random.randint(128, 513)
        cond_imgs = v2.functional.resize(
            cond_imgs, cond_size, interpolation=3, antialias=True
        ).clamp(0, 1)

        target_imgs = batch["target_imgs"]  # (B, 6, C, H, W)
        target_imgs = v2.functional.resize(
            target_imgs, 320, interpolation=3, antialias=True
        ).clamp(0, 1)
        target_imgs = rearrange(
            target_imgs, "b (x y) c h w -> b c (x h) (y w)", x=3, y=2
        )  # (B, C, 3H, 2W)
        target_imgs = target_imgs.to(self.device)

        control_imgs = batch[self.control_type]
        control_imgs = v2.functional.resize(
            control_imgs, 320, interpolation=3, antialias=True
        ).clamp(0, 1)
        control_imgs = rearrange(
            control_imgs, "b (x y) c h w -> b c (x h) (y w)", x=3, y=2
        )  # (B, C, 3H, 2W)
        control_imgs = control_imgs.to(self.device)

        return cond_imgs, target_imgs, control_imgs, batch["target_type"]

    @torch.no_grad()
    def forward_vision_encoder(self, images, target_types=None):
        dtype = next(self.pipeline.vision_encoder.parameters()).dtype
        image_pil = [
            v2.functional.to_pil_image(images[i]) for i in range(images.shape[0])
        ]
        image_pt = self.pipeline.feature_extractor_clip(
            images=image_pil, return_tensors="pt"
        ).pixel_values
        image_pt = image_pt.to(device=self.device, dtype=dtype)
        global_embeds = self.pipeline.vision_encoder(
            image_pt, output_hidden_states=False
        ).image_embeds
        global_embeds = global_embeds.unsqueeze(-2)

        if hasattr(self.pipeline, "encode_prompt"):
            encoder_hidden_states = self.pipeline.encode_prompt(
                target_types, self.device, 1, False
            )[0]
        else:
            encoder_hidden_states = self.pipeline._encode_prompt(
                target_types, self.device, 1, False
            )

        ramp = global_embeds.new_tensor(
            self.pipeline.config.ramping_coefficients
        ).unsqueeze(-1)
        encoder_hidden_states = encoder_hidden_states + global_embeds * ramp

        return encoder_hidden_states

    @torch.no_grad()
    def encode_condition_image(self, images):
        dtype = next(self.pipeline.vae.parameters()).dtype
        image_pil = [
            v2.functional.to_pil_image(images[i]) for i in range(images.shape[0])
        ]
        image_pt = self.pipeline.feature_extractor_vae(
            images=image_pil, return_tensors="pt"
        ).pixel_values
        image_pt = image_pt.to(device=self.device, dtype=dtype)
        latents = self.pipeline.vae.encode(image_pt).latent_dist.sample()
        return latents

    @torch.no_grad()
    def encode_target_images(self, images):
        dtype = next(self.pipeline.vae.parameters()).dtype
        # equals to scaling images to [-1, 1] first and then call scale_image
        images = (images - 0.5) / 0.8  # [-0.625, 0.625]
        posterior = self.pipeline.vae.encode(images.to(dtype)).latent_dist
        latents = posterior.sample() * self.pipeline.vae.config.scaling_factor
        latents = scale_latents(latents)
        return latents

    def forward_unet(self, latents, t, prompt_embeds, cond_latents, control):
        dtype = next(self.pipeline.unet.parameters()).dtype
        latents = latents.to(dtype)
        prompt_embeds = prompt_embeds.to(dtype)
        cond_latents = cond_latents.to(dtype)
        cross_attention_kwargs = dict(cond_lat=cond_latents)
        cross_attention_kwargs[f"control_{self.control_type}"] = control
        pred_noise = self.pipeline.unet(
            latents,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]
        return pred_noise

    def predict_start_from_z_and_v(self, x_t, t, v):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def get_v(self, x, noise, t):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )

    def training_step(self, batch, batch_idx):
        # get input
        cond_imgs, target_imgs, control_imgs, target_types = self.prepare_batch_data(
            batch
        )

        # sample random timestep
        B = cond_imgs.shape[0]

        t = torch.randint(0, self.num_timesteps, size=(B,)).long().to(self.device)

        # classifier-free guidance
        if np.random.rand() < self.drop_cond_prob:
            if hasattr(self.pipeline, "encode_prompt"):
                # prompt_embeds = self.pipeline.encode_prompt(
                #     [""] * B, self.device, 1, False
                # )[0]
                prompt_embeds = self.pipeline.encode_prompt(
                    target_types, self.device, 1, False
                )[0]
            else:
                # prompt_embeds = self.pipeline._encode_prompt(
                #     [""] * B, self.device, 1, False
                # )
                prompt_embeds = self.pipeline._encode_prompt(
                    target_types, self.device, 1, False
                )
            cond_latents = self.encode_condition_image(torch.zeros_like(cond_imgs))
        else:
            prompt_embeds = self.forward_vision_encoder(cond_imgs, target_types)
            cond_latents = self.encode_condition_image(cond_imgs)

        latents = self.encode_target_images(target_imgs)
        noise = torch.randn_like(latents)
        latents_noisy = self.train_scheduler.add_noise(latents, noise, t)

        v_pred = self.forward_unet(
            latents_noisy, t, prompt_embeds, cond_latents, control_imgs
        )
        v_target = self.get_v(latents, noise, t)

        loss, loss_dict = self.compute_loss(v_pred, v_target)

        # logging
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

        if self.global_step % 500 == 0 and self.global_rank == 0:
            with torch.no_grad():
                latents_pred = self.predict_start_from_z_and_v(latents_noisy, t, v_pred)

                latents = unscale_latents(latents_pred)
                images = unscale_image(
                    self.pipeline.vae.decode(
                        latents / self.pipeline.vae.config.scaling_factor,
                        return_dict=False,
                    )[0]
                )  # [-1, 1]
                images = (images * 0.5 + 0.5).clamp(0, 1)
                images = torch.cat([target_imgs, images], dim=-2)

                if self.control_type in ["normal", "depth"]:
                    images = torch.cat([images, control_imgs], dim=-2)
                else:
                    images = torch.cat(
                        [images, control_imgs[:, :3], control_imgs[:, 3:]], dim=-2
                    )

                grid = make_grid(
                    images, nrow=images.shape[0], normalize=True, value_range=(0, 1)
                )
                save_image(
                    grid,
                    os.path.join(
                        self.logdir, "images", f"train_{self.global_step:07d}.png"
                    ),
                )

        return loss

    def compute_loss(self, noise_pred, noise_gt):
        loss = F.mse_loss(noise_pred, noise_gt)

        prefix = "train"
        loss_dict = {}
        loss_dict.update({f"{prefix}/loss": loss})

        return loss, loss_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # get input
        cond_imgs, target_imgs, control_imgs, target_types = self.prepare_batch_data(
            batch
        )

        images_pil = [
            v2.functional.to_pil_image(cond_imgs[i]) for i in range(cond_imgs.shape[0])
        ]
        # controls_pil = [
        #     v2.functional.to_pil_image(control_imgs[i, :3])
        #     for i in range(control_imgs.shape[0])
        # ]
        controls_pil = control_imgs

        outputs = []
        for cond_img, control_img, prompt in zip(
            images_pil, controls_pil, target_types
        ):
            control_kwargs = dict()
            control_kwargs[f"{self.control_type}_image"] = control_img

            latent = self.pipeline(
                cond_img,
                prompt=prompt,
                num_inference_steps=75,
                output_type="latent",
                **control_kwargs,
            ).images
            image = unscale_image(
                self.pipeline.vae.decode(
                    latent / self.pipeline.vae.config.scaling_factor, return_dict=False
                )[0]
            )  # [-1, 1]
            image = (image * 0.5 + 0.5).clamp(0, 1)
            outputs.append(image)
        outputs = torch.cat(outputs, dim=0).to(self.device)
        images = torch.cat([target_imgs, outputs], dim=-2)
        if self.control_type in ["normal", "depth"]:
            images = torch.cat([images, control_imgs], dim=-2)
        else:
            images = torch.cat(
                [images, control_imgs[:, :3], control_imgs[:, 3:]], dim=-2
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

        optimizer = torch.optim.AdamW(self.unet.controlnet.parameters(), lr=lr)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, 3000, eta_min=lr / 4
        # )

        # return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

    def to(self, *args, **kwargs):
        self.pipeline.to(*args, **kwargs)
        self.unet.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.pipeline(*args, **kwargs)

    @torch.no_grad()
    def sd_pipe_forward(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        sync_latent_end: float = 0.2,
        exp_start: float = 0.0,
        exp_end: float = 0.0,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        mesh = kwargs.pop("mesh", None)
        step_tex_kwargs = kwargs.pop("step_tex_kwargs", {})

        if mesh is not None:
            latent_tex = mesh.latent_img.data

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        height = (
            height or self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor
        )
        width = width or self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.pipe.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self.pipe._guidance_scale = guidance_scale
        self.pipe._guidance_rescale = guidance_rescale
        self.pipe._clip_skip = clip_skip
        self.pipe._cross_attention_kwargs = cross_attention_kwargs
        self.pipe._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.pipe._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.pipe.cross_attention_kwargs.get("scale", None)
            if self.pipe.cross_attention_kwargs is not None
            else None
        )

        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.pipe.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.pipe.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.pipe.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.pipe.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.pipe.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        if timesteps is None:
            timesteps, num_inference_steps = retrieve_timesteps(
                self.pipe.scheduler, num_inference_steps, device, timesteps, sigmas
            )
        else:
            num_inference_steps = len(timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.pipe.unet.config.in_channels
        latents_none = latents is None
        latents = self.pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        if not latents_none and timesteps is not None:
            latents /= self.pipe.scheduler.init_noise_sigma

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.pipe.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.pipe.guidance_scale - 1).repeat(
                batch_size * num_images_per_prompt
            )
            timestep_cond = self.pipe.get_guidance_scale_embedding(
                guidance_scale_tensor,
                embedding_dim=self.pipe.unet.config.time_cond_proj_dim,
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        num_warmup_steps = (
            len(timesteps) - num_inference_steps * self.pipe.scheduler.order
        )
        self.pipe._num_timesteps = len(timesteps)
        with self.pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.pipe.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.pipe.do_classifier_free_guidance
                    else latents
                )
                latent_model_input = self.pipe.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # predict the noise residual
                noise_pred = self.pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.pipe.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.pipe.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.pipe.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                if (
                    self.pipe.do_classifier_free_guidance
                    and self.pipe.guidance_rescale > 0.0
                ):
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.pipe.guidance_rescale,
                    )

                # compute the previous noisy sample x_t -> x_t-1
                if mesh is not None and t > int(sync_latent_end * self.num_timesteps):
                    print("using mesh-based denoising")
                    this_exp = (
                        exp_start + (exp_end - exp_start) * i / num_inference_steps
                    )
                    step_tex_kwargs["fusion_method"] = this_exp
                    step_results = step_tex(
                        self.pipe.scheduler,
                        mesh,
                        noise_pred,
                        t,
                        latents,
                        latent_tex,
                        return_dict=True,
                        background=self.bg_latent,
                        **step_tex_kwargs,
                    )
                    # pred_original_sample = step_results["pred_original_sample"]
                    latents = step_results["prev_sample"].type(self.pipe.dtype)
                    latent_tex = step_results["prev_tex"]

                    # latents = self.pipe.scheduler.step(
                    #     noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                    # )[0]
                    # _, mask = mesh.render_zero123pp_6views_latents(
                    #     4.5, (120, 120), "bilinear"
                    # )
                    # mask = mask[None]
                    # background = self.bg_latent
                    # if t > 0:
                    #     alphas_cumprod = self.pipe.scheduler.alphas_cumprod[t]
                    #     noise = torch.normal(
                    #         0, 1, background.shape, device=background.device
                    #     )
                    #     background = (
                    #         1 - alphas_cumprod
                    #     ) * noise + alphas_cumprod * background
                    # latents = latents * mask + background * (1 - mask)
                    # latents = latents.type(self.pipe.dtype)
                else:
                    latents = self.pipe.scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                    )[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(
                        self.pipe, i, t, callback_kwargs
                    )

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.pipe.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.pipe.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            image = self.pipe.vae.decode(
                latents / self.pipe.vae.config.scaling_factor,
                return_dict=False,
                generator=generator,
            )[0]
            image, has_nsfw_concept = self.pipe.run_safety_checker(
                image, device, prompt_embeds.dtype
            )
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.pipe.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        # Offload all models
        self.pipe.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        )

    @torch.no_grad()
    def forward_pipeline(
        self,
        image: Image.Image = None,
        prompt="",
        *args,
        num_images_per_prompt: Optional[int] = 1,
        guidance_scale=4.0,
        depth_image: Image.Image = None,
        normal_image: Image.Image = None,
        normal_depth_image: torch.Tensor = None,
        output_type: Optional[str] = "pil",
        width=640,
        height=960,
        num_inference_steps=28,
        return_dict=True,
        **kwargs,
    ):
        # self.pipe.prepare()
        # if image is None:
        #     raise ValueError(
        #         "Inputting embeddings not supported for this pipeline. Please pass an image."
        #     )
        # assert not isinstance(image, torch.Tensor)
        # image = to_rgb_image(image)
        # image_1 = self.pipe.feature_extractor_vae(
        #     images=image, return_tensors="pt"
        # ).pixel_values
        # image_2 = self.pipe.feature_extractor_clip(
        #     images=image, return_tensors="pt"
        # ).pixel_values
        # if depth_image is not None and hasattr(self.unet, "controlnet"):
        #     if not isinstance(depth_image, torch.Tensor):
        #         depth_image = to_rgb_image(depth_image)
        #         depth_image = self.pipe.depth_transforms_multi(depth_image).to(
        #             device=self.unet.controlnet.device, dtype=self.unet.controlnet.dtype
        #         )
        # if normal_image is not None and hasattr(self.pipe.unet, "controlnet"):
        #     normal_image = to_rgb_image(normal_image)
        #     normal_image = self.pipe.normal_transforms_multi(normal_image).to(
        #         device=self.pipe.unet.controlnet.device,
        #         dtype=self.pipe.unet.controlnet.dtype,
        #     )
        # if normal_depth_image is not None and hasattr(self.pipe.unet, "controlnet"):
        #     # normal_depth_image = to_rgb_image(normal_depth_image)
        #     # normal_depth_image = self.pipe.normal_depth_transforms_multi(
        #     #     normal_depth_image
        #     # ).to(device=self.pipe.unet.controlnet.device, dtype=self.pipe.unet.controlnet.dtype)
        #     if isinstance(normal_depth_image, np.ndarray):
        #         normal_depth_image = torch.from_numpy(normal_depth_image)
        #     normal_depth_image = normal_depth_image.to(
        #         device=self.pipe.unet.controlnet.device,
        #         dtype=self.pipe.unet.controlnet.dtype,
        #     )
        # if (
        #     normal_depth_image is None
        #     and hasattr(self.pipe.unet, "controlnet")
        #     and normal_image is not None
        #     and depth_image is not None
        # ):
        #     normal_depth_image = torch.cat([normal_image, depth_image], dim=0)
        # image = image_1.to(device=self.pipe.vae.device, dtype=self.pipe.vae.dtype)
        # image_2 = image_2.to(device=self.pipe.vae.device, dtype=self.pipe.vae.dtype)
        # cond_lat = self.pipe.encode_condition_image(image)
        # if guidance_scale > 1:
        #     negative_lat = self.pipe.encode_condition_image(torch.zeros_like(image))
        #     cond_lat = torch.cat([negative_lat, cond_lat])
        # encoded = self.pipe.vision_encoder(image_2, output_hidden_states=False)
        # global_embeds = encoded.image_embeds
        # global_embeds = global_embeds.unsqueeze(-2)

        # if hasattr(self.pipe, "encode_prompt"):
        #     encoder_hidden_states = self.pipe.encode_prompt(
        #         prompt, self.pipe.device, num_images_per_prompt, False
        #     )[0]
        # else:
        #     encoder_hidden_states = self.pipe._encode_prompt(
        #         prompt, self.pipe.device, num_images_per_prompt, False
        #     )
        # ramp = global_embeds.new_tensor(
        #     self.pipe.config.ramping_coefficients
        # ).unsqueeze(-1)
        # encoder_hidden_states = encoder_hidden_states + global_embeds * ramp
        # cak = dict(cond_lat=cond_lat)
        # if hasattr(self.pipe.unet, "controlnet") and depth_image is not None:
        #     cak["control_depth"] = depth_image
        # if hasattr(self.pipe.unet, "controlnet") and normal_image is not None:
        #     cak["control_normal"] = normal_image
        # if hasattr(self.pipe.unet, "controlnet") and normal_depth_image is not None:
        #     cak["control_normal_depth"] = normal_depth_image
        # latents: torch.Tensor = self.sd_pipe_forward(
        #     None,
        #     *args,
        #     cross_attention_kwargs=cak,
        #     guidance_scale=guidance_scale,
        #     num_images_per_prompt=num_images_per_prompt,
        #     prompt_embeds=encoder_hidden_states,
        #     num_inference_steps=num_inference_steps,
        #     output_type="latent",
        #     negative_prompt=prompt,
        #     width=width,
        #     height=height,
        #     **kwargs,
        # ).images
        # latents = unscale_latents(latents)
        # if not output_type == "latent":
        #     image = unscale_image(
        #         self.pipe.vae.decode(
        #             latents / self.pipe.vae.config.scaling_factor, return_dict=False
        #         )[0]
        #     )
        # else:
        #     image = latents

        # image = self.pipe.image_processor.postprocess(image, output_type=output_type)
        # if not return_dict:
        #     return (image,)

        # return ImagePipelineOutput(images=image)

        self.pipe.prepare()
        if image is None:
            raise ValueError(
                "Inputting embeddings not supported for this pipeline. Please pass an image."
            )
        assert not isinstance(image, torch.Tensor)
        image = to_rgb_image(image)
        image_1 = self.pipe.feature_extractor_vae(
            images=image, return_tensors="pt"
        ).pixel_values
        image_2 = self.pipe.feature_extractor_clip(
            images=image, return_tensors="pt"
        ).pixel_values
        if depth_image is not None and hasattr(self.pipe.unet, "controlnet"):
            if not isinstance(depth_image, torch.Tensor):
                depth_image = to_rgb_image(depth_image)
                depth_image = self.pipe.depth_transforms_multi(depth_image).to(
                    device=self.pipe.unet.controlnet.device,
                    dtype=self.pipe.unet.controlnet.dtype,
                )
        if normal_image is not None and hasattr(self.pipe.unet, "controlnet"):
            normal_image = to_rgb_image(normal_image)
            normal_image = self.pipe.normal_transforms_multi(normal_image).to(
                device=self.pipe.unet.controlnet.device,
                dtype=self.pipe.unet.controlnet.dtype,
            )
        if normal_depth_image is not None and hasattr(self.pipe.unet, "controlnet"):
            # normal_depth_image = to_rgb_image(normal_depth_image)
            # normal_depth_image = self.pipe.normal_depth_transforms_multi(
            #     normal_depth_image
            # ).to(device=self.pipe.unet.controlnet.device, dtype=self.pipe.unet.controlnet.dtype)
            if isinstance(normal_depth_image, np.ndarray):
                normal_depth_image = torch.from_numpy(normal_depth_image)
            normal_depth_image = normal_depth_image.to(
                device=self.pipe.unet.controlnet.device,
                dtype=self.pipe.unet.controlnet.dtype,
            )
        if (
            normal_depth_image is None
            and hasattr(self.pipe.unet, "controlnet")
            and normal_image is not None
            and depth_image is not None
        ):
            normal_depth_image = torch.cat([normal_image, depth_image], dim=0)
        image = image_1.to(device=self.pipe.vae.device, dtype=self.pipe.vae.dtype)
        image_2 = image_2.to(device=self.pipe.vae.device, dtype=self.pipe.vae.dtype)
        cond_lat = self.pipe.encode_condition_image(image)

        if guidance_scale > 1:
            negative_lat = self.pipe.encode_condition_image(torch.zeros_like(image))
            cond_lat = torch.cat([negative_lat, cond_lat])
            # note this thing
            # prompt = ["", prompt]
        encoded = self.pipe.vision_encoder(image_2, output_hidden_states=False)
        global_embeds = encoded.image_embeds
        global_embeds = global_embeds.unsqueeze(-2)

        if hasattr(self.pipe, "encode_prompt"):
            encoder_hidden_states = self.pipe.encode_prompt(
                prompt, self.pipe.device, num_images_per_prompt, False
            )[0]
        else:
            encoder_hidden_states = self.pipe._encode_prompt(
                prompt, self.pipe.device, num_images_per_prompt, False
            )
        ramp = global_embeds.new_tensor(
            self.pipe.config.ramping_coefficients
        ).unsqueeze(-1)
        encoder_hidden_states = encoder_hidden_states + global_embeds * ramp
        cak = dict(cond_lat=cond_lat)
        if hasattr(self.pipe.unet, "controlnet") and depth_image is not None:
            cak["control_depth"] = depth_image
        if hasattr(self.pipe.unet, "controlnet") and normal_image is not None:
            cak["control_normal"] = normal_image
        if hasattr(self.pipe.unet, "controlnet") and normal_depth_image is not None:
            cak["control_normal_depth"] = normal_depth_image

        latents: torch.Tensor = self.sd_pipe_forward(
            None,
            *args,
            cross_attention_kwargs=cak,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=encoder_hidden_states,
            num_inference_steps=num_inference_steps,
            negative_prompt=prompt,
            output_type="latent",
            width=width,
            height=height,
            **kwargs,
        ).images
        latents = unscale_latents(latents)
        if not output_type == "latent":
            image = unscale_image(
                self.pipe.vae.decode(
                    latents / self.pipe.vae.config.scaling_factor, return_dict=False
                )[0]
            )
        else:
            image = latents

        image = self.pipe.image_processor.postprocess(image, output_type=output_type)
        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

    @torch.no_grad()
    def inference_partial_noise(
        self,
        image,
        prompt,
        depth_image=None,
        normal_image=None,
        normal_depth_image=None,
        rendered_image=None,
        num_inference_steps=36,
        start_timestep_idx=-1,
        **kwargs,
    ):
        pipe = self.pipe
        vae = self.pipe.vae.eval()
        scheduler = self.pipe.scheduler
        ip_tensor = to_tensor(rendered_image)[None] * 2 - 1
        # ip_tensor is in [-1, 1]
        ip_tensor = scale_image(ip_tensor)
        posterior = vae.encode(ip_tensor.to(torch.float16).to("cuda")).latent_dist
        latents = posterior.sample() * vae.config.scaling_factor
        latents = scale_latents(latents)
        scheduler.set_timesteps(num_inference_steps, "cuda")
        timesteps = scheduler.timesteps
        noise = torch.randn_like(latents)

        if start_timestep_idx >= 0:
            timesteps = timesteps[start_timestep_idx:]
            noisy_latents = scheduler.add_noise(latents, noise, timesteps[0:1])
        else:
            noisy_latents = None

        # fix sigma and timestep corresponding AFTER add noise
        if hasattr(scheduler, "_init_step_index"):
            scheduler._init_step_index(timesteps[0])

        res_image = self.forward_pipeline(
            image,
            prompt,
            depth_image=depth_image,
            normal_image=normal_image,
            normal_depth_image=normal_depth_image,
            num_inference_steps=num_inference_steps,
            latents=noisy_latents,
            timesteps=timesteps,
            **kwargs,
        )

        scheduler.set_timesteps(num_inference_steps, "cuda")

        return res_image

    # @torch.no_grad()
    # def inference_partial_noise(self, *args, **kwargs):
    #     return self.forward_pipeline(*args, **kwargs)

    def inference_with_latent_mesh(
        self,
        mesh: TexturedMesh,
        radius: float,
        dilation_size: int,
        # original
        image,
        prompt,
        depth_image=None,
        normal_image=None,
        normal_depth_image=None,
        rendered_image=None,
        num_inference_steps=36,
        start_timestep_idx=-1,
        fusion_method="vanilla",
        render_reso=(40, 40),
        interpolation_mode="nearest",
        exp_start=0,
        exp_end=8,
        sync_latent_end=0.5,
        **kwargs,
    ):
        initial_all_latents = torch.randn(1, 4, 120, 80, device="cuda")
        mesh.backproject_zero123pp_6views_latents(
            initial_all_latents, radius, fusion_method=fusion_method
        )

        # render initial latent frames
        initial_mesh_latents, mask = mesh.render_zero123pp_6views_latents(
            radius, reso=render_reso, interpolation_mode=interpolation_mode
        )
        # write_image("trash/mesh_latent.png", initial_mesh_latents[:3], "chw")
        # write_image("trash/mesh_mask.png", mask[:3], "chw")
        # initial_bg_latents = torch.randn_like(initial_mesh_latents)

        # initial_latents = initial_all_latents * (1 - mask) + initial_mesh_latents * mask
        initial_latents = self.bg_latent * (1 - mask) + initial_mesh_latents * mask

        # write_image("trash/initial_latents.png", initial_latents[0, :3], "chw")
        # initial_all_latents = torch.randn(
        #     1, 4, 120, 80, device="cuda", dtype=torch.float16
        # )

        initial_latents = initial_latents.to(self.pipeline.dtype).to(
            self.pipeline.device
        )

        # def mv_joint_denoise_callback(pipeline, i, t, callback_kwargs):
        #     latents = callback_kwargs["latents"].to(torch.float32)
        #     if i < 45:
        #         mesh.backproject_zero123pp_6views_latents(latents, radius)
        #         new_latents, mask = mesh.render_zero123pp_6views_latents(radius)

        #         new_latents = latents * (1 - mask) + new_latents * mask
        #     else:
        #         new_latents = latents

        #     return {"latents": new_latents.to(pipeline.dtype)}

        res_image = self.forward_pipeline(
            image,
            prompt,
            depth_image=depth_image,
            normal_image=normal_image,
            normal_depth_image=normal_depth_image,
            num_inference_steps=num_inference_steps,
            # latents=initial_latents,
            # timesteps=timesteps,
            # callback_on_step_end=mv_joint_denoise_callback,
            mesh=mesh,
            step_tex_kwargs={
                "fusion_method": fusion_method,
                "render_reso": render_reso,
                "interpolation_mode": interpolation_mode,
            },
            exp_start=exp_start,
            exp_end=exp_end,
            sync_latent_end=sync_latent_end,
            **kwargs,
        )

        return res_image
