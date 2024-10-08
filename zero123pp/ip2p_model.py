import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from torchvision.transforms import v2
from torchvision.utils import make_grid, save_image
from einops import rearrange
from PIL import Image

from meshgen.util import instantiate_from_config
from diffusers import (
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
)
from .pipeline_ip2p import (
    RefOnlyNoisedUNet,
    DepthControlUNet,
    NormalDepthControlUNet,
    NormalControlUNet,
)


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


class MVIp2p(pl.LightningModule):
    def __init__(
        self,
        stable_diffusion_config,
        drop_cond_prob=0.05,
        in_channels=8,
        unet_ckpt_path=None,
        ckpt_path=None,
    ):
        super(MVIp2p, self).__init__()

        self.drop_cond_prob = drop_cond_prob

        self.register_schedule()

        # init modules
        pipeline = DiffusionPipeline.from_pretrained(**stable_diffusion_config)
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing="trailing"
        )
        self.pipeline = pipeline

        train_sched = DDPMScheduler.from_config(self.pipeline.scheduler.config)

        if unet_ckpt_path is not None:
            sd = torch.load(unet_ckpt_path, map_location="cpu")
            if "state_dict" in sd:
                sd = sd["state_dict"]
            pipeline.unet.load_state_dict(sd)

        self.unet = pipeline.unet
        out_channels = self.unet.conv_in.out_channels
        self.unet.register_to_config(in_channels=in_channels)

        with torch.no_grad():
            new_conv_in = nn.Conv2d(
                in_channels,
                out_channels,
                self.unet.conv_in.kernel_size,
                self.unet.conv_in.stride,
                self.unet.conv_in.padding,
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(self.unet.conv_in.weight)
            new_conv_in.bias.copy_(self.unet.conv_in.bias)

            self.unet.conv_in = new_conv_in

        if isinstance(self.pipeline.unet, UNet2DConditionModel):
            self.pipeline.unet = RefOnlyNoisedUNet(
                self.pipeline.unet, train_sched, self.pipeline.scheduler
            )

        self.train_scheduler = train_sched  # use ddpm scheduler during training

        self.unet = pipeline.unet

        # ## debug
        # ori_pipeline = DiffusionPipeline.from_pretrained(**stable_diffusion_config)
        # ori_pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        #     ori_pipeline.scheduler.config, timestep_spacing="trailing"
        # )
        # self.ori_pipeline = ori_pipeline

        # if unet_ckpt_path is not None:
        #     sd = torch.load(unet_ckpt_path, map_location="cpu")
        #     if "state_dict" in sd:
        #         sd = sd["state_dict"]
        #     ori_pipeline.unet.load_state_dict(sd)

        # if isinstance(self.ori_pipeline.unet, UNet2DConditionModel):
        #     self.ori_pipeline.unet = RefOnlyNoisedUNet(
        #         self.ori_pipeline.unet, train_sched, self.ori_pipeline.scheduler
        #     )

        # self.train_scheduler = train_sched  # use ddpm scheduler during training
        # self.ori_unet = ori_pipeline.unet
        # ## debug

        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location="cpu")
            if "state_dict" in sd:
                sd = sd["state_dict"]
            self.load_state_dict(sd)

        # validation output buffer
        self.validation_step_outputs = []

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

        control_imgs = batch["ip2p_cond"]
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

    def forward_unet(self, latents, t, prompt_embeds, cond_latents, **kwargs):
        dtype = next(self.pipeline.unet.parameters()).dtype
        latents = latents.to(dtype)
        prompt_embeds = prompt_embeds.to(dtype)
        cond_latents = cond_latents.to(dtype)
        cross_attention_kwargs = dict(cond_lat=cond_latents)
        pred_noise = self.pipeline.unet(
            latents,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
            **kwargs,
        )[0]
        return pred_noise

    def forward_ori_unet(self, latents, t, prompt_embeds, cond_latents, **kwargs):
        dtype = next(self.ori_pipeline.unet.parameters()).dtype
        latents = latents.to(dtype)
        prompt_embeds = prompt_embeds.to(dtype)
        cond_latents = cond_latents.to(dtype)
        cross_attention_kwargs = dict(cond_lat=cond_latents)
        pred_noise = self.ori_pipeline.unet(
            latents,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
            **kwargs,
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
        cfg_rand = np.random.rand()

        def _prompt_encode(types):
            if hasattr(self.pipeline, "encode_prompt"):
                prompt_embeds = self.pipeline.encode_prompt(
                    types, self.device, 1, False
                )[0]
            else:
                prompt_embeds = self.pipeline._encode_prompt(
                    types, self.device, 1, False
                )
            return prompt_embeds

        if cfg_rand < self.drop_cond_prob:
            # drop text
            prompt_embeds = _prompt_encode(target_types)
            cond_latents = self.encode_condition_image(torch.zeros_like(cond_imgs))
            control_latents = self.encode_target_images(control_imgs)
        elif cfg_rand > self.drop_cond_prob and cfg_rand < 2 * self.drop_cond_prob:
            # drop image
            prompt_embeds = self.forward_vision_encoder(cond_imgs, target_types)
            cond_latents = self.encode_condition_image(cond_imgs)
            control_latents = self.encode_target_images(torch.zeros_like(control_imgs))
        elif cfg_rand > 2 * self.drop_cond_prob and cfg_rand < 3 * self.drop_cond_prob:
            # drop all
            prompt_embeds = _prompt_encode(target_types)
            cond_latents = self.encode_condition_image(torch.zeros_like(cond_imgs))
            control_latents = self.encode_target_images(torch.zeros_like(control_imgs))
        else:
            prompt_embeds = self.forward_vision_encoder(cond_imgs, target_types)
            cond_latents = self.encode_condition_image(cond_imgs)
            control_latents = self.encode_target_images(control_imgs)

        latents = self.encode_target_images(target_imgs)
        noise = torch.randn_like(latents)
        latents_noisy = self.train_scheduler.add_noise(latents, noise, t)
        latents_noisy = torch.cat([latents_noisy, control_latents], dim=1)
        cond_latents = torch.cat([cond_latents, torch.zeros_like(cond_latents)], dim=1)

        # ## debug
        # noise_ref = torch.randn_like(cond_latents)
        # output_0 = self.forward_unet(
        #     latents_noisy,
        #     t,
        #     prompt_embeds,
        #     cond_latents,
        #     noise=noise_ref,
        # )
        # output_1 = self.forward_ori_unet(
        #     latents_noisy[:, :4],
        #     t,
        #     prompt_embeds,
        #     cond_latents[:, :4],
        #     noise=noise_ref[:, :4],
        # )
        # breakpoint()

        # ## debug
        # latents_noisy_0 = torch.cat(
        #     [latents_noisy_original, torch.zeros_like(latents_noisy_original)], dim=1
        # )
        # latents_noisy_1 = torch.cat(
        #     [latents_noisy_original, torch.ones_like(latents_noisy_original)], dim=1
        # )
        # with torch.no_grad():
        #     noise = torch.randn_like(cond_latents)
        #     v_pred_0 = self.forward_unet(
        #         latents_noisy_0, t, prompt_embeds, cond_latents, noise=noise
        #     )
        #     v_pred_1 = self.forward_unet(
        #         latents_noisy_1, t, prompt_embeds, cond_latents, noise=noise
        #     )
        #     breakpoint()
        # ## debug

        v_pred = self.forward_unet(latents_noisy, t, prompt_embeds, cond_latents)
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
                latents_pred = self.predict_start_from_z_and_v(
                    latents_noisy[:, :4], t, v_pred
                )

                latents = unscale_latents(latents_pred)
                images = unscale_image(
                    self.pipeline.vae.decode(
                        latents / self.pipeline.vae.config.scaling_factor,
                        return_dict=False,
                    )[0]
                )  # [-1, 1]
                images = (images * 0.5 + 0.5).clamp(0, 1)

                images = torch.cat([target_imgs, images], dim=-2)
                images = torch.cat([images, control_imgs], dim=-2)

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
        controls_pil = [
            v2.functional.to_pil_image(control_imgs[i])
            for i in range(control_imgs.shape[0])
        ]

        outputs = []
        for cond_img, control_img, prompt in zip(
            images_pil, controls_pil, target_types
        ):
            prompt = ""

            latent = self.pipeline(
                cond_img,
                prompt=prompt,
                num_inference_steps=75,
                image_guidance_scale=1.0,
                output_type="latent",
                ip2p_cond=control_img,
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
        images = torch.cat([images, control_imgs], dim=-2)

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

        optimizer = torch.optim.AdamW(self.unet.parameters(), lr=lr)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, 3000, eta_min=lr / 4
        # )

        # return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

    def forward(self, **kwargs):
        return self.pipeline(**kwargs)

    def to(self, *args, **kwargs):
        self.pipeline.to(*args, **kwargs)
        self.unet.to(*args, **kwargs)
        return super().to(*args, **kwargs)
