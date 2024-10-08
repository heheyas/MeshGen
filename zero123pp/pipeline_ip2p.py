import numpy as np
from typing import Any, Dict, Optional, List
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers

import numpy
import PIL
import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.distributed
import transformers
from collections import OrderedDict
from PIL import Image
from torchvision import transforms
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from torchvision.transforms.functional import to_pil_image

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    EulerAncestralDiscreteScheduler,
    UNet2DConditionModel,
    ImagePipelineOutput,
    StableDiffusionInstructPix2PixPipeline,
)
from diffusers.loaders import TextualInversionLoaderMixin
from diffusers.utils import deprecate
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_instruct_pix2pix import (
    retrieve_latents,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor,
    XFormersAttnProcessor,
    AttnProcessor2_0,
)
from diffusers.utils.import_utils import is_xformers_available


def to_rgb_image(maybe_rgba: Image.Image):
    if maybe_rgba.mode == "RGB":
        return maybe_rgba
    elif maybe_rgba.mode == "RGBA":
        rgba = maybe_rgba
        img = numpy.random.randint(
            255, 256, size=[rgba.size[1], rgba.size[0], 3], dtype=numpy.uint8
        )
        img = Image.fromarray(img, "RGB")
        img.paste(rgba, mask=rgba.getchannel("A"))
        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)


class ReferenceOnlyAttnProc(torch.nn.Module):
    def __init__(self, chained_proc, enabled=False, name=None) -> None:
        super().__init__()
        self.enabled = enabled
        self.chained_proc = chained_proc
        self.name = name

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        mode="w",
        ref_dict: dict = None,
        is_cfg_guidance=False,
    ) -> Any:
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        if self.enabled and is_cfg_guidance:
            res0 = self.chained_proc(
                attn, hidden_states[:1], encoder_hidden_states[:1], attention_mask
            )
            hidden_states = hidden_states[1:]
            encoder_hidden_states = encoder_hidden_states[1:]
        if self.enabled:
            if mode == "w":
                ref_dict[self.name] = encoder_hidden_states
            elif mode == "r":
                encoder_hidden_states = torch.cat(
                    [encoder_hidden_states, ref_dict.pop(self.name)], dim=1
                )
            elif mode == "m":
                encoder_hidden_states = torch.cat(
                    [encoder_hidden_states, ref_dict[self.name]], dim=1
                )
            else:
                assert False, mode
        res = self.chained_proc(
            attn, hidden_states, encoder_hidden_states, attention_mask
        )
        if self.enabled and is_cfg_guidance:
            res = torch.cat([res0, res])
        return res


class RefOnlyNoisedUNet(torch.nn.Module):
    def __init__(
        self,
        unet: UNet2DConditionModel,
        train_sched: DDPMScheduler,
        val_sched: EulerAncestralDiscreteScheduler,
    ) -> None:
        super().__init__()
        self.unet = unet
        self.train_sched = train_sched
        self.val_sched = val_sched

        unet_lora_attn_procs = dict()
        for name, _ in unet.attn_processors.items():
            if torch.__version__ >= "2.0":
                default_attn_proc = AttnProcessor2_0()
            elif is_xformers_available():
                default_attn_proc = XFormersAttnProcessor()
            else:
                default_attn_proc = AttnProcessor()
            unet_lora_attn_procs[name] = ReferenceOnlyAttnProc(
                default_attn_proc, enabled=name.endswith("attn1.processor"), name=name
            )
        unet.set_attn_processor(unet_lora_attn_procs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)

    def forward_cond(
        self,
        noisy_cond_lat,
        timestep,
        encoder_hidden_states,
        class_labels,
        ref_dict,
        is_cfg_guidance,
        **kwargs,
    ):
        if is_cfg_guidance:
            encoder_hidden_states = encoder_hidden_states[1:]
            class_labels = class_labels[1:]
        self.unet(
            noisy_cond_lat,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels,
            cross_attention_kwargs=dict(mode="w", ref_dict=ref_dict),
            **kwargs,
        )

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        class_labels=None,
        *args,
        cross_attention_kwargs,
        down_block_res_samples=None,
        mid_block_res_sample=None,
        noise=None,
        **kwargs,
    ):
        cond_lat = cross_attention_kwargs["cond_lat"]
        is_cfg_guidance = cross_attention_kwargs.get("is_cfg_guidance", False)
        if noise is None:
            noise = torch.randn_like(cond_lat)
        if self.training:
            noisy_cond_lat = self.train_sched.add_noise(cond_lat, noise, timestep)
            noisy_cond_lat = self.train_sched.scale_model_input(
                noisy_cond_lat, timestep
            )
        else:
            noisy_cond_lat = self.val_sched.add_noise(
                cond_lat, noise, timestep.reshape(-1)
            )
            noisy_cond_lat = self.val_sched.scale_model_input(
                noisy_cond_lat, timestep.reshape(-1)
            )
        ref_dict = {}
        self.forward_cond(
            noisy_cond_lat,
            timestep,
            encoder_hidden_states,
            class_labels,
            ref_dict,
            is_cfg_guidance,
            **kwargs,
        )
        weight_dtype = self.unet.dtype
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states,
            *args,
            class_labels=class_labels,
            cross_attention_kwargs=dict(
                mode="r", ref_dict=ref_dict, is_cfg_guidance=is_cfg_guidance
            ),
            down_block_additional_residuals=(
                [sample.to(dtype=weight_dtype) for sample in down_block_res_samples]
                if down_block_res_samples is not None
                else None
            ),
            mid_block_additional_residual=(
                mid_block_res_sample.to(dtype=weight_dtype)
                if mid_block_res_sample is not None
                else None
            ),
            **kwargs,
        )


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


class NormalDepthControlUNet(torch.nn.Module):
    def __init__(
        self,
        unet: RefOnlyNoisedUNet,
        controlnet: Optional[diffusers.ControlNetModel] = None,
        conditioning_scale=1.0,
    ) -> None:
        super().__init__()
        self.unet = unet
        if controlnet is None:
            self.controlnet = diffusers.ControlNetModel.from_unet(
                unet.unet, conditioning_channels=6
            )
        else:
            self.controlnet = controlnet
        DefaultAttnProc = AttnProcessor2_0
        if is_xformers_available():
            DefaultAttnProc = XFormersAttnProcessor
        self.controlnet.set_attn_processor(DefaultAttnProc())
        self.conditioning_scale = conditioning_scale

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        class_labels=None,
        *args,
        cross_attention_kwargs: dict,
        **kwargs,
    ):
        cross_attention_kwargs = dict(cross_attention_kwargs)
        control_depth = cross_attention_kwargs.pop("control_normal_depth")
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=control_depth,
            conditioning_scale=self.conditioning_scale,
            return_dict=False,
        )
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_res_samples=down_block_res_samples,
            mid_block_res_sample=mid_block_res_sample,
            cross_attention_kwargs=cross_attention_kwargs,
        )


class DepthControlUNet(torch.nn.Module):
    def __init__(
        self,
        unet: RefOnlyNoisedUNet,
        controlnet: Optional[diffusers.ControlNetModel] = None,
        conditioning_scale=1.0,
    ) -> None:
        super().__init__()
        self.unet = unet
        if controlnet is None:
            self.controlnet = diffusers.ControlNetModel.from_unet(unet.unet)
        else:
            self.controlnet = controlnet
        DefaultAttnProc = AttnProcessor2_0
        if is_xformers_available():
            DefaultAttnProc = XFormersAttnProcessor
        self.controlnet.set_attn_processor(DefaultAttnProc())
        self.conditioning_scale = conditioning_scale

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        class_labels=None,
        *args,
        cross_attention_kwargs: dict,
        **kwargs,
    ):
        cross_attention_kwargs = dict(cross_attention_kwargs)
        control_depth = cross_attention_kwargs.pop("control_depth")
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=control_depth,
            conditioning_scale=self.conditioning_scale,
            return_dict=False,
        )
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_res_samples=down_block_res_samples,
            mid_block_res_sample=mid_block_res_sample,
            cross_attention_kwargs=cross_attention_kwargs,
        )


class NormalControlUNet(torch.nn.Module):
    def __init__(
        self,
        unet: RefOnlyNoisedUNet,
        controlnet: Optional[diffusers.ControlNetModel] = None,
        conditioning_scale=1.0,
    ) -> None:
        super().__init__()
        self.unet = unet
        if controlnet is None:
            self.controlnet = diffusers.ControlNetModel.from_unet(unet.unet)
        else:
            self.controlnet = controlnet
        DefaultAttnProc = AttnProcessor2_0
        if is_xformers_available():
            DefaultAttnProc = XFormersAttnProcessor
        self.controlnet.set_attn_processor(DefaultAttnProc())
        self.conditioning_scale = conditioning_scale

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        class_labels=None,
        *args,
        cross_attention_kwargs: dict,
        **kwargs,
    ):
        cross_attention_kwargs = dict(cross_attention_kwargs)
        control_depth = cross_attention_kwargs.pop("control_normal")
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=control_depth,
            conditioning_scale=self.conditioning_scale,
            return_dict=False,
        )
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_res_samples=down_block_res_samples,
            mid_block_res_sample=mid_block_res_sample,
            cross_attention_kwargs=cross_attention_kwargs,
        )


class ModuleListDict(torch.nn.Module):
    def __init__(self, procs: dict) -> None:
        super().__init__()
        self.keys = sorted(procs.keys())
        self.values = torch.nn.ModuleList(procs[k] for k in self.keys)

    def __getitem__(self, key):
        return self.values[self.keys.index(key)]


class SuperNet(torch.nn.Module):
    def __init__(self, state_dict: Dict[str, torch.Tensor]):
        super().__init__()
        state_dict = OrderedDict((k, state_dict[k]) for k in sorted(state_dict.keys()))
        self.layers = torch.nn.ModuleList(state_dict.values())
        self.mapping = dict(enumerate(state_dict.keys()))
        self.rev_mapping = {v: k for k, v in enumerate(state_dict.keys())}

        # .processor for unet, .self_attn for text encoder
        self.split_keys = [".processor", ".self_attn"]

        # we add a hook to state_dict() and load_state_dict() so that the
        # naming fits with `unet.attn_processors`
        def map_to(module, state_dict, *args, **kwargs):
            new_state_dict = {}
            for key, value in state_dict.items():
                num = int(key.split(".")[1])  # 0 is always "layers"
                new_key = key.replace(f"layers.{num}", module.mapping[num])
                new_state_dict[new_key] = value

            return new_state_dict

        def remap_key(key, state_dict):
            for k in self.split_keys:
                if k in key:
                    return key.split(k)[0] + k
            return key.split(".")[0]

        def map_from(module, state_dict, *args, **kwargs):
            all_keys = list(state_dict.keys())
            for key in all_keys:
                replace_key = remap_key(key, state_dict)
                new_key = key.replace(
                    replace_key, f"layers.{module.rev_mapping[replace_key]}"
                )
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        self._register_state_dict_hook(map_to)
        self._register_load_state_dict_pre_hook(map_from, with_module=True)


class Zero123PlusIp2pPipeline(StableDiffusionInstructPix2PixPipeline):
    tokenizer: transformers.CLIPTokenizer
    text_encoder: transformers.CLIPTextModel
    vision_encoder: transformers.CLIPVisionModelWithProjection

    feature_extractor_clip: transformers.CLIPImageProcessor
    unet: UNet2DConditionModel
    scheduler: diffusers.schedulers.KarrasDiffusionSchedulers

    vae: AutoencoderKL
    ramping: nn.Linear

    feature_extractor_vae: transformers.CLIPImageProcessor

    depth_transforms_multi = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    normal_transforms_multi = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    normal_depth_transforms_multi = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        vision_encoder: transformers.CLIPVisionModelWithProjection,
        feature_extractor_clip: CLIPImageProcessor,
        feature_extractor_vae: CLIPImageProcessor,
        ramping_coefficients: Optional[list] = None,
        safety_checker=None,
    ):
        DiffusionPipeline.__init__(self)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            vision_encoder=vision_encoder,
            feature_extractor_clip=feature_extractor_clip,
            feature_extractor_vae=feature_extractor_vae,
        )
        self.register_to_config(ramping_coefficients=ramping_coefficients)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def prepare(self):
        train_sched = DDPMScheduler.from_config(self.scheduler.config)
        if isinstance(self.unet, UNet2DConditionModel):
            self.unet = RefOnlyNoisedUNet(self.unet, train_sched, self.scheduler).eval()

    def add_controlnet(
        self,
        controlnet: Optional[diffusers.ControlNetModel] = None,
        conditioning_scale=1.0,
    ):
        self.prepare()
        self.unet = DepthControlUNet(self.unet, controlnet, conditioning_scale)
        return SuperNet(OrderedDict([("controlnet", self.unet.controlnet)]))

    def encode_condition_image(self, image: torch.Tensor):
        image = self.vae.encode(image).latent_dist.sample()
        return image

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image = None,
        prompt="",
        *args,
        num_images_per_prompt: Optional[int] = 1,
        guidance_scale=4.0,
        image_guidance_scale=1.0,
        depth_image: Image.Image = None,
        normal_image: Image.Image = None,
        normal_depth_image: torch.Tensor = None,
        output_type: Optional[str] = "pil",
        width=640,
        height=960,
        num_inference_steps=28,
        return_dict=True,
        ip2p_cond=None,
        **kwargs,
    ):
        self.prepare()
        if image is None:
            raise ValueError(
                "Inputting embeddings not supported for this pipeline. Please pass an image."
            )
        assert not isinstance(image, torch.Tensor)
        image = to_rgb_image(image)
        image_1 = self.feature_extractor_vae(
            images=image, return_tensors="pt"
        ).pixel_values
        image_2 = self.feature_extractor_clip(
            images=image, return_tensors="pt"
        ).pixel_values
        if depth_image is not None and hasattr(self.unet, "controlnet"):
            if not isinstance(depth_image, torch.Tensor):
                depth_image = to_rgb_image(depth_image)
                depth_image = self.depth_transforms_multi(depth_image).to(
                    device=self.unet.controlnet.device, dtype=self.unet.controlnet.dtype
                )
        if normal_image is not None and hasattr(self.unet, "controlnet"):
            normal_image = to_rgb_image(normal_image)
            normal_image = self.normal_transforms_multi(normal_image).to(
                device=self.unet.controlnet.device, dtype=self.unet.controlnet.dtype
            )
        if normal_depth_image is not None and hasattr(self.unet, "controlnet"):
            # normal_depth_image = to_rgb_image(normal_depth_image)
            # normal_depth_image = self.normal_depth_transforms_multi(
            #     normal_depth_image
            # ).to(device=self.unet.controlnet.device, dtype=self.unet.controlnet.dtype)
            if isinstance(normal_depth_image, np.ndarray):
                normal_depth_image = torch.from_numpy(normal_depth_image)
            normal_depth_image = normal_depth_image.to(
                device=self.unet.controlnet.device, dtype=self.unet.controlnet.dtype
            )
        if (
            normal_depth_image is None
            and hasattr(self.unet, "controlnet")
            and normal_image is not None
            and depth_image is not None
        ):
            normal_depth_image = torch.cat([normal_image, depth_image], dim=0)
        image = image_1.to(device=self.vae.device, dtype=self.vae.dtype)
        image_2 = image_2.to(device=self.vae.device, dtype=self.vae.dtype)
        cond_lat = self.encode_condition_image(image)

        if ip2p_cond is not None:
            # if isinstance(ip2p_cond, Image.Image):
            #     ip2p_cond = to_rgb_image(ip2p_cond)
            # elif isinstance(ip2p_cond, torch.Tensor):
            #     ip2p_cond = to_pil_image(ip2p_cond)
            # ip2p_cond_th = self.feature_extractor_vae(
            #     images=ip2p_cond, return_tensors="pt"
            # ).pixel_values
            # ip2p_cond_th = self.encode_condition_image(
            #     ip2p_cond_th.to(device=self.vae.device, dtype=self.vae.dtype)
            # )
            cond_lat = torch.cat([cond_lat, torch.zeros_like(cond_lat)], dim=1)

        if guidance_scale > 1:
            negative_lat = self.encode_condition_image(torch.zeros_like(image))
            if ip2p_cond is not None:
                negative_lat = torch.cat(
                    [negative_lat, torch.zeros_like(negative_lat)], dim=1
                )
        ## to align with the ip2p pipeline
        # breakpoint()
        cond_lat = torch.cat([cond_lat, cond_lat, negative_lat])

        encoded = self.vision_encoder(image_2, output_hidden_states=False)
        global_embeds = encoded.image_embeds
        global_embeds = global_embeds.unsqueeze(-2)

        if hasattr(self, "encode_prompt"):
            encoder_hidden_states = self.encode_prompt(
                prompt, self.device, num_images_per_prompt, False
            )[0]
        else:
            encoder_hidden_states = self._encode_prompt(
                prompt, self.device, num_images_per_prompt, False
            )
        ramp = global_embeds.new_tensor(self.config.ramping_coefficients).unsqueeze(-1)
        encoder_hidden_states = encoder_hidden_states + global_embeds * ramp
        cak = dict(cond_lat=cond_lat)
        if hasattr(self.unet, "controlnet") and depth_image is not None:
            cak["control_depth"] = depth_image
        if hasattr(self.unet, "controlnet") and normal_image is not None:
            cak["control_normal"] = normal_image
        if hasattr(self.unet, "controlnet") and normal_depth_image is not None:
            cak["control_normal_depth"] = normal_depth_image
        latents: torch.Tensor = (
            super()
            .__call__(
                None,
                *args,
                image=ip2p_cond,
                cross_attention_kwargs=cak,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                prompt_embeds=encoder_hidden_states,
                num_inference_steps=num_inference_steps,
                negative_prompt=prompt,
                output_type="latent",
                width=width,
                height=height,
                image_guidance_scale=image_guidance_scale,
                **kwargs,
            )
            .images
        )
        latents = unscale_latents(latents)
        if not output_type == "latent":
            image = unscale_image(
                self.vae.decode(
                    latents / self.vae.config.scaling_factor, return_dict=False
                )[0]
            )
        else:
            image = latents

        image = self.image_processor.postprocess(image, output_type=output_type)
        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

    @torch.no_grad()
    def encode_target_images(self, images):
        dtype = next(self.vae.parameters()).dtype
        # equals to scaling images to [-1, 1] first and then call scale_image
        images = (images - 0.5) / 0.8  # [-0.625, 0.625]
        posterior = self.vae.encode(images.to(dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        latents = scale_latents(latents)
        return latents

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_ prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1
            ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                # logger.warning(
                #     "The following part of your input was truncated because CLIP can only handle sequences up to"
                #     f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                # )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device), attention_mask=attention_mask
            )
            prompt_embeds = prompt_embeds[0]

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        else:
            prompt_embeds_dtype = self.unet.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=prompt_embeds_dtype, device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            # pix2pix has two negative embeddings, and unlike in other pipelines latents are ordered [prompt_embeds, negative_prompt_embeds, negative_prompt_embeds]
            prompt_embeds = torch.cat(
                [prompt_embeds, negative_prompt_embeds, negative_prompt_embeds]
            )

        return prompt_embeds

    def prepare_image_latents(
        self,
        image,
        batch_size,
        num_images_per_prompt,
        dtype,
        device,
        do_classifier_free_guidance,
        generator=None,
    ):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            image_latents = image
        else:
            # image_latents = retrieve_latents(
            #     self.vae.encode(image), sample_mode="argmax"
            # )
            image = image * 0.5 + 0.5
            image_latents = self.encode_target_images(image)

        if (
            batch_size > image_latents.shape[0]
            and batch_size % image_latents.shape[0] == 0
        ):
            # expand image_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {image_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate(
                "len(prompt) != len(image)",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat(
                [image_latents] * additional_image_per_prompt, dim=0
            )
        elif (
            batch_size > image_latents.shape[0]
            and batch_size % image_latents.shape[0] != 0
        ):
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = torch.cat([image_latents], dim=0)

        if do_classifier_free_guidance:
            uncond_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat(
                [image_latents, image_latents, uncond_image_latents], dim=0
            )

        return image_latents
