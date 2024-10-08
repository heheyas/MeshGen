import gc
import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from pathlib import Path
import tqdm
import cumcubes
from copy import deepcopy
from einops import rearrange, repeat
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.transforms import v2
from contextlib import contextmanager
from torchvision.utils import make_grid, save_image
import lpips

from meshgen.util import instantiate_from_config
from meshgen.model.base import BaseModel
from meshgen.utils.ema import LitEma
from meshgen.utils.io import load_mesh, write_video
from meshgen.utils.render import render_mesh_spiral_offscreen
from meshgen.utils.ops import logit_normal


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class RectifiedFlowUNet2D(BaseModel):
    def __init__(
        self,
        autoencoder,
        unet,
        cond_encoder,
        input_key="surface",
        cond_key="images",
        shift_factor=0.0,
        scale_factor=0.25,
        weight_decay=0.0,
        ckpt_path=None,
        ignore_keys=[],
        scheduler_config=None,
        sample_kwargs={},
        render_kwargs={},
        vis_every=None,
        rf_mu=1.0986,
        rf_sigma=1.0,
        timestep_sample="uniform",
        rescale_image=False,
        use_ema=False,
        skip_validation=False,
        force_reinit_ema=False,
        _no_load_ckpt=False,
        *args,
        **kwargs,
    ):
        # TODO: add ema model
        super().__init__(*args, **kwargs)

        self.input_key = input_key
        self.cond_key = cond_key

        self.autoencoder = instantiate_from_config(autoencoder)
        self.autoencoder = self.autoencoder.eval()
        self.autoencoder.train = disabled_train
        for n, p in self.autoencoder.named_parameters():
            p.requires_grad = False

        self.unet = instantiate_from_config(unet)
        self.cond_encoder = instantiate_from_config(cond_encoder)

        self.force_reinit_ema = force_reinit_ema
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.unet)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None and not _no_load_ckpt:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.skip_validation = skip_validation
        self.shift_factor = shift_factor
        self.scale_factor = scale_factor
        self.scheduler_config = scheduler_config
        self.use_scheduler = scheduler_config is not None
        self.weight_decay = weight_decay

        self.sample_kwargs = sample_kwargs
        self.render_kwargs = render_kwargs
        self.vis_every = vis_every

        self.rf_mu = rf_mu
        self.rf_sigma = rf_sigma
        self.timestep_sample = timestep_sample

        self.latent_shape = (
            self.autoencoder.triplane_ch,
            self.autoencoder.triplane_res * 3,
            self.autoencoder.triplane_res,
        )

        self.rescale_image = rescale_image

        self.train()

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.unet)

    def on_load_checkpoint(self, checkpoint):
        if self.use_ema:
            contain_ema = False
            for k in checkpoint["state_dict"]:
                if "model_ema" in k:
                    contain_ema = True
                    break
            if not contain_ema or self.force_reinit_ema:
                ema_sd = {}
                for k, v in self.unet.state_dict().items():
                    ema_sd[f"model_ema.{k.replace('.', '')}"] = v
                ema_sd["model_ema.num_updates"] = torch.tensor(0, dtype=torch.int)
                ema_sd["model_ema.decay"] = torch.tensor(0.9999, dtype=torch.float32)
                checkpoint["state_dict"].update(ema_sd)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.unet.parameters())
            self.model_ema.copy_to(self.unet)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.unet.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    @torch.no_grad()
    def encode_shape(self, pcd):
        z = self.autoencoder.encode(pcd)
        z = rearrange(z, "b n c h w -> b c (n h) w")

        return (z + self.shift_factor) * self.scale_factor

    @torch.no_grad()
    def decode_shape(self, z, **kwargs):
        z = z / self.scale_factor - self.shift_factor
        return self.autoencoder.decode_shape(z, **kwargs)

    def decode(self, z, queries=None):
        z = z / self.scale_factor - self.shift_factor
        return self.autoencoder.decode(z, queries, upsample=True)

    @torch.no_grad()
    def encode_cond(self, cond):
        return self.cond_encoder(cond)

    def forward(self, input, cond):
        X0 = self.encode_shape(input)
        cond_emb = self.encode_cond(cond)

        # t_input = logit_normal(
        #     self.rf_mu, self.rf_sigma, (bs,), device=self.device, dtype=self.dtype
        # )
        bs = X0.shape[0]
        if self.timestep_sample == "uniform":
            t_input = torch.rand((bs,), device=self.device, dtype=self.dtype)
        else:
            t_input = logit_normal(
                self.rf_mu, self.rf_sigma, (bs,), device=self.device, dtype=self.dtype
            )

        t = t_input.view(bs, *((1,) * (len(X0.shape) - 1)))

        X1 = torch.randn_like(X0)
        Xt = X1 * t + X0 * (1 - t)

        pred = self.unet(Xt, cond_emb, t_input)

        loss = F.mse_loss((X1 - X0), pred)

        return loss

    def shared_step(self, batch, batch_idx):
        input = batch[self.input_key]
        cond = batch[self.cond_key]

        loss = self(input, cond)
        loss_dict = {"loss": loss}

        return loss, loss_dict

    @torch.no_grad()
    def sample_one(
        self,
        cond,
        uncond,
        cfg,
        n_steps,
        n_samples=1,
        seed=1234,
        x_init=None,
        timestep_callback=None,
    ):
        generator = torch.Generator(self.device).manual_seed(seed + self.local_rank)
        if x_init is None:
            x_init = torch.randn(
                (n_samples,) + self.latent_shape,
                device=self.device,
                generator=generator,
                dtype=self.dtype,
            )
        ts = [i / n_steps for i in range(n_steps + 1)]
        if timestep_callback is not None:
            print("Using timestep callback")
            # ts = [timestep_callback(t) for t in ts]
            ts = timestep_callback(ts)

        cond_emb = self.encode_cond(cond)
        uncond_emb = self.encode_cond(uncond)
        cond_emb = repeat(cond_emb, "1 ... -> n ...", n=n_samples)
        uncond_emb = repeat(uncond_emb, "1 ... -> n ...", n=n_samples)

        x = x_init
        for s, t in tqdm.tqdm(list(zip(ts, ts[1:]))[::-1], disable=True):
            # pred = nnet(x, t=torch.full((x.size(0),), t).to(x))
            this_t = torch.full((x.size(0),), t).to(x)

            cond_pred = self.unet(x, cond_emb, this_t)
            uncond_pred = self.unet(x, uncond_emb, this_t)
            pred = cond_pred + (cond_pred - uncond_pred) * cfg

            x = x + pred * (s - t)

        return x

    @torch.no_grad()
    def test_in_the_wild(self, save_dir):
        if self.skip_validation:
            return
        torch.cuda.empty_cache()
        gc.collect()
        if self.cond_key == "images":
            image_dir = Path("./data/images/GSO")
            images = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
            images = sorted(images)[self.local_rank :: self.trainer.world_size]

            for image in images:
                cond = np.array(Image.open(image))
                if self.rescale_image:
                    cond = cond.astype(np.float32) / 255.0
                uncond = np.zeros_like(cond)

                denosied = self.sample_one(
                    cond, uncond, n_samples=1, **self.sample_kwargs
                )
                try:
                    v, f = self.decode_shape(denosied)
                    frames = render_mesh_spiral_offscreen(v, f, **self.render_kwargs)

                    write_video(save_dir / f"{image.stem}.mp4", frames)
                except IndexError:
                    pass
        elif self.cond_key == "text":
            with open("data/texts/benchmark_captions.txt") as f:
                text_prompts = f.read().strip().split("\n")
            text_prompts = text_prompts[:32]
            text_prompts = sorted(text_prompts)[
                self.local_rank :: self.trainer.world_size
            ]
            for text in text_prompts:
                denosied = self.sample_one(text, "", n_samples=1, **self.sample_kwargs)
                try:
                    v, f = self.decode_shape(denosied)
                    frames = render_mesh_spiral_offscreen(v, f, **self.render_kwargs)

                    caption = text.replace(" ", "_")[:30]
                    write_video(save_dir / f"{caption}.mp4", frames)
                except IndexError:
                    pass
        else:
            raise NotImplementedError

    def get_optim_groups(self):
        return self.unet.parameters()
