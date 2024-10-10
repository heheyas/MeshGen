import shutil
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import default_collate
from pathlib import Path
from einops import rearrange, repeat
import tqdm
import nvdiffrast.torch as dr
# from diso import DiffDMC
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from skimage import measure

from meshgen.util import instantiate_from_config
from diffusers.training_utils import EMAModel
from meshgen.utils.io import load_mesh, write_video, write_image, export_mesh
from meshgen.utils.render import render_mesh_spiral_offscreen
from meshgen.utils.ops import (
    sample_on_surface,
    sample_from_planes,
    generate_planes,
    calc_normal,
    get_projection_matrix,
    safe_normalize,
    compute_tv_loss,
)
from meshgen.utils.ray_utils import RaySampler, RayGenerator
from meshgen.utils.math_utils import get_ray_limits_box, linspace
from meshgen.utils.misc import get_device
from meshgen.modules.shape2vecset import DiagonalGaussianDistribution


class TriplaneAEModel(pl.LightningModule):
    def __init__(
        self,
        encoder,
        deconv_decoder,
        mlp_decoder,
        triplane_res,
        triplane_ch,
        ckpt_path=None,
        ignore_keys=[],
        scheduler_config=None,
        use_ema=False,
        weight_decay=0.0,
        monitor=None,
        is_shapenet=False,
        box_warp=0.55 * 2,
        use_diso=False,
    ):
        super().__init__()
        self.encoder = instantiate_from_config(encoder)

        self.deconv_decoder = instantiate_from_config(deconv_decoder)
        self.mlp_decoder = instantiate_from_config(mlp_decoder)

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = EMAModel(self.parameters(), decay=0.999)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.scheduler_config = scheduler_config
        self.use_scheduler = scheduler_config is not None
        self.weight_decay = weight_decay

        if monitor is not None:
            self.monitor = monitor

        self.triplane_res = triplane_res
        self.triplane_ch = triplane_ch

        self.loss = torch.nn.BCEWithLogitsLoss()
        self.train()

        self.is_shapenet = is_shapenet

        # self.plane_axes = generate_planes()
        self.register_buffer("plane_axes", generate_planes())
        self.box_warp = box_warp

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.use_diso = use_diso
        if self.use_diso:
            raise NotImplementedError("DiffDMC is not implemented yet.")
            # self.diffdmc = DiffDMC(torch.float32)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0 or len(unexpected) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def encode(self, x):
        z = self.encoder(x)
        z = z.reshape(-1, 3, self.triplane_res, self.triplane_res, self.triplane_ch)
        z = rearrange(z, "b n h w c -> b n c h w")

        return z

    def upsample(self, z):
        if z.ndim == 5:
            z = rearrange(z, "b n c h w -> b c (n h) w")
        z = self.deconv_decoder(z)
        z = rearrange(z, "b c (n h) w -> b n c h w", n=3)

        return z

    def decode(self, z, queries, upsample=False):
        if upsample:
            z = self.upsample(z)
        features = sample_from_planes(
            self.plane_axes, z, queries, box_warp=self.box_warp
        )
        features = rearrange(features, "b np q c -> b q (np c)")
        output = self.mlp_decoder(features)

        return output

    @torch.no_grad()
    def diff_decode_shape(self, z, R=256, skip_upsample=False):
        if not skip_upsample:
            z = self.upsample(z)
        extend = 0.55 if not self.is_shapenet else 1.05
        grid = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(R),
                    torch.arange(R),
                    torch.arange(R),
                ),
                dim=-1,
            )
            + 0.5
        )
        grid = grid.reshape(-1, 3).to(self.device).float() / R * 2 * extend - extend
        # thresh = 0.5
        query = grid
        rec = self.decode(z, query[None])
        occpancies = torch.sigmoid(rec).reshape(R, R, R)

        return occpancies

    @torch.no_grad()
    def decode_shape(self, z, thresh=0.5, R=256, skip_upsample=False):
        if not skip_upsample:
            z = self.upsample(z)
        mini_bs = 65536 * 16
        extend = 0.55 if not self.is_shapenet else 1.05
        grid = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(R),
                    torch.arange(R),
                    torch.arange(R),
                ),
                dim=-1,
            )
            + 0.5
        )
        occs = []
        grid = grid.reshape(-1, 3).to(self.device).float() / R * 2 * extend - extend
        # thresh = 0.5
        for start in tqdm.trange(0, grid.shape[0], mini_bs, disable=True):
            end = min(start + mini_bs, grid.shape[0])
            query = grid[start:end]
            rec = self.decode(z, query[None])
            occpancy = torch.sigmoid(rec)
            occs.append(occpancy.cpu())

        occs = torch.cat(occs, dim=0).reshape(R, R, R).to(self.device)
        if not self.use_diso:
            vertices_pred, faces_pred, normals_pred, _ = measure.marching_cubes(
                occs.detach().cpu().float().numpy(), thresh, method="lewiner"
            )
        else:
            vertices_pred, faces_pred = self.diffdmc(
                -occs.float(), isovalue=-thresh, normalize=True
            )

        return vertices_pred, faces_pred

    def forward(self, x, queries):
        z = self.encode(x)
        z = self.upsample(z)
        x_hat = self.decode(z, queries)

        return x_hat

    def shared_step(self, batch, batch_idx):
        surface, queries, occupancies = (
            batch["surface"],
            batch["queries"],
            batch["occupancies"],
        )

        pred = self(surface, queries)
        target = occupancies

        loss_dict = {"loss": self.loss(pred, target).item()}

        return self.loss(pred, target), loss_dict

    def log_prefix(self, loss_dict, prefix):
        for k, v in loss_dict.items():
            self.log(
                f"{prefix}/{k}",
                v,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, batch_idx)
        self.log_prefix(loss_dict, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, batch_idx)
        self.log_prefix(loss_dict, "val")
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        param_groups = self.get_optim_groups()
        opt = torch.optim.AdamW(param_groups, lr=lr)
        if self.use_scheduler:
            assert "target" in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    "scheduler": LambdaLR(opt, lr_lambda=scheduler.schedule),
                    "interval": "step",
                    "frequency": 1,
                }
            ]
            return [opt], scheduler
        return opt

    def get_optim_groups(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (
            torch.nn.LayerNorm,
            torch.nn.Embedding,
            torch.nn.Parameter,
            torch.nn.GroupNorm,
        )
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith("embed"):
                    no_decay.add(fpn)
                elif pn.endswith("pos_emb"):
                    no_decay.add(fpn)
                elif pn.endswith("query"):
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]

        return optim_groups

    @torch.no_grad()
    def eval_one(self, surface, do_rendering=False):
        if isinstance(surface, (str, Path)):
            vertices, faces = load_mesh(surface, self.device)
            surface = sample_on_surface(vertices, faces, self.encoder.num_inputs)
        latents = self.encode(surface[None])
        v, f = self.decode_shape(latents)

        if do_rendering:
            renders = render_mesh_spiral_offscreen(
                v, f, elevation=20, radius=3, num_frames=90
            )
            return v, f, renders
        else:
            return v, f

    def on_train_epoch_end(self, *args, **kwargs):
        self.eval()
        logdir = self.trainer.logdir  ###
        this_logdir = Path(logdir) / f"spirals/epoch_{self.current_epoch}"
        this_logdir.mkdir(exist_ok=True, parents=True)

        local_rank = self.local_rank
        world_size = self.trainer.world_size
        meshes = list(Path("./data/eval_meshes").glob("*.off"))[:8]
        meshes = meshes[local_rank::world_size]

        for mesh in tqdm.tqdm(meshes, disable=local_rank):
            try:
                _, _, rendered = self.eval_one(mesh, do_rendering=True)
                write_video(this_logdir / f"{mesh.stem}.mp4", rendered)
            except TypeError:
                pass

        self.train()

    def on_train_start(self):
        return
        self.eval()
        logdir = self.trainer.logdir  ###
        this_logdir = Path(logdir) / f"spirals/before_training"
        this_logdir.mkdir(exist_ok=True, parents=True)

        local_rank = self.local_rank
        world_size = self.trainer.world_size
        meshes = list(Path("./data/eval_meshes").glob("*.off"))[:8]
        meshes = meshes[local_rank::world_size]

        for mesh in tqdm.tqdm(meshes, disable=local_rank):
            _, _, rendered = self.eval_one(mesh, do_rendering=True)
            write_video(this_logdir / f"{mesh.stem}.mp4", rendered)

        self.train()


class TriplaneKLModel(TriplaneAEModel):
    def __init__(self, *args, kl_weight=1e-5, tv_loss_weight=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.kl_weight = kl_weight
        self.tv_loss_weight = tv_loss_weight

    def encode(self, x, return_kl=False):
        z = self.encoder(x)
        z = z.reshape(-1, 3, self.triplane_res, self.triplane_res, self.triplane_ch * 2)
        mean, logvar = z.chunk(2, dim=-1)
        posterior = DiagonalGaussianDistribution(mean, logvar)
        z = posterior.sample()
        z = rearrange(z, "b n h w c -> b n c h w")

        if return_kl:
            kl = posterior.kl()
            return z, kl
        else:
            return z

    def forward(self, x, queries):
        z, kl = self.encode(x, return_kl=True)
        z = self.upsample(z)
        x_hat = self.decode(z, queries)

        return z, x_hat, kl

    def shared_step(self, batch, batch_idx):
        surface, queries, occupancies = (
            batch["surface"],
            batch["queries"],
            batch["occupancies"],
        )

        z, pred, kl = self(surface, queries)
        kl = torch.mean(kl)
        target = occupancies

        loss_dict = {"recon_loss": self.loss(pred, target).item(), "kl": kl.item()}

        loss = self.loss(pred, target) + self.kl_weight * kl

        if self.tv_loss_weight > 0:
            tv_loss = compute_tv_loss(z)
            loss += self.tv_loss_weight * tv_loss
            loss_dict["tv_loss"] = tv_loss

        return loss, loss_dict
