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
from diso import DiffDMC
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
            self.diffdmc = DiffDMC(torch.float32)

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


class TriplaneKLModel_w_RenderLoss(TriplaneKLModel):
    def __init__(
        self,
        *args,
        occ_loss_weight=1.0,
        render_loss_weight=1.0,
        sparse_loss_weight=0.0,
        mesh_extraction_reso: int = 128,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.occ_loss_weight = occ_loss_weight
        self.render_loss_weight = render_loss_weight
        self.sparse_loss_weight = sparse_loss_weight

        # change to register buffer
        self.fov = 50
        self.render_reso = 512
        proj_mtx = get_projection_matrix(self.fov, self.render_reso).astype(np.float32)
        self.register_buffer("proj_mtx", torch.from_numpy(proj_mtx))
        self.diffdmc = DiffDMC(torch.float32)
        self.lpips = LearnedPerceptualImagePatchSimilarity("vgg", normalize=False)
        self.mesh_extraction_reso = mesh_extraction_reso

        # try:
        #     self.glctx = dr.RasterizeGLContext()
        # except:
        #     self.glctx = dr.RasterizeCudaContext()
        # else:
        #     self.glctx = dr.RasterizeCudaContext()
        # self.glctx = dr.RasterizeGLContext()
        torch.autograd.set_detect_anomaly(True)

    def forward(self, x, queries):
        z, kl = self.encode(x, return_kl=True)
        z = self.upsample(z)
        x_hat = self.decode(z, queries)

        return x_hat, kl, z

    def diff_render_mesh(self, v, f, c2w, reso):
        if not hasattr(self, "glctx"):
            self.glctx = dr.RasterizeCudaContext(device=self.device)

        f = f.to(torch.int32)
        v_cam = torch.matmul(
            F.pad(v, pad=(0, 1), mode="constant", value=1.0), torch.inverse(c2w).T
        ).unsqueeze(0)
        v_clip = v_cam @ self.proj_mtx.T
        v_clip[..., 1] *= -1

        rast, rast_db = dr.rasterize(self.glctx, v_clip, f, (reso, reso))

        # alpha = (rast[0, ..., 3:] > 0).float()
        hard_alpha = rast[..., 3:].clamp(0.0, 1.0)
        alpha = dr.antialias(hard_alpha, rast, v_clip, f)

        vn = calc_normal(v, f)

        normal, _ = dr.interpolate(vn.unsqueeze(0).contiguous(), rast, f)
        normal = dr.antialias(normal, rast, v_clip, f)
        normal = safe_normalize(normal)
        normal = torch.lerp(
            torch.ones_like(normal), (normal + 1.0) / 2.0, hard_alpha.float()
        )

        return normal[0], alpha[0]

    def diff_decode_shape(self, z, R=256, upsample=False):
        if upsample:
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
        grid = grid.reshape(-1, 3).to(self.device) / R * 2 * extend - extend
        # thresh = 0.5
        query = grid
        rec = self.decode(z, query[None])

        # occpancies = -1 * (torch.sigmoid(rec).reshape(R, R, R) - 0.5)
        occpancies = -rec.reshape(R, R, R)

        v, f = self.diffdmc(occpancies.clone(), isovalue=0.0, normalize=True)
        v = v * 2 * extend - extend

        return v, f, rec.reshape(R, R, R)

        # return vertices_pred, faces_pred

    def shared_step(self, batch, batch_idx):
        # assumes batch_size == 1
        start = 72
        end = 108
        surface, queries, occupancies = (
            batch["surface"],
            batch["queries"],
            batch["occupancies"],
        )

        poses, normals = batch["poses"], batch["normals"]
        depths = batch["depths"]
        # breakpoint()

        pred, kl, z = self(surface, queries)
        kl = torch.mean(kl)
        target = occupancies

        v, f, rec = self.diff_decode_shape(z, R=self.mesh_extraction_reso)
        start_idx = self.trainer.train_dataloader.dataset.datasets.start_idx
        if batch_idx % 50 == 0:
            export_mesh(v, f, f"paper/ablation_diso/{start_idx}/{batch_idx}_no.obj")

        normals = normals[0][start:end]
        depths = depths[0][start:end]

        gt_mask = (normals.norm(dim=-1) < 1.5).float()
        # mask_frames = repeat(gt_mask, "t h w -> t h w 3")
        # write_video("trash/test_mask.mp4", mask_frames)
        # breakpoint()

        if len(v) and self.render_loss_weight > 0.0:
            # if len(v):
            frames = []
            masks = []
            with torch.autocast("cuda", enabled=False):
                for idx in range(start, end):
                    rendered_normals, rendered_mask = self.diff_render_mesh(
                        v, f, poses[0][idx], 512
                    )
                    frames.append(rendered_normals)
                    masks.append(rendered_mask)
            frames = torch.stack(frames, dim=0)
            masks = torch.stack(masks, dim=0)

            if self.local_rank == 0:
                if batch_idx % 10 == 0:
                    if batch_idx == 0:
                        shutil.rmtree(Path("trash/test_diso"), ignore_errors=True)
                    write_video(
                        f"trash/test_diso/{self.global_step}.mp4",
                        torch.cat([frames, normals], dim=2),
                    )

            frames = rearrange(frames, "t h w c -> t c h w") * 2 - 1
            normals = rearrange(normals, "t h w c -> t c h w") * 2 - 1

            # render_loss = F.mse_loss(frames, normals) + self.lpips(frames, normals)
            # render_loss = F.mse_loss(frames, normals)
            similarity = (frames * normals).sum(dim=-3).abs()
            render_loss = 1 - similarity[gt_mask > 0].mean()

            render_loss += self.lpips(frames, normals)

            mask_loss = F.mse_loss(masks, gt_mask[..., None])
        else:
            render_loss = 0.0
            mask_loss = 0.0

        sparse_loss = torch.mean(torch.sigmoid(rec))

        opacity_loss = self.compute_opacity_loss(
            gt_mask, poses[0][start:end], depths, torch.sigmoid(rec), mask_reso=512
        )

        recon_loss = self.loss(pred, target)

        loss_dict = {
            "recon_loss": recon_loss,
            "kl": kl,
            "render_loss": render_loss,
            "sparse_loss": sparse_loss,
            "mask_loss": mask_loss,
            "opacity": opacity_loss,
        }

        loss = (
            self.occ_loss_weight * recon_loss
            + self.kl_weight * kl
            + self.render_loss_weight * render_loss
            # + self.sparse_loss_weight * sparse_loss
            # + mask_loss * 1.0
            # + opacity_loss * 0.5
        )

        if self.tv_loss_weight > 0:
            tv_loss = compute_tv_loss(z)
            loss += self.tv_loss_weight * tv_loss
            loss_dict["tv_loss"] = tv_loss

        return loss, loss_dict

    def compute_opacity_loss(self, mask, c2ws, depths, rec, mask_reso):
        ## NOTE: the mask should be inverted
        # mask = rearrange(mask, "t h w c -> t c h w")
        # mask = F.interpolate(mask, (mask_reso, mask_reso), mode="bilinear")
        # mask = mask.view(mask.shape[0], -1).bool()

        # rays_o, rays_d = RayGenerator()(c2ws, self.fov, mask_reso)
        # breakpoint()
        # rays_o = rays_o[mask]
        # rays_d = rays_d[mask]
        # ray_samples = RaySampler(
        #     num_samples_per_ray=128,
        #     bbox_length=self.box_warp,
        #     disparity=False,
        #     drop_invalid=True,
        # )(rays_o, rays_d).view(-1, 3)

        # side_length = self.box_warp / 2
        # ray_samples = ray_samples / side_length

        # opacities = F.grid_sample(
        #     rec[None, None], ray_samples[None, None, None], padding_mode="border"
        # ).squeeze()

        # opacity_loss = torch.mean(opacities)

        # return opacity_loss

        # # the max version
        # mask = rearrange(mask, "t h w c -> t c h w")
        # mask = F.interpolate(mask, (mask_reso, mask_reso), mode="bilinear")
        # mask = mask.view(mask.shape[0], -1).bool()

        # rays_o, rays_d = RayGenerator()(c2ws, self.fov, mask_reso)
        # rays_o = rays_o[mask]
        # rays_d = rays_d[mask]
        # ray_samples = RaySampler(
        #     num_samples_per_ray=128,
        #     bbox_length=self.box_warp,
        #     disparity=False,
        #     drop_invalid=True,
        # )(rays_o, rays_d)

        # side_length = self.box_warp / 2
        # ray_samples = ray_samples / side_length

        # opacities = F.grid_sample(
        #     rec[None, None], ray_samples[None, None], padding_mode="border"
        # ).squeeze()  # [n_samples, n_rays]
        # opacities = rearrange(opacities, "n_samples n_rays -> n_rays n_samples")

        # opacity_loss = torch.mean(torch.max(opacities, dim=1)[0])

        # the MeshLRM loss version
        # mask = rearrange(mask, "t h w c -> t c h w")
        # mask = F.interpolate(mask, (mask_reso, mask_reso), mode="bilinear")
        # mask = mask.view(mask.shape[0], -1).bool()
        num_samples = 64

        rays_o, rays_d = RayGenerator()(c2ws, self.fov, mask_reso)

        z_axis = -c2ws[:, :3, 2]
        rays_d_normalized_z = rays_d.clone()
        z_comps = torch.einsum("bnc,bc->bn", rays_d_normalized_z, z_axis)
        rays_d_normalized_z /= z_comps[..., None]

        depths = rearrange(depths, "t h w c -> t (h w) c")

        # t_start, t_end = get_ray_limits_box(rays_o, rays_d, self.box_warp)
        t_start, t_end = get_ray_limits_box(rays_o, rays_d_normalized_z, self.box_warp)
        is_ray_valid = t_end > t_start
        depths = torch.where(depths == 0.0, depths, t_end)
        t_end = depths
        # debug
        t_end -= 1.732 / self.mesh_extraction_reso * self.box_warp
        t_end = torch.maximum(t_end, t_start)
        # t_end = t_end.clamp(min=0.0)
        # t_end = t_start

        is_ray_valid = is_ray_valid[..., 0]
        t_start = t_start[is_ray_valid]
        t_end = t_end[is_ray_valid]
        depths = depths[is_ray_valid]
        rays_o = rays_o[is_ray_valid]
        rays_d = rays_d[is_ray_valid]
        rays_d_normalized_z = rays_d_normalized_z[is_ray_valid]

        ray_depths = linspace(t_start, t_end, num_samples)
        ray_depths += (
            torch.rand_like(ray_depths) * (t_end - t_start) / (num_samples - 1)
        )

        # make this depth to ray distance
        ray_samples = rays_o[None] + ray_depths * rays_d_normalized_z[None]
        surface_points = rays_o + depths * rays_d_normalized_z

        side_length = self.box_warp / 2
        ray_samples = ray_samples / side_length

        opacities = F.grid_sample(
            rec[None, None], ray_samples[None, None], padding_mode="border"
        ).squeeze()  # [n_samples, n_rays]

        opacities = rearrange(opacities, "n_samples n_rays -> n_rays n_samples")

        # dists = (surface_points - ray_samples).norm(dim=-1)
        # surface_opacities = F.grid_sample(
        #     rec[None, None], ray_samples[None, None], padding_mode="border"
        # ).squeeze()  # [n_samples, n_rays]
        # surface_opacities = rearrange()

        # opacity_loss = 1 - torch.exp(-opacities * dists)
        # opacity_loss = torch.mean(opacities)
        # opacity_loss = torch.mean(torch.max(opacities, dim=1)[0])
        opacity_loss = torch.mean(torch.sum(opacities, dim=-1))

        return opacity_loss
