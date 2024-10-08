import datetime
import numpy as np
import torch
import tqdm
from pathlib import Path
from omegaconf import OmegaConf
import torch.distributed as dist
from einops import repeat
from PIL import Image
import json

from meshgen.util import instantiate_from_config
from meshgen.utils.io import write_video, export_mesh, write_image
from meshgen.utils.render import (
    render_mesh_spiral_offscreen,
)
from meshgen.utils.images import preprocess_image
from meshgen.utils.remesh import auto_remesh, instantmesh_remesh, pyacvd_remesh

import tyro

import warnings

warnings.filterwarnings("ignore")


def jointgen(
    images: str,
    output: str,
    config: str = "configs/shapegen.yaml",
    ckpt: str = None,
    cfg: float = 3.0,
    n_steps: int = 50,
    n_samples: int = 1,
    rotate: bool = False,
    no_preprocess: bool = False,
    border_ratio: float = 0.35,
    remove_bg: bool = True,
    ignore_alpha: bool = False,
    alpha_matting: bool = False,
    export: bool = False,
    thresh: float = 0.5,
    R: int = 256,
    remesh: bool = False,
    snr_shifting: float = 1.0,
    seed: int = 1234,
    ema: bool = True,
    rembg_backend: str = "bria",
    use_diso: bool = False,
    bf16: bool = True,
):
    render_kwargs = {
        "num_frames": 90,
        "elevation": 0,
        "radius": 2.0,
        "rotate": rotate,
        # "color": np.array([20, 100, 246]) / 255,
    }

    if dist.is_initialized():
        local_rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        world_size = 1

    torch.cuda.set_device(local_rank)
    device = "cuda"
    config_file = config
    print(f"Using config file: {config_file}")
    config = OmegaConf.load(config_file)
    if ckpt is None:
        config.params.ckpt_path = ckpt
    # config.model.params.force_reinit_ema = False
    # config.model.params.autoencoder.params.use_diso = use_diso
    model = instantiate_from_config(config).to(device)
    model.eval()

    cond_key = model.cond_key
    if cond_key == "images":
        images = Path(images)
        if not images.is_dir():
            image_files = [images]
        else:
            image_files = (
                list(images.glob("*.png"))
                + list(images.glob("*.jpg"))
                + list(images.glob("*.webp"))
                + list(images.glob("*.PNG"))
                + list(images.glob("*.JPG"))
                + list(images.glob("*.WEBP"))
                + list(images.glob("*.jpeg"))
                + list(images.glob("*.JPEG"))
            )
    elif cond_key == "text":
        with open(images, "r") as f:
            image_files = f.read().strip().split("\n")
    else:
        raise ValueError(f"Unknown cond_key: {model.cond_key}")

    image_files = sorted(image_files)[local_rank::world_size]

    do_ema = ema and config.params.get("use_ema", False)
    do_refine = hasattr(model, "control_model")

    output_dir = Path(output)
    output_dir.mkdir(exist_ok=True, parents=True)
    if export:
        mesh_output_dir = output_dir / "meshes"
        mesh_output_dir.mkdir(exist_ok=True, parents=True)
        meta = []

    timestep_callback = None
    if snr_shifting != 1.0:
        timestep_callback = lambda ts: [
            t / (snr_shifting - snr_shifting * t + t) for t in ts
        ]

        # fmt: off
        # timestep_callback = lambda ts: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.765, 0.78, 0.795, 0.81, 0.825, 0.84, 0.855, 0.87, 0.885, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]
        # fmt: on

    @torch.no_grad()
    @torch.autocast("cuda", torch.bfloat16, enabled=bf16)
    def vis_one(image_file):
        if cond_key == "images":
            this_image = Image.open(image_file)

            if not no_preprocess:
                this_image = preprocess_image(
                    this_image,
                    size=512,
                    border_ratio=border_ratio,
                    remove_bg=remove_bg,
                    ignore_alpha=ignore_alpha,
                    alpha_matting=alpha_matting,
                    backend=rembg_backend,
                )

            cond = np.array(this_image)
            uncond = np.zeros_like(cond)
            save_name = image_file.stem
        else:
            cond = image_file
            uncond = ""
            save_name = image_file.replace(" ", "_")[:30]

        denoised = model.sample_one(
            cond,
            uncond,
            n_samples=n_samples,
            cfg=cfg,
            n_steps=n_steps,
            timestep_callback=timestep_callback,
            seed=seed,
        )
        if do_refine:
            refined_denoised = model.sample_one(
                cond,
                uncond,
                coarse=denoised,
                n_samples=n_samples,
                cfg=cfg,
                n_steps=n_steps,
                timestep_callback=timestep_callback,
                seed=seed,
            )
        for i in range(n_samples):
            v, f = model.decode_shape(denoised[i : i + 1], thresh=thresh, R=R)
            if do_refine:
                v_refined, f_refined = model.decode_shape(
                    refined_denoised[i : i + 1], thresh=thresh, R=R
                )
            if remesh:
                # v, f = instantmesh_remesh(v, f, target_num_faces=5000)
                v, f = pyacvd_remesh(v, f, target_num_faces=20000)
                v = np.array(v)
                f = np.array(f)
                if do_refine:
                    v_ref_refinedined, f_refined = instantmesh_remesh(
                        v_ref_refinedined, f_refined, target_num_f_refinedaces=3000
                    )
                    v_ref_refinedined = np.array(v_ref_refinedined)
                    f_refined = np.array(f_refined)
            frames = render_mesh_spiral_offscreen(v, f, **render_kwargs)
            if do_refine:
                refined_frames = render_mesh_spiral_offscreen(
                    v_refined, f_refined, **render_kwargs
                )

            if cond_key == "images":
                this_image_float = cond
                cond_frames = repeat(
                    this_image_float, "h w c -> t h w c", t=frames.shape[0]
                )
                frames = np.concatenate([cond_frames, frames], axis=-2)
                if do_refine:
                    frames = np.concatenate([frames, refined_frames], axis=-2)

            write_video(output_dir / f"{save_name}_sample_{i}.mp4", frames)
            if export:
                export_mesh(v, f, mesh_output_dir / f"{save_name}_sample_{i}.obj")
                this_meta = {
                    "mesh": (mesh_output_dir / f"{save_name}_sample_{i}.obj")
                    .absolute()
                    .as_posix(),
                    "image": image_file.absolute().as_posix(),
                }
                meta.append(this_meta)
                if do_refine:
                    export_mesh(
                        v_refined,
                        f_refined,
                        mesh_output_dir / f"{save_name}_sample_{i}_refined.obj",
                    )
                    this_meta = {
                        "mesh": (
                            mesh_output_dir / f"{save_name}_sample_{i}_refined.obj"
                        )
                        .absolute()
                        .as_posix(),
                        "image": image_file.absolute().as_posix(),
                    }
                    meta.append(this_meta)

        if do_ema:
            with model.ema_scope():
                denoised = model.sample_one(
                    cond,
                    uncond,
                    n_samples=n_samples,
                    cfg=cfg,
                    n_steps=n_steps,
                    timestep_callback=timestep_callback,
                    seed=seed,
                )
            for i in range(n_samples):
                v, f = model.decode_shape(denoised[i : i + 1], thresh=thresh, R=R)
                if remesh:
                    v, f = instantmesh_remesh(v, f, target_num_faces=3000)
                    v = np.array(v)
                    f = np.array(f)
                frames = render_mesh_spiral_offscreen(v, f, **render_kwargs)

                if cond_key == "images":
                    this_image_float = cond
                    cond_frames = repeat(
                        this_image_float, "h w c -> t h w c", t=frames.shape[0]
                    )
                    frames = np.concatenate([cond_frames, frames], axis=-2)

                write_video(output_dir / f"{save_name}_sample_ema_{i}.mp4", frames)
                if export:
                    export_mesh(
                        v, f, mesh_output_dir / f"{save_name}_sample_ema_{i}.obj"
                    )
                    this_meta = {
                        "mesh": (mesh_output_dir / f"{save_name}_sample_ema_{i}.obj")
                        .absolute()
                        .as_posix(),
                        "image": image_file.absolute().as_posix(),
                    }
                    meta.append(this_meta)

    for img in tqdm.tqdm(image_files, disable=local_rank):
        vis_one(img)

    if export:
        meta_gathered = [None for _ in range(world_size)]
        dist.barrier()
        dist.all_gather_object(meta_gathered, meta)

        if local_rank == 0:
            merged = []
            for d in meta_gathered:
                merged += d

            json.dump(merged, open(output_dir / "meta.json", "w"), indent=4)


if __name__ == "__main__":
    dist.init_process_group("nccl")
    tyro.cli(jointgen)
