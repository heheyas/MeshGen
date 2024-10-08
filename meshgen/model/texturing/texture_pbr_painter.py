import numpy as np
import torch
from PIL import Image, ImageFilter
from einops import repeat, rearrange
from pathlib import Path
import openai
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image, to_tensor

from meshgen.modules.mesh.textured_mesh import TexturedMesh, uv_padding
from meshgen.utils.render_ops import (
    render_zero123pp_6views,
    render_orthogonal_4views,
    render_zero123pp_6views_rgbs,
)
from meshgen.utils.captioner import captioning
from meshgen.util import instantiate_from_config
from meshgen.utils.io import write_image, write_video
from meshgen.utils.ops import (
    preprocess_ip,
    dilate_depth_outline,
    dilate_mask,
    extract_bg_mask,
)


def get_concat_v(im1, im2):
    dst = Image.new("RGB", (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def get_concat_h(im1, im2):
    dst = Image.new("RGB", (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


class SparseViewPBRPainter(torch.nn.Module):
    def __init__(
        self,
        exp_dir,
        # front_view_generator,
        multiview_generator,
        render_reso=320,
        mesh_kwargs={},
        renderer_kwargs={},
        radius=2.5,
        device="cuda",
        sr_model=None,
        start_timestep_idx=10,
        control_scale=[0.4, 0.1],
        inpainter=None,
        zero123pp_view_idx=[0, 1, 2, 3, 4, 5],
        inpaint_elevations=[],
        inpaint_azimuths=[],
        use_original_ip=True,
        flip_normals=False,
        albedo_prompt="albedo",
        texture_inpainter=None,
        use_pbr=True,
        erode_size: int = 21,
        sync_exp_start: float = 0.0,
        sync_exp_end: float = 8.0,
        sync_latent_end: float = 0.2,
        pbr_decomposer=None,
        spiral_elevation=0,
    ):
        super().__init__()
        self.exp_dir = Path(exp_dir)

        self.zero123pp = instantiate_from_config(multiview_generator).eval()
        self.zero123pp = self.zero123pp.to(device)
        self.zero123pp = self.zero123pp.to(torch.float16)

        if inpainter is not None:
            self.inpainter = instantiate_from_config(inpainter).eval()
            self.inpainter.to(device)
        else:
            self.inpainter = None

        if texture_inpainter is not None:
            self.texture_inpainter = (
                instantiate_from_config(texture_inpainter).eval().to(device)
            )
        else:
            self.texture_inpainter = None

        self.use_original_ip = use_original_ip
        self.erode_size = erode_size
        self.zero123pp_view_idx = zero123pp_view_idx

        self.render_reso = render_reso
        self.sr_model = None
        if sr_model is not None:
            self.sr_model = instantiate_from_config(sr_model).to(device)
            self.render_reso *= 4

        self.radius = radius
        self.use_pbr = use_pbr
        self.sync_exp_start = sync_exp_start
        self.sync_exp_end = sync_exp_end
        self.sync_latent_end = sync_latent_end
        self.mesh_kwargs = mesh_kwargs
        self.mesh_kwargs.exp_path = str(self.exp_dir)
        self.mesh_kwargs.use_pbr = use_pbr
        self.render_kwargs = renderer_kwargs

        self.azimuths = np.deg2rad(np.array([30, 90, 150, 210, 270, 330])).astype(
            np.float32
        )
        self.elevations = (
            -np.deg2rad(np.array([20, -10, 20, -10, 20, -10])) + np.pi / 2
        ).astype(np.float32)
        # self.elevations = (
        #     -np.deg2rad(np.array([30, -20, 30, -20, 30, -20])) + np.pi / 2
        # ).astype(np.float32)
        self.start_timestep_idx = start_timestep_idx
        self.control_scale = control_scale
        self.flip_normals = flip_normals
        self.albedo_prompt = albedo_prompt
        if self.albedo_prompt is None:
            self.albedo_prompt = ""

        self.inpaint_elevations = (
            -np.deg2rad(np.array(inpaint_elevations)) + np.pi / 2
        ).astype(np.float32)
        self.inpaint_azimuths = np.deg2rad(np.array(inpaint_azimuths)).astype(
            np.float32
        )
        # self.uv_tiler = img2imgControlNet()

        self.use_decomposer_target = False
        print("sync_latent_end", self.sync_latent_end)

        if pbr_decomposer is not None:
            self.pbr_decomposer = (
                instantiate_from_config(pbr_decomposer)
                .eval()
                .to(device)
                .to(torch.float16)
            )
            self.use_decomposer_target = True
            print("Using PBR decomposer")
        else:
            self.pbr_decomposer = None

        self.device = device
        self.spiral_elevation = spiral_elevation

    def forward_texturing(
        self,
        textured_mesh,
        zero123pp_6views,
        this_exp_dir,
        front_view_only=False,
        verbose=False,
        metalic_target=None,
        roughness_target=None,
        debug=False,
    ):
        if front_view_only:
            # DEBUG
            return textured_mesh

        zero123pp_6views = np.asarray(zero123pp_6views)

        zero123pp_6views = rearrange(
            zero123pp_6views, "(a h) (b w) c -> (a b) h w c", a=3, b=2
        )

        if metalic_target is not None:
            metalic_target = np.asarray(metalic_target)
            metalic_target = rearrange(
                metalic_target, "(a h) (b w) c -> (a b) h w c", a=3, b=2
            )
        if roughness_target is not None:
            roughness_target = np.asarray(roughness_target)
            roughness_target = rearrange(
                roughness_target, "(a h) (b w) c -> (a b) h w c", a=3, b=2
            )

        def process_target(_target, field="albedo"):
            if self.sr_model is not None:
                _target = self.sr_model(_target)
            reso = textured_mesh.renderer.dims
            _target = _target.astype(np.float32) / 255.0
            _target = torch.from_numpy(_target).to(textured_mesh.device)
            write_image(
                this_exp_dir / f"_view_{view_idx + 1}_{field}_target.png",
                _target,
                format="hwc",
            )
            _target = rearrange(_target, "h w c -> c h w")
            # _target = F.interpolate(
            #     _target[None], (self.render_reso, self.render_reso), mode="bilinear"
            # )
            _target = F.interpolate(_target[None], reso, mode="bilinear")

            return _target

        # for view_idx, (azi, ele, target) in enumerate(
        #     zip(self.azimuths, self.elevations, zero123pp_6views)
        # ):
        for view_idx in self.zero123pp_view_idx:
            azi = self.azimuths[view_idx]
            ele = self.elevations[view_idx]

            target = zero123pp_6views[view_idx]

            # if self.sr_model is not None:
            #     target = self.sr_model(target)
            # target = target.astype(np.float32) / 255.0
            # target = torch.from_numpy(target).to(textured_mesh.device)
            # write_image(
            #     this_exp_dir / f"_view_{view_idx + 1}_target.png", target, format="hwc"
            # )
            # target = rearrange(target, "h w c -> c h w")
            # target = F.interpolate(
            #     target[None], (self.render_reso, self.render_reso), mode="bilinear"
            # )
            target = process_target(target)
            if metalic_target is not None:
                this_metalic_target = process_target(
                    metalic_target[view_idx], "metalic"
                )
            else:
                this_metalic_target = None
            if roughness_target is not None:
                this_roughness_target = process_target(
                    roughness_target[view_idx], "roughness"
                )
            else:
                this_roughness_target = None

            # debug
            textured_mesh.forward_texturing(
                theta=ele,
                phi=azi,
                view_target=target,
                radius=self.radius,
                view_id=view_idx + 1,
                save_result_dir=this_exp_dir,
                verbose=verbose,
                erode_size=self.erode_size,
                flip_normals=self.flip_normals,
                depth_filter=True,
                metalic_target=this_metalic_target,
                roughness_target=this_roughness_target,
            )
        return textured_mesh

    def forward_texturing_zero123pp(
        self,
        textured_mesh,
        zero123pp_6views,
        zero123pp_6views_masks,
        this_exp_dir,
        verbose=False,
        debug=False,
    ):
        zero123pp_6views = np.asarray(zero123pp_6views)

        zero123pp_6views = rearrange(
            zero123pp_6views, "(a h) (b w) c -> (a b) h w c", a=3, b=2
        )
        zero123pp_6views_masks = rearrange(
            zero123pp_6views_masks, "(a h) (b w) c -> (a b) h w c", a=3, b=2
        )

        for view_idx, (azi, ele, target) in enumerate(
            zip(self.azimuths, self.elevations, zero123pp_6views)
        ):
            if self.sr_model is not None:
                target = self.sr_model(target)
            target = target.astype(np.float32) / 255.0
            target = torch.from_numpy(target).to(textured_mesh.device)
            write_image(
                this_exp_dir / f"_view_{view_idx}_target.png", target, format="hwc"
            )
            target = rearrange(target, "h w c -> c h w")
            target = F.interpolate(
                target[None], (self.render_reso, self.render_reso), mode="bilinear"
            )
            textured_mesh.forward_texturing(
                theta=ele,
                phi=azi,
                view_target=target,
                radius=self.radius,
                view_id=view_idx + 1,
                save_result_dir=this_exp_dir,
                # depth_filter=True,
                verbose=verbose,
            )

            if debug:
                frames = textured_mesh.render_spiral(90, 20, self.radius)
                write_video(this_exp_dir / f"spiral_after_{view_idx}.mp4", frames)
                textured_mesh.export_glb(this_exp_dir / f"mesh_after_{view_idx}.glb")

        return textured_mesh

    @torch.no_grad()
    def forward(
        self,
        mesh_filename,
        image_filename,
        skip_captioning=False,
        verbose=False,
        ignore_alpha=True,
        **kwargs,
    ):
        this_exp_dir = self.exp_dir / Path(image_filename).stem
        this_exp_dir.mkdir(exist_ok=True, parents=True)
        this_name = Path(image_filename).stem
        reso = 512
        ip = Image.open(image_filename)
        if not skip_captioning:
            try:
                prompt = captioning(ip.convert("RGB"), "high")
            except openai.BadRequestError:
                prompt = "A delicated 3D asset with detailed textures and high quality rendering."
            if verbose:
                print("=" * 30)
                print(f"Prompt for {image_filename}")
                print(prompt)
                print("=" * 30)
        else:
            prompt = "A delicated 3D asset with detailed textures and high quality rendering."

        self.mesh_kwargs.exp_path = str(this_exp_dir)
        mesh = TexturedMesh(
            mesh_filename=mesh_filename,
            renderer_kwargs=self.render_kwargs,
            device=self.device,
            **self.mesh_kwargs,
        ).to(self.device)

        rendered = render_orthogonal_4views(
            mesh.mesh,
            reso,
            radius=self.radius,
            depth_min_val="disp",
            renderer=mesh.renderer,
        )
        depth_map = rendered["depths"][0]
        normal_map = rendered["normals"][0]
        foreground_mask = rendered["masks"][0].cpu().numpy()
        ip = preprocess_ip(
            ip,
            ref_mask=foreground_mask.squeeze(),
            ref_image=depth_map,
            ignore_alpha=ignore_alpha,
        ).resize((reso, reso))
        ip.save(this_exp_dir / f"processed_ip.png")
        depth_map = repeat(
            (depth_map.clamp(0, 1) * 255).cpu().numpy(), "h w 1 -> h w c", c=3
        )
        normal_map = Image.fromarray(
            (normal_map.clamp(0, 1) * 255).cpu().numpy().astype(np.uint8)
        ).resize((reso, reso))
        depth_map = Image.fromarray(depth_map.astype("uint8")).resize((reso, reso))
        nd = get_concat_v(depth_map, normal_map)
        if verbose:
            nd.save(this_exp_dir / f"{this_name}_depth_normal.png")

        depth_bias = self.radius - mesh.mesh_scale * np.sqrt(3)
        depth_max = 2 * mesh.mesh_scale * np.sqrt(3)

        zero123pp_renders = render_zero123pp_6views_rgbs(
            mesh,
            radius=self.radius,
            renderer=mesh.renderer,
            depth_min_val="raw",
            flip_normals=self.flip_normals,
        )
        zero123pp_6views_masks = zero123pp_renders["masks"]
        zero123pp_depths = zero123pp_renders["depths"]
        zero123pp_depths = (zero123pp_depths - depth_bias) / depth_max
        zero123pp_depths[..., 3:] = zero123pp_6views_masks

        zero123pp_6views_rgb = zero123pp_renders["rgbs"]

        zero123pp_normals = zero123pp_renders["normals"]
        zero123pp_img = Image.fromarray(
            (zero123pp_6views_rgb * 255).cpu().numpy().astype(np.uint8)
        )
        zero123pp_depths_img = Image.fromarray(
            (zero123pp_depths * 255).cpu().numpy().astype(np.uint8)
        )
        zero123pp_normals_img = Image.fromarray(
            (zero123pp_normals * 255).cpu().numpy().astype(np.uint8)
        )

        zero123pp_concat = get_concat_h(
            get_concat_h(zero123pp_img, zero123pp_depths_img), zero123pp_normals_img
        )
        # zero123pp_img.save(this_exp_dir / f"{this_name}_zero123pp.png")
        zero123pp_depths_img.save(this_exp_dir / f"{this_name}_zero123pp_depth.png")
        zero123pp_normals_img.save(this_exp_dir / f"{this_name}_zero123pp_normals.png")
        zero123pp_concat.save(this_exp_dir / f"{this_name}_zero123pp_concat.png")

        frames = []

        with torch.inference_mode():
            if self.sync_latent_end < 1 and self.sync_latent_end > 0:
                self.zero123pp.change_scheduler("ddpm")
            else:
                self.zero123pp.change_scheduler("ddim")
            # zero123pp_6views_shaded = self.zero123pp.inference_with_latent_mesh(
            #     mesh,
            #     radius=self.radius,
            #     dilation_size=0,
            #     image=ip if self.use_original_ip else front_view_img,
            #     prompt="",
            #     depth_image=zero123pp_depths_img,
            #     normal_image=zero123pp_normals_img,
            #     num_inference_steps=50,
            #     fusion_method="vanilla",
            #     render_reso=(120, 120),
            #     interpolation_mode="bilinear",
            #     exp_start=self.sync_exp_start,
            #     exp_end=self.sync_exp_end,
            #     sync_latent_end=self.sync_latent_end,
            # ).images[0]
            zero123pp_6views_shaded = self.zero123pp.forward_pipeline(
                image=ip,
                prompt="",
                depth_image=zero123pp_depths_img,
                normal_image=zero123pp_normals_img,
                num_inference_steps=50,
            ).images[0]

            if verbose:
                zero123pp_6views_shaded.save(this_exp_dir / f"zero123pp_shaded.png")

            if self.pbr_decomposer is not None:
                image = ip
                mv = zero123pp_6views_shaded
                albedo = self.pbr_decomposer(
                    image=image,
                    ip2p_cond=mv,
                    prompt="albedo",
                    guidance_scale=5,
                    image_guidance_scale=1.0,
                    num_inference_steps=30,
                ).images[0]

                metalic = self.pbr_decomposer(
                    image=image,
                    ip2p_cond=mv,
                    prompt="metalic",
                    guidance_scale=5,
                    image_guidance_scale=1.0,
                    num_inference_steps=30,
                ).images[0]

                roughness = self.pbr_decomposer(
                    image=image,
                    ip2p_cond=mv,
                    prompt="roughness",
                    guidance_scale=5,
                    image_guidance_scale=1.0,
                    num_inference_steps=30,
                ).images[0]
                pbr_results = get_concat_h(get_concat_h(albedo, metalic), roughness)
                if verbose:
                    pbr_results.save(this_exp_dir / "pbr_decomposer.png")

            if self.use_pbr:
                if not self.use_decomposer_target:
                    zero123pp_6views_sync_roughness = (
                        self.zero123pp.inference_with_latent_mesh(
                            mesh,
                            radius=self.radius,
                            dilation_size=0,
                            image=ip,
                            prompt="roughness",
                            depth_image=zero123pp_depths_img,
                            normal_image=zero123pp_normals_img,
                            num_inference_steps=50,
                            fusion_method="vanilla",
                            render_reso=(120, 120),
                            interpolation_mode="bilinear",
                            exp_start=self.sync_exp_start,
                            exp_end=self.sync_exp_end,
                            sync_latent_end=self.sync_latent_end,
                        ).images[0]
                    )
                    zero123pp_6views_sync_roughness.save(
                        this_exp_dir / f"zero123pp_sync_roughness.png"
                    )

                    zero123pp_6views_sync_metalic = (
                        self.zero123pp.inference_with_latent_mesh(
                            mesh,
                            radius=self.radius,
                            dilation_size=0,
                            image=ip,
                            prompt="metalic",
                            depth_image=zero123pp_depths_img,
                            normal_image=zero123pp_normals_img,
                            num_inference_steps=50,
                            fusion_method="vanilla",
                            render_reso=(120, 120),
                            interpolation_mode="bilinear",
                            exp_start=self.sync_exp_start,
                            exp_end=self.sync_exp_end,
                            sync_latent_end=self.sync_latent_end,
                        ).images[0]
                    )
                    zero123pp_6views_sync_metalic.save(
                        this_exp_dir / f"zero123pp_sync_metalic.png"
                    )
                else:
                    zero123pp_6views_sync_roughness = roughness
                    zero123pp_6views_sync_metalic = metalic
            else:
                zero123pp_6views_sync_roughness = None
                zero123pp_6views_sync_metalic = None
            self.zero123pp.change_scheduler("ddim")

            self.forward_texturing(
                mesh,
                zero123pp_6views_shaded,
                this_exp_dir,
                verbose=verbose,
                **kwargs,
            )
            fused_texture = mesh.texture_fusion("softmax", pbr=False, use_masked=True)

            with mesh.texture_scope(fused_texture):
                mesh.postprocess_texture_map(upsampler=self.sr_model, upsample=False)
                this_frames = mesh.render_spiral(90, self.spiral_elevation, self.radius)
                frames.append(this_frames)
                mesh.export_glb(this_exp_dir / f"mesh_softmax_baked_in.glb")

            if not self.use_pbr:
                return (
                    zero123pp_6views_shaded,
                    pbr_results,
                    str(this_exp_dir / f"mesh_softmax_baked_in.glb"),
                )

            mesh.refresh_texture(clear_texture_list=True)

            self.forward_texturing(
                mesh,
                zero123pp_6views_shaded if not self.use_decomposer_target else albedo,
                this_exp_dir,
                verbose=verbose,
                metalic_target=zero123pp_6views_sync_metalic,
                roughness_target=zero123pp_6views_sync_roughness,
                **kwargs,
            )
            if self.use_pbr:
                for fuse_m in ["softmax"]:
                    fused_texture, fused_metalic, fused_roughness = mesh.texture_fusion(
                        fuse_m
                    )
                    with mesh.texture_scope(
                        fused_texture, fused_metalic, fused_roughness
                    ):
                        mesh.postprocess_texture_map(
                            upsampler=self.sr_model, upsample=False
                        )
                        this_frames = mesh.render_spiral(
                            90, self.spiral_elevation, self.radius
                        )
                        mesh.export_glb(this_exp_dir / f"mesh_{fuse_m}.glb")
                    frames.append(this_frames)
                for fuse_m in ["softmax"]:
                    fused_texture, fused_metalic, fused_roughness = mesh.texture_fusion(
                        fuse_m, use_masked=True
                    )
                    with mesh.texture_scope(
                        fused_texture, fused_metalic, fused_roughness
                    ):
                        mesh.postprocess_texture_map(
                            upsampler=self.sr_model, upsample=False
                        )
                        this_frames = mesh.render_spiral(
                            90, self.spiral_elevation, self.radius
                        )
                        mesh.export_glb(this_exp_dir / f"mesh_{fuse_m}_masked.glb")
                    frames.append(this_frames)
            else:
                for fuse_m in ["softmax"]:
                    fused_texture = mesh.texture_fusion(fuse_m)
                    with mesh.texture_scope(fused_texture):
                        this_frames = mesh.render_spiral(
                            90, self.spiral_elevation, self.radius
                        )
                        mesh.export_glb(this_exp_dir / f"mesh_{fuse_m}.glb")
                    frames.append(this_frames)

        if self.texture_inpainter is not None:
            prompt = "A UV texture map of " + prompt
            uv_inpainted = self.uv_inpainting(mesh, prompt, ip, this_exp_dir)
            fused_texture, fused_metalic, fused_roughness = mesh.texture_fusion(
                "softmax", use_masked=True
            )
            with mesh.texture_scope(fused_texture, fused_metalic, fused_roughness):
                mesh.postprocess_texture_map(upsampler=self.sr_model, upsample=False)
                this_frames = mesh.render_spiral(90, self.spiral_elevation, self.radius)
                # mesh.export_glb(this_exp_dir / "mesh_uv_inpainted.glb")
            frames.append(this_frames)

        UV_pos = mesh.UV_pos_render((1024, 1024))[0]
        UV_pos = rearrange(UV_pos, "h w c -> c h w")
        UV_pos = to_pil_image(UV_pos)

        # uv_inpainted = to_pil_image(uv_inpainted[0])
        # uv_tiled = self.uv_tiler.inference(
        #     uv_inpainted,
        #     UV_pos,
        #     prompt="best quality",
        #     strength=0.5,
        # ).images[0]
        # uv_tiled.save(this_exp_dir / "_uv_tiled.png")
        # uv_tiled = to_tensor(uv_tiled).to(mesh.device)
        # UV_mask = 1 - mesh.UV_mask_render((1024, 1024))[0]
        # UV_mask = rearrange(UV_mask, "h w c -> c h w").mean(0, keepdim=True)
        # to_pil_image(UV_mask).save(this_exp_dir / "_UV_mask.png")
        # uv_tiled = uv_padding(uv_tiled, UV_mask, format="chw")[None]
        # with mesh.texture_scope(uv_tiled):
        #     mesh.postprocess_texture_map(upsampler=self.sr_model)
        #     this_frames = mesh.render_spiral(90, 0, self.radius)
        #     frames.append(this_frames)

        frames = torch.cat(frames, dim=2)
        write_video(self.exp_dir / f"{this_name}_all_spiral.mp4", frames)

    # def inpainting(
    #     self, textured_mesh: TexturedMesh, prompt, ip, this_exp_dir, verbose=True
    # ):
    #     # inpaint views
    #     def blend_bg(img, mask, bg_type="white"):
    #         img = img * mask
    #         if bg_type == "white":
    #             img += 1 - mask

    #         return img

    #     UV_normal = textured_mesh.UV_normal_render()
    #     UV_normal = rearrange(UV_normal, "b h w c -> b c h w")
    #     UV_valid_mask = (UV_normal != 0).any(dim=1, keepdim=True).float()
    #     UV_uncolored_mask = (
    #         (
    #             textured_mesh.texture_img.data
    #             == torch.tensor(textured_mesh.default_color)[None, :, None, None].to(
    #                 textured_mesh.device
    #             )
    #         )
    #         .all(dim=1, keepdim=True)
    #         .float()
    #     )
    #     UV_mask = UV_valid_mask * UV_uncolored_mask
    #     write_image(this_exp_dir / "_UV_valid_mask.png", UV_valid_mask[0], format="chw")
    #     write_image(
    #         this_exp_dir / "_UV_uncolored_mask.png", UV_uncolored_mask[0], format="chw"
    #     )
    #     write_image(this_exp_dir / "_UV_joint_mask.png", UV_mask[0], format="chw")
    #     fused_texture = textured_mesh.texture_fusion("softmax")
    #     # original_texture = textured_mesh.texture_img.data
    #     fused_texture_img = (
    #         textured_mesh.texture_img.data * (1 - UV_mask) + UV_mask * fused_texture
    #     )
    #     write_image(
    #         this_exp_dir / "fused_texture.png", fused_texture_img[0], format="chw"
    #     )

    #     for idx, (ele, azi) in enumerate(
    #         zip(self.inpaint_elevations, self.inpaint_azimuths)
    #     ):
    #         this_rendered = textured_mesh.render(
    #             theta=ele,
    #             phi=azi,
    #             radius=self.radius,
    #             depth_min_val="disp",
    #         )
    #         this_mask = this_rendered["mask"][0]
    #         this_image = this_rendered["image"][0]
    #         this_depth = this_rendered["depth"][0]
    #         this_normal = this_rendered["normals"][0]
    #         this_uncolored_mask = this_rendered["uncolored_mask"][0]
    #         this_uncolored_mask[this_uncolored_mask > 0.5] = 1.0
    #         this_uncolored_mask[this_uncolored_mask < 0.5] = 0.0

    #         original_texture = textured_mesh.texture_img.data
    #         textured_mesh.texture_img.data = fused_texture_img
    #         this_rendered = textured_mesh.render(
    #             theta=ele,
    #             phi=azi,
    #             radius=self.radius,
    #             depth_min_val=-0.5,
    #         )
    #         this_image = this_rendered["image"][0]

    #         textured_mesh.texture_img.data = original_texture

    #         this_image = to_pil_image(blend_bg(this_image, this_mask, "white"))
    #         this_depth = to_pil_image(blend_bg(this_depth, this_mask, "black"))

    #         this_depth_original = this_depth.copy()
    #         this_depth = dilate_depth_outline(this_depth)
    #         depth_comp = get_concat_h(this_depth_original, this_depth)
    #         depth_comp.save(this_exp_dir / f"_inpainted_{idx}_depth_comp.png")

    #         this_normal = to_pil_image(
    #             blend_bg(this_normal * 0.5 + 0.5, this_mask, "white")
    #         )

    #         this_uncolored_mask = to_pil_image(this_uncolored_mask)

    #         this_uncolored_mask = np.asarray(this_uncolored_mask)
    #         this_uncolored_mask = cv2.GaussianBlur(this_uncolored_mask, (13, 13), 0)
    #         this_uncolored_mask = cv2.threshold(
    #             this_uncolored_mask, 100, 255, cv2.THRESH_BINARY
    #         )[1]
    #         this_uncolored_mask = Image.fromarray(this_uncolored_mask)

    #         this_uncolored_mask_original = this_uncolored_mask.copy()
    #         this_uncolored_mask = dilate_mask(this_uncolored_mask, kernel_size=13)
    #         this_uncolored_mask_comp = get_concat_h(
    #             this_uncolored_mask_original, this_uncolored_mask
    #         )
    #         this_uncolored_mask_comp.save(
    #             this_exp_dir / f"_inpainted_{idx}_uncolored_mask_comp.png"
    #         )

    #         this_image.save(this_exp_dir / f"_inpainted_{idx}_image.png")
    #         this_depth.save(this_exp_dir / f"_inpainted_{idx}_depth.png")
    #         this_uncolored_mask.save(
    #             this_exp_dir / f"_inpainted_{idx}_uncolored_mask.png"
    #         )
    #         this_uncolored_mask_original.save(
    #             this_exp_dir / f"_inpainted_{idx}_uncolored_mask_original.png"
    #         )

    #         this_result = get_concat_h(
    #             get_concat_v(this_image, this_depth),
    #             get_concat_v(this_normal, this_uncolored_mask),
    #         )
    #         this_result.save(this_exp_dir / f"_inpainted_{idx}_source.png")

    #         this_inpainted = self.inpainter.inference(
    #             image=this_image,
    #             mask=this_uncolored_mask,
    #             prompt=prompt,
    #             control=[this_depth, this_normal],
    #             control_scale=[1.0, 0.0, 1.0],
    #             # ip=ip,
    #             strength=1.0,
    #             mask_filtering=False,
    #         )[0]

    #         this_inpainted.save(this_exp_dir / f"_inpainted_{idx}_result.png")
    #         if self.sr_model is not None:
    #             this_inpainted = self.sr_model(np.asarray(this_inpainted))
    #         this_inpainted = this_inpainted.astype(np.float32) / 255.0
    #         this_inpainted = torch.from_numpy(this_inpainted).to(textured_mesh.device)
    #         this_inpainted = rearrange(this_inpainted, "h w c -> c h w")
    #         this_inpainted = F.interpolate(
    #             this_inpainted[None],
    #             (self.render_reso, self.render_reso),
    #             mode="bilinear",
    #         )
    #         textured_mesh.forward_texturing(
    #             theta=ele,
    #             phi=azi,
    #             view_target=this_inpainted,
    #             radius=self.radius,
    #             view_id=f"inpaint_{idx}",
    #             save_result_dir=this_exp_dir,
    #             verbose=verbose,
    #             erode_size=3,
    #         )

    def uv_inpainting(
        self,
        textured_mesh: TexturedMesh,
        prompt,
        ip,
        this_exp_dir,
    ):
        dims = textured_mesh.texture_img.data.shape[-2:]
        UV_pos = textured_mesh.UV_pos_render(dims=dims)[0]
        UV_pos = rearrange(UV_pos, "h w c -> c h w")
        UV_pos_img = to_pil_image(UV_pos)
        UV_normal = textured_mesh.UV_normal_render(dims)[0]
        UV_normal = rearrange(UV_normal, "h w c -> c h w")
        UV_normal_img = to_pil_image(UV_normal)

        UV_valid_mask = (UV_normal != 0).any(dim=0, keepdim=True).float()
        UV_valid_mask_original = UV_valid_mask.clone()
        # UV_valid_mask[UV_valid_mask] = 1.0
        # UV_valid_mask[~UV_valid_mask] = 0.0
        # UV_valid_mask = dilate_mask(UV_valid_mask, kernel_size=2, format="chw")

        if self.use_pbr:
            fused_texture, fused_metalic, fused_roughness = (
                textured_mesh.texture_fusion("softmax", use_masked=True)
            )
        else:
            fused_texture = textured_mesh.texture_fusion("softmax")
        albedo = fused_texture[0]

        # albedo = textured_mesh.texture_img.data[0]
        albedo = albedo * UV_valid_mask
        # albedo = to_pil_image(albedo).resize((512, 512))
        albedo = to_pil_image(albedo)

        albedo.save(this_exp_dir / "_albedo_before_uv_inpaint.png")
        UV_pos_img.save(this_exp_dir / "_UV_pos_before_uv_inpaint.png")
        UV_normal_img.save(this_exp_dir / "_UV_normal_before_uv_inpaint.png")

        # mask_dilated = extract_bg_mask(albedo, dilate_kernel=1)
        # mask_dilated = extract_bg_mask(albedo, dilate_kernel=0)
        mask_dilated = textured_mesh.uncolored_texture_map()
        mask_dilated = to_pil_image(mask_dilated[0].float())

        mask_dilated.save(this_exp_dir / "_mask_dilated.png")

        inpainted_uv = self.texture_inpainter.inference(
            prompt,
            image=albedo,
            mask=mask_dilated,
            UV_pos=UV_pos,
            UV_normal=UV_normal,
            control_scale=1.0,
            guidance_scale=3.0,
            strength=1.0,
        )
        # inpainted_uv = self.texture_inpainter.inference(
        #     prompt,
        #     image=inpainted_uv,
        #     mask=mask_dilated,
        #     UV_pos=UV_pos,
        #     UV_normal=UV_normal,
        # )
        # inpainted_uv = self.texture_inpainter.inference(
        #     prompt,
        #     image=inpainted_uv,
        #     mask=mask_dilated,
        #     UV_pos=UV_pos,
        #     UV_normal=UV_normal,
        # )

        inpainted_uv.save(this_exp_dir / "_inpainted_uv.png")

        # textured_mesh.texture_img.data[0] = to_tensor(inpainted_uv)
        inpainted_uv = to_tensor(inpainted_uv).to(textured_mesh.device)

        inpainted_uv = inpainted_uv * UV_valid_mask_original
        inpainted_uv = inpainted_uv[None]

        mask_dilated_th = (
            to_tensor(mask_dilated)
            .to(textured_mesh.device)[None]
            .mean(dim=1, keepdim=True)
        )
        inpainted_uv = mask_dilated_th * inpainted_uv + (
            1 - mask_dilated_th
        ) * to_tensor(albedo)[None].to(textured_mesh.device)

        if self.use_pbr:
            textured_mesh.texture_list.append(
                (
                    inpainted_uv,
                    inpainted_uv,
                    mask_dilated_th,
                    mask_dilated_th,
                )
            )
        else:
            textured_mesh.texture_list.append(
                (
                    inpainted_uv,
                    inpainted_uv,
                    mask_dilated_th,
                    mask_dilated_th,
                    inpainted_uv.mean(dim=1, keepdim=True),
                    inpainted_uv.mean(),
                )
            )

        return inpainted_uv

    def change_exp_dir(self, exp_dir):
        self.exp_dir = Path(exp_dir)
        self.mesh_kwargs.exp_path = str(self.exp_dir)

    @torch.no_grad()
    def multiview_shaded_gen(
        self,
        mesh_filename,
        image_filename,
        exp_dir=None,
        verbose=False,
        ignore_alpha=True,
        skip_captioning=False,
    ):
        if isinstance(image_filename, Image.Image):
            ip = image_filename
            image_filename = "input.png"
        else:
            ip = Image.open(image_filename)
        if exp_dir is None:
            exp_dir = Path(self.exp_dir) / Path(image_filename).stem
        this_exp_dir = exp_dir
        this_exp_dir.mkdir(exist_ok=True, parents=True)
        this_name = Path(image_filename).stem
        reso = 512
        if not skip_captioning:
            try:
                prompt = captioning(ip.convert("RGB"), "high")
            except openai.BadRequestError:
                prompt = "A delicated 3D asset with detailed textures and high quality rendering."
            if verbose:
                print("=" * 30)
                print(f"Prompt for {image_filename}")
                print(prompt)
                print("=" * 30)
        else:
            prompt = "A delicated 3D asset with detailed textures and high quality rendering."

        self.mesh_kwargs.exp_path = str(this_exp_dir)
        mesh = TexturedMesh(
            mesh_filename=mesh_filename,
            renderer_kwargs=self.render_kwargs,
            device=self.device,
            **self.mesh_kwargs,
        ).to(self.device)

        rendered = render_orthogonal_4views(
            mesh.mesh,
            reso,
            radius=self.radius,
            depth_min_val="disp",
            renderer=mesh.renderer,
        )
        depth_map = rendered["depths"][0]
        normal_map = rendered["normals"][0]
        foreground_mask = rendered["masks"][0].cpu().numpy()
        ip = preprocess_ip(
            ip,
            ref_mask=foreground_mask.squeeze(),
            ref_image=depth_map,
            ignore_alpha=ignore_alpha,
        ).resize((reso, reso))
        ip.save(this_exp_dir / f"processed_ip.png")
        depth_map = repeat(
            (depth_map.clamp(0, 1) * 255).cpu().numpy(), "h w 1 -> h w c", c=3
        )
        normal_map = Image.fromarray(
            (normal_map.clamp(0, 1) * 255).cpu().numpy().astype(np.uint8)
        ).resize((reso, reso))
        depth_map = Image.fromarray(depth_map.astype("uint8")).resize((reso, reso))
        nd = get_concat_v(depth_map, normal_map)
        if verbose:
            nd.save(this_exp_dir / f"{this_name}_depth_normal.png")

        depth_bias = self.radius - mesh.mesh_scale * np.sqrt(3)
        depth_max = 2 * mesh.mesh_scale * np.sqrt(3)

        zero123pp_renders = render_zero123pp_6views_rgbs(
            mesh,
            radius=self.radius,
            renderer=mesh.renderer,
            depth_min_val="raw",
            flip_normals=self.flip_normals,
        )
        zero123pp_6views_masks = zero123pp_renders["masks"]
        zero123pp_depths = zero123pp_renders["depths"]
        zero123pp_depths = (zero123pp_depths - depth_bias) / depth_max
        zero123pp_depths[..., 3:] = zero123pp_6views_masks

        zero123pp_6views_rgb = zero123pp_renders["rgbs"]

        zero123pp_normals = zero123pp_renders["normals"]
        zero123pp_img = Image.fromarray(
            (zero123pp_6views_rgb * 255).cpu().numpy().astype(np.uint8)
        )
        zero123pp_depths_img = Image.fromarray(
            (zero123pp_depths * 255).cpu().numpy().astype(np.uint8)
        )
        zero123pp_normals_img = Image.fromarray(
            (zero123pp_normals * 255).cpu().numpy().astype(np.uint8)
        )

        zero123pp_concat = get_concat_h(
            get_concat_h(zero123pp_img, zero123pp_depths_img), zero123pp_normals_img
        )
        if verbose:
            zero123pp_depths_img.save(this_exp_dir / f"{this_name}_zero123pp_depth.png")
            zero123pp_normals_img.save(
                this_exp_dir / f"{this_name}_zero123pp_normals.png"
            )
            zero123pp_concat.save(this_exp_dir / f"{this_name}_zero123pp_concat.png")

        with torch.inference_mode():
            if self.sync_latent_end < 1 and self.sync_latent_end > 0:
                self.zero123pp.change_scheduler("ddpm")
            else:
                self.zero123pp.change_scheduler("ddim")

            zero123pp_6views_shaded = self.zero123pp.forward_pipeline(
                image=ip,
                prompt="",
                depth_image=zero123pp_depths_img,
                normal_image=zero123pp_normals_img,
                num_inference_steps=50,
            ).images[0]

            if verbose:
                zero123pp_6views_shaded.save(this_exp_dir / f"zero123pp_shaded.png")

        self.zero123pp.change_scheduler("ddim")

        return ip, zero123pp_6views_shaded, mesh

    @torch.no_grad()
    def pbr_decompose(
        self, zero123pp_6views_shaded, ip, this_exp_dir=None, verbose=False
    ):
        image = ip
        mv = zero123pp_6views_shaded
        albedo = self.pbr_decomposer(
            image=image,
            ip2p_cond=mv,
            prompt="albedo",
            guidance_scale=5,
            image_guidance_scale=1.0,
            num_inference_steps=30,
        ).images[0]

        metalic = self.pbr_decomposer(
            image=image,
            ip2p_cond=mv,
            prompt="metalic",
            guidance_scale=5,
            image_guidance_scale=1.0,
            num_inference_steps=30,
        ).images[0]

        roughness = self.pbr_decomposer(
            image=image,
            ip2p_cond=mv,
            prompt="roughness",
            guidance_scale=5,
            image_guidance_scale=1.0,
            num_inference_steps=30,
        ).images[0]
        pbr_results = get_concat_h(get_concat_h(albedo, metalic), roughness)
        if verbose and this_exp_dir is not None:
            pbr_results.save(this_exp_dir / "pbr_decomposer.png")

        return albedo, roughness, metalic

    @torch.no_grad()
    def texture_bp(
        self, mesh, mv_shaded, this_exp_dir, export_dir, verbose=False, **kwargs
    ):
        this_exp_dir = Path(this_exp_dir)
        export_dir = Path(export_dir)
        self.forward_texturing(
            mesh,
            mv_shaded,
            this_exp_dir,
            verbose=verbose,
            **kwargs,
        )
        fused_texture = mesh.texture_fusion("softmax", pbr=False, use_masked=True)

        with mesh.texture_scope(fused_texture):
            mesh.postprocess_texture_map(upsampler=self.sr_model, upsample=False)
            this_frames = mesh.render_spiral(90, self.spiral_elevation, self.radius)
            mesh.export_glb(export_dir / f"mesh_softmax_baked_in.glb")
            write_video(export_dir / f"spiral.mp4", this_frames)

        return str(export_dir / f"mesh_softmax_baked_in.glb"), str(
            export_dir / f"spiral.mp4"
        )
