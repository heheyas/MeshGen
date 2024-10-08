import os
import cv2
import numpy as np
import kaolin as kal
from PIL import Image
from loguru import logger
from pathlib import Path
import datetime
import trimesh
from trimesh.visual import texture, TextureVisuals
from trimesh import Trimesh
from einops import rearrange
import pygltflib
import torch.nn.functional as F
from contextlib import contextmanager
from torchvision.transforms.functional import to_pil_image, to_tensor
import itertools

import torch
import torch.nn as nn

from .mesh import Mesh
from .render import Renderer

from meshgen.utils.io import save_tensor_image, write_image
from meshgen.utils.ops import dilate_mask
from meshgen.utils.box_uv_unwrap import box_projection_uv_unwrap


def inpaint_atlas(image, append_mask=None):
    src_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    src_h = src_hls[:, :, 0]
    tgt_range, thres = 150, 1
    lowerb = tgt_range - thres
    upperb = tgt_range + thres
    mask = cv2.inRange(src=src_h, lowerb=lowerb, upperb=upperb)

    if append_mask is not None:
        mask = np.clip(mask + append_mask[..., 0], 0, 1).astype(np.uint8)
    image_inpaint = cv2.inpaint(
        src=image, inpaintMask=mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA
    )
    return image_inpaint


def uv_padding(image, hole_mask, padding=2, uv_padding_block=4, format="hwc"):
    if format == "chw":
        image = rearrange(image, "c h w -> h w c")
        hole_mask = rearrange(hole_mask, "c h w -> h w c")
    uv_padding_size = padding
    image1 = (image.detach().cpu().numpy() * 255).astype(np.uint8)
    hole_mask = (hole_mask.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)

    block = uv_padding_block
    res = image1.shape[0]
    chunk = res // block
    inpaint_image = np.zeros_like(image1)
    prods = list(itertools.product(range(block), range(block)))
    for i, j in prods:
        patch = cv2.inpaint(
            image1[i * chunk : (i + 1) * chunk, j * chunk : (j + 1) * chunk],
            hole_mask[i * chunk : (i + 1) * chunk, j * chunk : (j + 1) * chunk],
            uv_padding_size,
            cv2.INPAINT_TELEA,
        )
        inpaint_image[i * chunk : (i + 1) * chunk, j * chunk : (j + 1) * chunk] = patch
    inpaint_image = inpaint_image / 255.0
    if format == "chw":
        inpaint_image = rearrange(inpaint_image, "h w c -> c h w")
    return torch.from_numpy(inpaint_image).to(image)


class TexturedMesh(nn.Module):
    def __init__(
        self,
        mesh_filename,
        mesh_scale=1.0,
        initial_texture=None,
        default_color=[0.8, 0.1, 0.8],
        force_run_xatlas=False,
        texture_resolution=[1024, 1024],
        exp_path=None,
        remove_mesh_part_names=["MI_CH_Top"],
        remove_unsupported_buffers=["filamat"],
        device="cuda",
        rotate=True,
        vmap=False,
        renderer_kwargs={},
        unwarp_backend="xatlas",
        texture_update_method="vanilla",
        use_pbr=False,
        use_latent=False,
    ):
        super().__init__()
        if exp_path is None:
            now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            this_name = now + "_" + Path(mesh_filename).stem
            exp_path = Path("log") / this_name
            exp_path.mkdir(parents=True, exist_ok=True)
        else:
            exp_path = Path(exp_path)
        self.exp_path = exp_path
        # self.cache_path = self.exp_path / "cache"
        self.cache_path = None
        self.device = device
        self.unwarp_backend = unwarp_backend
        self.mesh_scale = mesh_scale
        self.use_pbr = use_pbr
        self.use_latent = use_latent
        self.mesh = Mesh(
            mesh_filename,
            self.device,
            target_scale=mesh_scale,
            mesh_dy=0.0,
            remove_mesh_part_names=remove_mesh_part_names,
            remove_unsupported_buffers=remove_unsupported_buffers,
            intermediate_dir=self.exp_path / "convert_results",
            rotate=rotate,
        )
        self.initial_texture_path = initial_texture
        self.force_run_xatlas = force_run_xatlas
        self.default_color = default_color
        self.texture_resolution = [
            texture_resolution[0],
            texture_resolution[1] * self.mesh.material_num,
        ]
        self.renderer = Renderer(
            mesh_face_num=self.mesh.faces.shape[0],
            device=self.device,
            **renderer_kwargs,
        )
        self.refresh_texture()
        self.vt, self.ft = self.init_texture_map(vmap)
        self.face_attributes = kal.ops.mesh.index_vertices_by_faces(
            self.vt.unsqueeze(0), self.ft.long()
        ).detach()
        # texture map list for texture fusion
        self.texture_list = []
        self.texture_update_method = texture_update_method

    def init_paint(self):
        if self.initial_texture_path is not None:
            texture_map = (
                Image.open(self.initial_texture_path)
                .convert("RGB")
                .resize(self.texture_resolution)
            )
            texture = (
                torch.Tensor(np.array(texture_map))
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self.device)
                / 255.0
            )
        else:
            texture = torch.ones(1, 3, *self.texture_resolution).to(
                self.device
            ) * torch.Tensor(self.default_color).reshape(1, 3, 1, 1).to(self.device)
        texture_img = nn.Parameter(texture)
        if self.use_pbr:
            self.metalic_img = nn.Parameter(torch.zeros_like(texture_img))
            self.roughness_img = nn.Parameter(torch.zeros_like(texture_img))
        if self.use_latent:
            self.latent_img = nn.Parameter(
                torch.randn(
                    1,
                    4,
                    self.texture_resolution[0],
                    self.texture_resolution[1],
                )
            )
        return texture_img

    def refresh_texture(self, clear_texture_list=False):
        self.texture_img = self.init_paint()
        self.texture_mask = torch.zeros_like(self.texture_img)
        self.postprocess_edge = torch.zeros_like(self.texture_img)
        self.meta_texture_img = nn.Parameter(torch.zeros_like(self.texture_img))
        self.texture_img_postprocess = None

        if self.use_pbr:
            self.metalic_img = nn.Parameter(torch.zeros_like(self.texture_img))
            self.roughness_img = nn.Parameter(torch.zeros_like(self.texture_img))

        if clear_texture_list:
            self.texture_list = []

    def init_texture_map(self, vmap=True):
        cache_path = self.cache_path
        if cache_path is None:
            cache_exists_flag = False
        else:
            vt_cache, ft_cache = cache_path / "vt.pth", cache_path / "ft.pth"
            cache_exists_flag = vt_cache.exists() and ft_cache.exists()

        run_xatlas = False

        if (
            self.mesh.vt is not None
            and self.mesh.ft is not None
            and self.mesh.vt.shape[0] > 0
            and self.mesh.ft.min() > -1
        ):
            vt = self.mesh.vt.to(self.device)
            ft = self.mesh.ft.to(self.device)
        elif cache_exists_flag:
            vt = torch.load(vt_cache).to(self.device)
            ft = torch.load(ft_cache).to(self.device)
        else:
            run_xatlas = True

        if run_xatlas or self.force_run_xatlas:
            v_num = self.mesh.vertices.shape[0]
            f_num = self.mesh.faces.shape[0]
            # if v_num > 10000 or f_num > 10000:
            #     self.unwarp_backend = "box"
            #     print("!" * 40)
            #     print("Too many vertices or faces, using box unwarp instead")
            #     print("!" * 40)
            if self.unwarp_backend == "xatlas":
                import xatlas

                v_np = self.mesh.vertices.cpu().numpy()
                f_np = self.mesh.faces.int().cpu().numpy()

                # atlas = xatlas.Atlas()
                # atlas.add_mesh(v_np, f_np)
                # chart_options = xatlas.ChartOptions()
                # chart_options.max_iterations = 4
                # atlas.generate(chart_options=chart_options)
                # vmapping, ft_np, vt_np = atlas[0]
                vmapping, ft_np, vt_np = xatlas.parametrize(v_np, f_np)
                self.vmapping = vmapping

                vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(self.device)
                ft = torch.from_numpy(ft_np.astype(np.int64)).long().to(self.device)
                vmapping = (
                    torch.from_numpy(vmapping.astype(np.int64)).long().to(self.device)
                )

                if vmap:
                    self.mesh.vertices = self.mesh.vertices[vmapping]
                    self.mesh.faces = ft
                if cache_path is not None:
                    os.makedirs(cache_path, exist_ok=True)
                    torch.save(vt.cpu(), vt_cache)
                    torch.save(ft.cpu(), ft_cache)
            elif self.unwarp_backend == "box":
                uv, indices = box_projection_uv_unwrap(
                    self.mesh.vertices, self.mesh.vn, self.mesh.faces
                )
                individual_vertices = self.mesh.vertices[self.mesh.faces].reshape(-1, 3)
                individual_faces = torch.arange(
                    individual_vertices.shape[0],
                    device=individual_vertices.device,
                    dtype=self.mesh.faces.dtype,
                ).reshape(-1, 3)
                uv_flat = uv[indices].reshape((-1, 2))
                self.mesh.vertices = individual_vertices
                self.mesh.faces = individual_faces
                self.mesh.vn = self.mesh._compute_normals()
                vt = uv_flat
                ft = individual_faces
        return vt, ft

    def forward(self, x):
        raise NotImplementedError

    def get_params(self):
        return [self.texture_img, self.meta_texture_img]

    @torch.no_grad()
    def export_obj(self, path, export_texture_only=False):
        texture_img = self.texture_img.permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        texture_img = Image.fromarray(
            (texture_img[0].cpu().detach().numpy() * 255).astype(np.uint8)
        )
        if not os.path.exists(path):
            os.makedirs(path)
        texture_img.save(os.path.join(path, f"albedo.png"))

        if self.texture_img_postprocess is not None:
            texture_img_post = (
                self.texture_img_postprocess.permute(0, 2, 3, 1)
                .contiguous()
                .clamp(0, 1)
            )
            texture_img_post = Image.fromarray(
                (texture_img_post[0].cpu().detach().numpy() * 255).astype(np.uint8)
            )
            os.system(
                "mv {} {}".format(
                    texture_img.save(os.path.join(path, f"albedo.png")),
                    texture_img.save(os.path.join(path, f"albedo_before.png")),
                )
            )
            texture_img_post.save(os.path.join(path, f"albedo.png"))

        if export_texture_only:
            return 0

        v, f = self.mesh.vertices, self.mesh.faces.int()
        v_np = v.cpu().numpy()  # [N, 3]
        f_np = f.cpu().numpy()  # [M, 3]
        vt_np = self.vt.detach().cpu().numpy()
        ft_np = self.ft.detach().cpu().numpy()

        # save obj (v, vt, f /)
        obj_file = os.path.join(path, f"mesh.obj")
        mtl_file = os.path.join(path, f"mesh.mtl")

        logger.info(
            f"writing obj mesh to {obj_file} with: vertices:{v_np.shape} uv:{vt_np.shape} faces:{f_np.shape}"
        )
        with open(obj_file, "w") as fp:
            fp.write(f"mtllib mesh.mtl \n")

            for v in v_np:
                fp.write(f"v {v[0]} {v[1]} {v[2]} \n")

            for v in vt_np:
                fp.write(f"vt {v[0]} {v[1]} \n")

            fp.write(f"usemtl mat0 \n")
            for i in range(len(f_np)):
                fp.write(
                    f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} {f_np[i, 1] + 1}/{ft_np[i, 1] + 1} {f_np[i, 2] + 1}/{ft_np[i, 2] + 1} \n"
                )

        with open(mtl_file, "w") as fp:
            fp.write(f"newmtl mat0 \n")
            fp.write(f"Ka 1.000000 1.000000 1.000000 \n")

            fp.write(f"Ks 0.000000 0.000000 0.000000 \n")
            fp.write(f"Tr 1.000000 \n")
            fp.write(f"illum 1 \n")
            fp.write(f"Ns 0.000000 \n")
            fp.write(f"map_Kd albedo.png \n")

        if self.mesh.material_cvt is not None:
            logger.info("Postprocess for multiple texture maps or converted mesh~")
            convert_results_dir = os.path.join(path, "convert_results")
            if not os.path.exists(convert_results_dir):
                os.makedirs(convert_results_dir)
            h, w = self.mesh.material_cvt.shape[:2]
            if w % h != 0:
                logger.info(
                    "Number of material may be inaccurate, please check manually~"
                )
            for material_id, material in enumerate(
                np.split(np.array(texture_img), w // h, axis=1)
            ):
                cv2.imwrite(
                    os.path.join(
                        convert_results_dir, "texture_split{}.png".format(material_id)
                    ),
                    cv2.cvtColor(material, cv2.COLOR_RGB2BGR),
                )

    # @torch.no_grad()
    # def export_glb(self, path):
    #     scene = trimesh.Scene()
    #     albedo_img = (
    #         rearrange(self.texture_img[0], "c h w -> h w c").cpu().numpy().clip(0, 1)
    #     )
    #     albedo_img = Image.fromarray((albedo_img * 255).astype(np.uint8))
    #     uvs = self.vt.cpu().numpy()
    #     material = texture.SimpleMaterial(image=albedo_img)
    #     texture_visual = TextureVisuals(uv=uvs, image=albedo_img, material=material)
    #     face_vertices = kal.ops.mesh.index_vertices_by_faces(
    #         self.mesh.vertices[None], self.mesh.faces
    #     )
    #     face_normals = kal.ops.mesh.face_normals(face_vertices).cpu().numpy()[0]
    #     mesh = Trimesh(
    #         vertices=self.mesh.vertices.cpu().numpy(),
    #         faces=self.mesh.faces.cpu().numpy(),
    #         face_normals=face_normals,
    #         visual=texture_visual,
    #         validate=False,
    #         process=False,
    #     )
    #     scene.add_geometry(mesh)
    #     scene.export(os.path.join(path, "mesh.glb"))

    def forward_texturing(
        self,
        view_target,
        theta,
        phi,
        radius,
        save_result_dir,
        view_id=None,
        verbose=False,
        erode_size=0,
        flip_normals=False,
        metalic_target=None,
        roughness_target=None,
        dims=None,
        skip_texture_update=False,
        interpolation_mode=None,
        depth_filter=False,
    ):
        outputs = self.render(theta=theta, phi=phi, radius=radius, dims=dims)

        uncolored_mask_render = outputs["uncolored_mask"]  # bchw, [0,1]
        uncolored_mask_render[uncolored_mask_render > 0] = 1.0
        # uncolored_mask_render = outputs["mask"]  # bchw, [0,1]

        uncolored_mask_render = (
            torch.from_numpy(
                cv2.erode(
                    uncolored_mask_render[0, 0].detach().cpu().numpy(),
                    np.ones((erode_size, erode_size), np.uint8),
                )
            )
            .to(uncolored_mask_render.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        uncolored_mask_render[uncolored_mask_render > 0] = 1.0

        if depth_filter:
            depth = outputs["depth"]
            depth_img = to_pil_image(depth[0])
            depth_edges = Image.fromarray(cv2.Canny(np.array(depth_img), 10, 110))
            depth_masks = to_tensor(depth_edges).unsqueeze(0).to(uncolored_mask_render)
            depth_masks = dilate_mask(depth_masks[0], 20, format="chw")[None]
            depth_masks[depth_masks > 0] = 1.0
            save_tensor_image(
                depth_masks,
                os.path.join(save_result_dir, f"_view_{view_id}_depth_masks.png"),
            )
            # uncolored_mask_render = (uncolored_mask_render - depth_masks).clamp(0, 1)
            uncolored_mask_render = uncolored_mask_render * (1 - depth_masks)
            save_tensor_image(
                uncolored_mask_render,
                os.path.join(save_result_dir, f"_view_{view_id}_depth_filtered.png"),
            )

        (
            cur_texture_map,
            cur_metalic_map,
            cur_roughness_map,
            cur_texture_mask,
            weight_map,
            cur_texture_map_original,
        ) = self.renderer.forward_texturing_render(
            self.mesh.vertices,
            self.mesh.faces,
            self.face_attributes,
            theta=theta,
            phi=phi,
            radius=radius,
            view_target=view_target,
            uncolored_mask=uncolored_mask_render,
            texture_dims=self.texture_resolution,
            flip_normals=flip_normals,
            metalic_target=metalic_target,
            roughness_target=roughness_target,
            interpolation_mode=interpolation_mode,
        )
        cur_texture_mask = cur_texture_mask.mean(dim=1, keepdim=True)
        cur_texture_masked = cur_texture_mask * cur_texture_map

        if not skip_texture_update:
            weight_map_masked = cur_texture_mask * weight_map
            if not self.use_pbr:
                self.texture_list.append(
                    (cur_texture_masked, cur_texture_map, weight_map, weight_map_masked)
                )
            else:
                self.texture_list.append(
                    (
                        cur_texture_masked,
                        cur_texture_map,
                        weight_map,
                        weight_map_masked,
                        cur_metalic_map,
                        cur_roughness_map,
                    )
                )
        if verbose:
            save_tensor_image(
                view_target,
                os.path.join(save_result_dir, f"_view_{view_id}_view_target.png"),
            )
            save_tensor_image(
                uncolored_mask_render.repeat(1, 3, 1, 1),
                os.path.join(
                    save_result_dir, f"_view_{view_id}_uncolored_mask_render.png"
                ),
            )
            save_t = view_target * uncolored_mask_render.repeat(1, 3, 1, 1)
            save_tensor_image(
                save_t,
                os.path.join(
                    save_result_dir, f"_view_{view_id}_uncolored_masked_img.png"
                ),
            )
            save_tensor_image(
                cur_texture_map,
                os.path.join(save_result_dir, f"_view_{view_id}_cur_texture_map.png"),
            )
            save_tensor_image(
                cur_texture_masked,
                os.path.join(
                    save_result_dir, f"_view_{view_id}_cur_texture_map_masked.png"
                ),
            )
            save_tensor_image(
                cur_texture_mask,
                os.path.join(save_result_dir, f"_view_{view_id}_cur_texture_mask.png"),
            )
            save_tensor_image(
                weight_map,
                os.path.join(save_result_dir, f"_view_{view_id}_weight_map.png"),
            )
            if metalic_target is not None:
                save_tensor_image(
                    cur_metalic_map,
                    os.path.join(
                        save_result_dir, f"_view_{view_id}_cur_metalic_map.png"
                    ),
                )
            if roughness_target is not None:
                save_tensor_image(
                    cur_roughness_map,
                    os.path.join(
                        save_result_dir, f"_view_{view_id}_cur_roughness_map.png"
                    ),
                )

        # if not skip_texture_update:
        #     if self.texture_update_method == "vanilla":

        #         def _update_fn(new_map, old_map):
        #             return new_map * cur_texture_mask + old_map * (1 - cur_texture_mask)

        #         self.texture_img.data = _update_fn(cur_texture_map, self.texture_img)
        #         if metalic_target is not None:
        #             self.metalic_img.data = _update_fn(
        #                 cur_metalic_map, self.metalic_img
        #             )
        #         if roughness_target is not None:
        #             self.roughness_img.data = _update_fn(
        #                 cur_roughness_map, self.roughness_img
        #             )
        #     elif self.texture_update_method == "dilate_then_update":
        #         dilated_mask = None

        #         def _update_fn(new_map, old_map):
        #             pass

        return cur_texture_map, cur_texture_mask, weight_map

    def texture_fusion(
        self,
        fuse_method="softmax",
        use_masked=False,
        edge_padding=True,
        pbr=True,
    ):
        _, texture, weight_map, *_ = self.texture_list[0]

        if use_masked:
            t_idx = 0
            w_idx = 3
        else:
            t_idx = 1
            w_idx = 2

        if self.use_pbr and pbr:
            extra_indices = [4, 5]
        else:
            extra_indices = []

        # for idx, (tex_masked, tex, wei) in enumerate(self.texture_list):
        #     write_image(
        #         "trash/test_fusion/mask_texture_{}.png".format(idx),
        #         tex_masked[0],
        #         format="chw",
        #     )
        #     write_image(
        #         "trash/test_fusion/texture_{}.png".format(idx), tex[0], format="chw"
        #     )
        #     write_image(
        #         "trash/test_fusion/weight_{}.png".format(idx), wei[0], format="chw"
        #     )

        texture_fusion = torch.zeros_like(texture, dtype=torch.float32)
        extra_textures = [torch.zeros_like(texture) for _ in extra_indices]

        weight_maps = torch.cat([tl[w_idx] for tl in self.texture_list], dim=1)
        if fuse_method == "softmax":
            weight_maps[weight_maps < 1e-3] = -1e3
            fused_weights = torch.unbind(torch.softmax(weight_maps * 10, dim=1), dim=1)
        elif fuse_method == "l2":
            fused_weights = weight_maps**2 / (weight_maps**2).sum(dim=1, keepdim=True)
            fused_weights = torch.unbind(fused_weights, dim=1)
        elif fuse_method == "l6":
            weight_maps[weight_maps < 1e-2] = 0.0
            fused_weights = weight_maps**6 / (
                (weight_maps**6).sum(dim=1, keepdim=True) + 1e-5
            )
            fused_weights = torch.unbind(fused_weights, dim=1)
        elif fuse_method == "max":
            weight_maps_indices = weight_maps.argmax(dim=1)
            weight_maps = F.one_hot(
                weight_maps_indices, num_classes=weight_maps.shape[1]
            )
            fused_weights = torch.unbind(weight_maps, dim=1)

        for i in range(len(self.texture_list)):
            this_texture = self.texture_list[i][t_idx]
            original_weight = self.texture_list[i][w_idx]

            try:
                this_extras = [self.texture_list[i][idx] for idx in extra_indices]
            except IndexError:
                this_extras = []

            if edge_padding:
                mask = (original_weight < 1e-2).float()
                this_texture = uv_padding(this_texture[0], mask[0], format="chw")[None]
                this_extras = list(
                    map(
                        lambda x: uv_padding(x[0], mask[0], format="chw")[None],
                        this_extras,
                    )
                )

            weight_map = fused_weights[i].unsqueeze(1)
            texture_fusion += this_texture * weight_map

            for j, extra in enumerate(this_extras):
                extra_textures[j] += extra * weight_map

        if self.use_pbr and pbr:
            return texture_fusion, extra_textures[0], extra_textures[1]
        return texture_fusion

    def empty_texture_cache(self):
        self.texture_list = []

    def texture_postprocess(self, texture_img=None):
        ### DO uv edge padding
        if texture_img is None:
            texture_img = self.texture_img
        texture_img_npy = texture_img.permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        texture_img_npy = (texture_img_npy[0].cpu().detach().numpy() * 255).astype(
            np.uint8
        )

        append_mask_edge = (
            self.postprocess_edge.permute(0, 2, 3, 1).contiguous().clamp(0, 1)
        )
        append_mask_edge = (append_mask_edge[0].cpu().detach().numpy() * 255).astype(
            np.uint8
        )
        texture_img_npy_inpaint = inpaint_atlas(texture_img_npy, append_mask_edge)
        self.texture_img_postprocess = nn.Parameter(
            torch.from_numpy(texture_img_npy_inpaint / 255.0)
            .unsqueeze(0)
            .permute(0, 3, 1, 2)
        )

    def render(
        self,
        theta=None,
        phi=None,
        radius=None,
        use_meta_texture=False,
        render_cache=None,
        dims=None,
        depth_min_val=0.5,
        use_latent=False,
        skip_uncolored_mask=False,
    ):
        if render_cache is None:
            assert theta is not None and phi is not None and radius is not None
        if use_meta_texture:
            texture_img = self.meta_texture_img
        else:
            texture_img = self.texture_img

        if use_latent:
            assert self.use_latent
            texture_img = self.latent_img

        rgb, depth, mask, uncolored_mask, normals, render_cache = (
            self.renderer.render_single_view_texture(
                self.mesh.vertices,
                self.mesh.faces,
                self.face_attributes,
                texture_img,
                theta=theta,
                phi=phi,
                radius=radius,
                render_cache=render_cache,
                dims=dims,
                texture_default_color=self.default_color,
                depth_min_val=depth_min_val,
                skip_uncolored_mask=skip_uncolored_mask,
            )
        )
        if not use_meta_texture and not use_latent:
            rgb = rgb.clamp(0, 1)

        return {
            "image": rgb,
            "mask": mask.detach(),
            "uncolored_mask": uncolored_mask,
            "depth": depth,
            "normals": normals,
            "render_cache": render_cache,
            "texture_map": texture_img,
        }

    def UV_pos_render(self, dims=None):
        if dims is None:
            dims = self.texture_resolution
        UV_pos = self.renderer.UV_pos_render(
            self.mesh.vertices,
            self.mesh.faces,
            self.face_attributes,
            texture_dims=dims,
        )
        return UV_pos

    def UV_normal_render(self, dims=None):
        verts = self.mesh.vertices
        faces = self.mesh.faces
        texture_dims = self.texture_resolution if dims is None else dims
        uv_face_attr = self.face_attributes

        face_vertices = kal.ops.mesh.index_vertices_by_faces(
            self.mesh.vertices[None], self.mesh.faces
        )
        face_normals = kal.ops.mesh.face_normals(face_vertices)
        vertex_normals = trimesh.geometry.mean_vertex_normals(
            vertex_count=verts.size(0),
            faces=faces.cpu(),
            face_normals=face_normals[0].cpu(),
        )  # V,3
        vertex_normals = (
            torch.from_numpy(vertex_normals).unsqueeze(0).float().to(self.device)
        )
        face_vertices_normals = kal.ops.mesh.index_vertices_by_faces(
            vertex_normals, faces
        )

        face_vertices_world = kal.ops.mesh.index_vertices_by_faces(
            verts.unsqueeze(0), faces
        )
        face_vertices_z = torch.zeros_like(
            face_vertices_world[:, :, :, -1], device=self.device
        )

        uv_position, face_idx = kal.render.mesh.rasterize(
            texture_dims[0],
            texture_dims[1],
            face_vertices_z,
            uv_face_attr * 2 - 1,
            face_features=face_vertices_normals,
        )
        uv_position = torch.clamp(uv_position, -1, 1)

        uv_position = uv_position / 2 + 0.5
        uv_position[face_idx == -1] = 0
        return uv_position

    def UV_mask_render(self, dims=None):
        verts = self.mesh.vertices
        faces = self.mesh.faces
        texture_dims = self.texture_resolution if dims is None else dims
        uv_face_attr = self.face_attributes

        face_vertices = kal.ops.mesh.index_vertices_by_faces(
            self.mesh.vertices[None], self.mesh.faces
        )
        face_normals = kal.ops.mesh.face_normals(face_vertices)
        vertex_normals = trimesh.geometry.mean_vertex_normals(
            vertex_count=verts.size(0),
            faces=faces.cpu(),
            face_normals=face_normals[0].cpu(),
        )  # V,3
        vertex_normals = (
            torch.from_numpy(vertex_normals).unsqueeze(0).float().to(self.device)
        )
        face_vertices_normals = kal.ops.mesh.index_vertices_by_faces(
            vertex_normals, faces
        )

        face_vertices_world = kal.ops.mesh.index_vertices_by_faces(
            verts.unsqueeze(0), faces
        )
        face_vertices_z = torch.zeros_like(
            face_vertices_world[:, :, :, -1], device=self.device
        )

        uv_position, face_idx = kal.render.mesh.rasterize(
            texture_dims[0],
            texture_dims[1],
            face_vertices_z,
            uv_face_attr * 2 - 1,
            face_features=torch.ones_like(face_vertices_normals),
        )
        uv_position = torch.clamp(uv_position, 0, 1)
        uv_position[face_idx == -1] = 0

        return uv_position

    @torch.no_grad()
    def render_spiral(self, num_frames, elevation, radius, dims=(512, 512)):
        azimuths = np.linspace(0, np.pi * 2, num_frames, endpoint=False).astype(
            np.float32
        )
        ele = np.deg2rad(90 - elevation).astype(np.float32)
        frames = []

        for azi in azimuths:
            this_rendered = self.render(theta=ele, phi=azi, radius=radius, dims=dims)
            frames.append(
                this_rendered["mask"][0] * this_rendered["image"][0]
                + (1 - this_rendered["mask"][0]) * 1.0
            )

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "t c h w -> t h w c")

        return frames

    @torch.no_grad()
    def export_glb(self, path):
        face_vertices = kal.ops.mesh.index_vertices_by_faces(
            self.mesh.vertices[None], self.mesh.faces
        )
        vn_np = (
            kal.ops.mesh.face_normals(face_vertices).cpu().numpy()[0].astype(np.float32)
        )

        f_np = self.mesh.faces.detach().cpu().numpy().astype(np.uint32)
        v_np = self.mesh.vertices.detach().cpu().numpy().astype(np.float32)

        if hasattr(self, "vmapping"):
            v_np = (
                self.mesh.vertices.detach()
                .cpu()
                .numpy()
                .astype(np.float32)[self.vmapping]
            )
        else:
            v_np = self.mesh.vertices.detach().cpu().numpy().astype(np.float32)
        f_np = self.ft.detach().cpu().numpy().astype(np.uint32)
        vt_np = self.vt.detach().cpu().numpy().astype(np.float32)
        vt_np[..., 1] = 1 - vt_np[..., 1]

        albedo = rearrange(self.texture_img[0], "c h w -> h w c")
        albedo = albedo.detach().cpu().numpy().clip(0, 1)
        albedo = (albedo * 255).astype(np.uint8)
        albedo = cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR)

        f_np_blob = f_np.flatten().tobytes()
        v_np_blob = v_np.tobytes()
        vn_np_blob = vn_np.tobytes()
        vt_np_blob = vt_np.tobytes()
        albedo_blob = cv2.imencode(".png", albedo)[1].tobytes()

        gltf = pygltflib.GLTF2(
            scene=0,
            scenes=[pygltflib.Scene(nodes=[0])],
            nodes=[pygltflib.Node(mesh=0)],
            meshes=[
                pygltflib.Mesh(
                    primitives=[
                        pygltflib.Primitive(
                            # indices to accessors (0 is triangles)
                            attributes=pygltflib.Attributes(
                                POSITION=1,
                                NORMAL=2,
                                TEXCOORD_0=3,
                            ),
                            indices=0,
                            material=0,
                        )
                    ]
                )
            ],
            materials=[
                pygltflib.Material(
                    pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                        baseColorTexture=pygltflib.TextureInfo(index=0, texCoord=0),
                        metallicFactor=0.0,
                        roughnessFactor=1.0,
                    ),
                    alphaCutoff=0,
                    doubleSided=True,
                )
            ],
            textures=[
                pygltflib.Texture(sampler=0, source=0),
            ],
            samplers=[
                pygltflib.Sampler(
                    magFilter=pygltflib.LINEAR,
                    minFilter=pygltflib.LINEAR_MIPMAP_LINEAR,
                    wrapS=pygltflib.REPEAT,
                    wrapT=pygltflib.REPEAT,
                ),
            ],
            images=[
                # use embedded (buffer) image
                pygltflib.Image(bufferView=3, mimeType="image/png"),
            ],
            buffers=[
                pygltflib.Buffer(
                    byteLength=len(f_np_blob)
                    + len(v_np_blob)
                    + len(vn_np_blob)
                    + len(vt_np_blob)
                    + len(albedo_blob)
                )
            ],
            # buffer view (based on dtype)
            bufferViews=[
                # triangles; as flatten (element) array
                pygltflib.BufferView(
                    buffer=0,
                    byteLength=len(f_np_blob),
                    target=pygltflib.ELEMENT_ARRAY_BUFFER,  # GL_ELEMENT_ARRAY_BUFFER (34963)
                ),
                # positions, normals; as vec3 array
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(f_np_blob),
                    byteLength=len(v_np_blob) + len(vn_np_blob),
                    byteStride=12,  # vec3
                    target=pygltflib.ARRAY_BUFFER,  # GL_ARRAY_BUFFER (34962)
                ),
                # texcoords; as vec2 array
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(f_np_blob) + len(v_np_blob) + len(vn_np_blob),
                    byteLength=len(vt_np_blob),
                    byteStride=8,  # vec2
                    target=pygltflib.ARRAY_BUFFER,
                ),
                # texture; as none target
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(f_np_blob)
                    + len(v_np_blob)
                    + len(vn_np_blob)
                    + len(vt_np_blob),
                    byteLength=len(albedo_blob),
                ),
            ],
            accessors=[
                # 0 = triangles
                pygltflib.Accessor(
                    bufferView=0,
                    componentType=pygltflib.UNSIGNED_INT,  # GL_UNSIGNED_INT (5125)
                    count=f_np.size,
                    type=pygltflib.SCALAR,
                    max=[int(f_np.max())],
                    min=[int(f_np.min())],
                ),
                # 1 = positions
                pygltflib.Accessor(
                    bufferView=1,
                    componentType=pygltflib.FLOAT,  # GL_FLOAT (5126)
                    count=len(v_np),
                    type=pygltflib.VEC3,
                    max=v_np.max(axis=0).tolist(),
                    min=v_np.min(axis=0).tolist(),
                ),
                # 2 = normals
                pygltflib.Accessor(
                    bufferView=1,
                    componentType=pygltflib.FLOAT,
                    # byteOffset=len(v_np_blob),
                    count=len(vn_np),
                    type=pygltflib.VEC3,
                    max=vn_np.max(axis=0).tolist(),
                    min=vn_np.min(axis=0).tolist(),
                ),
                # 3 = texcoords
                pygltflib.Accessor(
                    bufferView=2,
                    componentType=pygltflib.FLOAT,
                    count=len(vt_np),
                    type=pygltflib.VEC2,
                    max=vt_np.max(axis=0).tolist(),
                    min=vt_np.min(axis=0).tolist(),
                ),
            ],
        )

        if self.use_pbr:
            metalic_roughness = torch.cat(
                [
                    self.roughness_img.mean(dim=1, keepdim=True),
                    self.roughness_img.mean(dim=1, keepdim=True),
                    self.metalic_img.mean(dim=1, keepdim=True),
                ],
                dim=1,
            )[0]
            metalic_roughness = rearrange(metalic_roughness, "c h w -> h w c")
            metalic_roughness = metalic_roughness.detach().cpu().numpy().clip(0, 1)
            metalic_roughness = (metalic_roughness * 255).astype(np.uint8)
            metalic_roughness = cv2.cvtColor(metalic_roughness, cv2.COLOR_RGB2BGR)
            metalic_roughness_blob = cv2.imencode(".png", metalic_roughness)[
                1
            ].tobytes()

            gltf.materials[0].pbrMetallicRoughness.metallicFactor = 1.0
            gltf.materials[0].pbrMetallicRoughness.roughnessFactor = 1.0
            gltf.materials[0].pbrMetallicRoughness.metallicRoughnessTexture = (
                pygltflib.TextureInfo(index=1, texCoord=0)
            )
            gltf.textures.append(pygltflib.Texture(sampler=1, source=1))

            gltf.samplers.append(
                pygltflib.Sampler(
                    magFilter=pygltflib.LINEAR,
                    minFilter=pygltflib.LINEAR_MIPMAP_LINEAR,
                    wrapS=pygltflib.REPEAT,
                    wrapT=pygltflib.REPEAT,
                )
            )

            gltf.images.append(pygltflib.Image(bufferView=4, mimeType="image/png"))

            byteOffset = (
                len(f_np_blob)
                + len(v_np_blob)
                + len(vn_np_blob)
                + len(vt_np_blob)
                + len(albedo_blob)
            )

            gltf.bufferViews.append(
                # index = 4, metallicRoughness texture; as none target
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=byteOffset,
                    byteLength=len(metalic_roughness_blob),
                )
            )

        # set actual data
        if not self.use_pbr:
            gltf.set_binary_blob(
                f_np_blob + v_np_blob + vn_np_blob + vt_np_blob + albedo_blob
            )
        else:
            gltf.set_binary_blob(
                f_np_blob
                + v_np_blob
                + vn_np_blob
                + vt_np_blob
                + albedo_blob
                + metalic_roughness_blob
            )

        # glb = b"".join(gltf.save_to_bytes())
        gltf.save(path)

    def forward_texturing_zero123pp(
        self,
        zero123pp_6views,
        radius,
        sr_model=None,
        this_exp_dir=None,
        verbose=False,
        **kwargs,
    ):
        azimuths = np.deg2rad(np.array([30, 90, 150, 210, 270, 330])).astype(np.float32)
        elevations = (
            -np.deg2rad(np.array([20, -10, 20, -10, 20, -10])) + np.pi / 2
        ).astype(np.float32)
        zero123pp_6views = np.asarray(zero123pp_6views)

        zero123pp_6views = rearrange(
            zero123pp_6views, "(a h) (b w) c -> (a b) h w c", a=3, b=2
        )

        for view_idx, (azi, ele, target) in enumerate(
            zip(azimuths, elevations, zero123pp_6views)
        ):
            if sr_model is not None:
                target = sr_model(target)
            target = target.astype(np.float32) / 255.0
            target = torch.from_numpy(target).to(self.device)
            target = rearrange(target, "h w c -> c h w")
            target = F.interpolate(target[None], self.renderer.dims, mode="bilinear")
            self.forward_texturing(
                theta=ele,
                phi=azi,
                view_target=target,
                radius=radius,
                view_id=view_idx + 1,
                save_result_dir=this_exp_dir,
                verbose=verbose,
                **kwargs,
            )

    def render_zero123pp_6views_latents(
        self, radius, reso=(40, 40), interpolation_mode="nearest"
    ):
        # reso = (320, 320)
        # reso = (40, 40)
        azimuths = np.deg2rad(np.array([30, 90, 150, 210, 270, 330])).astype(np.float32)
        elevations = (
            -np.deg2rad(np.array([20, -10, 20, -10, 20, -10])) + np.pi / 2
        ).astype(np.float32)

        latents = []
        masks = []
        move_axis = lambda x: rearrange(x, "c h w -> h w c")
        for azi, ele in zip(azimuths, elevations):
            rendered = self.render(
                ele,
                azi,
                radius,
                dims=reso,
                depth_min_val="raw",
                use_latent=True,
                skip_uncolored_mask=True,
            )
            latents.append(move_axis(rendered["image"][0]))
            masks.append(move_axis(rendered["mask"][0]))

        masks = torch.stack(masks, dim=0)
        latents = torch.stack(latents, dim=0)

        [latents, masks] = [
            rearrange(img, "(a b) h w c -> c (a h) (b w)", a=3, b=2)
            for img in [latents, masks]
        ]

        if reso != (40, 40):
            [latents, masks] = [
                F.interpolate(img[None], (120, 80), mode=interpolation_mode)[0]
                for img in [latents, masks]
            ]

        return latents, masks

    def backproject_zero123pp_6views_latents(
        self, latents, radius, this_exp_dir=None, verbose=False, fusion_method="softmax"
    ):
        azimuths = np.deg2rad(np.array([30, 90, 150, 210, 270, 330])).astype(np.float32)
        elevations = (
            -np.deg2rad(np.array([20, -10, 20, -10, 20, -10])) + np.pi / 2
        ).astype(np.float32)

        if latents.dim() == 4:
            latents = latents[0]

        latents = rearrange(latents, "c (a h) (b w) -> (a b) c h w", a=3, b=2).to(
            torch.float32
        )

        latent_textures = []

        for view_idx, (azi, ele, target) in enumerate(
            zip(azimuths, elevations, latents)
        ):
            # target = F.interpolate(target[None], self.renderer.dims, mode="bilinear")
            ret = self.forward_texturing(
                theta=ele,
                phi=azi,
                view_target=target[None],
                radius=radius,
                view_id=view_idx + 1,
                dims=target.shape[-2:],
                save_result_dir=this_exp_dir,
                verbose=verbose,
                skip_texture_update=True,
                # interpolation_mode="nearest",
                interpolation_mode="bilinear",
            )
            # ret: texture_map, mask, weight
            latent_textures.append(ret)

        weights = (
            torch.zeros(1, 1, *self.latent_img.shape[-2:], device=self.device) + 1e-5
        )
        latent_img = torch.zeros_like(self.latent_img)

        if fusion_method == "vanilla":
            for r in latent_textures:
                # mask = r[1].mean(dim=1, keepdim=True) == 1.0
                mask = r[1]
                # mask = dilate_mask(mask[0], 20, format="chw")[None]
                # latent_img[mask] += r[0][mask]
                latent_img += r[0] * mask.mean(1, keepdim=True)
                weights += mask.mean(1)
        elif fusion_method == "l6":
            # print("here")
            for r in latent_textures:
                # mask = r[1].mean(dim=1, keepdim=True) == 1.0
                # mask = r[1]
                weight = r[2]
                # weight = dilate_mask(weight[0], 20, format="chw")[None]
                weight = weight**6
                latent_img += r[0] * weight
                weights += weight
        elif fusion_method == "softmax":
            weight_maps = torch.cat([r[2] for r in latent_textures], dim=1)
            fused_weights = torch.unbind(torch.softmax(weight_maps * 10, dim=1), dim=1)
            for r, weight in zip(latent_textures, fused_weights):
                latent_img += r[0] * weight
                weights += weight
        elif isinstance(fusion_method, (float, int)):
            for r in latent_textures:
                # mask = r[1].mean(dim=1, keepdim=True) == 1.0
                # mask = r[1]
                weight = r[2]
                # weight = dilate_mask(weight[0], 20, format="chw")[None]
                weight = weight**fusion_method
                latent_img += r[0] * weight
                weights += weight
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        # breakpoint()
        latent_img /= weights
        # breakpoint()
        self.latent_img.data = latent_img
        write_image("trash/latent_img.png", latent_img[0, :3], format="chw")

    @torch.inference_mode()
    def render_experiment_spirals(self, radius=4.5):
        frames = self.render_spiral(90, 10, radius)
        frames = [frames]

        for fuse_m in [
            "softmax",
        ]:
            fused_texture = self.texture_fusion(fuse_m)
            self.texture_img.data = fused_texture
            frames.append(self.render_spiral(90, 10, radius))

        for fuse_m in [
            "softmax",
        ]:
            fused_texture = self.texture_fusion(fuse_m, use_masked=True)
            self.texture_img.data = fused_texture
            frames.append(self.render_spiral(90, 10, radius))

        return torch.cat(frames, dim=2)

    @contextmanager
    def texture_scope(
        self, temp_texture_img=None, temp_metalic_img=None, temp_roughness_img=None
    ):
        original_texture_img = self.texture_img.data.clone()
        original_texture_list = self.texture_list.copy()
        self.texture_list = []
        if temp_metalic_img is not None:
            original_metalic_img = self.metalic_img.data.clone()
            self.metalic_img.data = temp_metalic_img.to(self.metalic_img.data)
        if temp_roughness_img is not None:
            original_roughness_img = self.roughness_img.data.clone()
            self.roughness_img.data = temp_roughness_img.to(self.roughness_img.data)
        if temp_texture_img is None:
            temp_texture_img = torch.ones(1, 3, *self.texture_resolution).to(
                self.device
            ) * torch.Tensor(self.default_color).reshape(1, 3, 1, 1).to(self.device)
        self.texture_img.data = temp_texture_img.to(self.texture_img.data)

        try:
            yield None
        finally:
            self.texture_img.data = original_texture_img
            self.texture_list = original_texture_list
            if temp_metalic_img is not None:
                self.metalic_img.data = original_metalic_img
            if temp_roughness_img is not None:
                self.roughness_img.data = original_roughness_img

    def uncolored_texture_map(self):
        weight_maps = []
        for i in range(len(self.texture_list)):
            _, texture, weight_map, *_ = self.texture_list[i]
            weight_maps.append(weight_map)

        weight_maps = torch.stack(weight_maps, dim=0).sum(0)

        return weight_maps < 1e-2

    @torch.no_grad()
    def postprocess_texture_map(
        self, upsample=True, uv_edge_padding=True, upsampler=None
    ):
        if upsample:
            tex_reso = self.texture_img.data.shape[-1]
            tex = to_pil_image(self.texture_img.data[0]).resize(
                (tex_reso // 4, tex_reso // 4)
            )
            up_tex = Image.fromarray(upsampler(tex))
            self.texture_img.data = to_tensor(up_tex).unsqueeze(0).to(self.device)

        if uv_edge_padding:
            UV_mask = 1 - self.UV_mask_render(self.texture_img.shape[-2:])[0]
            UV_mask = rearrange(UV_mask, "h w c -> c h w").mean(0, keepdim=True)
            self.texture_img.data = uv_padding(
                self.texture_img.data[0], UV_mask, format="chw"
            )[None]
