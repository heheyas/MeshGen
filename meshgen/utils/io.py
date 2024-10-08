import os
import numpy as np
import torch
import trimesh
import kaolin
import mediapy
from PIL import Image
from einops import rearrange
from pathlib import Path


def load_mesh(filename, device="cpu"):
    try:
        filename = str(filename)
        if filename.endswith(".obj"):
            mesh = kaolin.io.obj.import_mesh(filename)
        elif filename.endswith(".glb"):
            mesh = kaolin.io.gltf.import_mesh(filename)
        elif filename.endswith(".off"):
            mesh = kaolin.io.off.import_mesh(filename)
        else:
            raise NotImplementedError(f"Unsupported file format: {filename}")
        vertices = mesh.vertices
        faces = mesh.faces
    except kaolin.io.utils.NonHomogeneousMeshError:
        mesh = trimesh.load(filename)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        vertices = torch.from_numpy(mesh.vertices).float()
        faces = torch.from_numpy(mesh.faces).long()

    if device != "np":
        return vertices.to(device), faces.to(device)
    else:
        return vertices.cpu().numpy(), faces.cpu().numpy()


# def normalize_mesh(vertices, faces, band=1 / 256):
#     input_np = False
#     if isinstance(vertices, np.ndarray):
#         input_np = True
#         vertices = torch.from_numpy(vertices)
#         faces = torch.from_numpy(faces)
#     tris = vertices[faces]
#     a = tris.min(0)[0].min(0)[0]
#     vertices -= a
#     tris -= a
#     vertices = (vertices / tris.max() + band) / (1 + band * 2)
#     vertices -= 0.5

#     if input_np:
#         vertices = vertices.numpy()
#         faces = faces.numpy()

#     return vertices, faces


def normalize_mesh(vertices, faces, target_scale=0.55):
    offset = (vertices.min(dim=0)[0] + vertices.max(dim=0)[0]) / 2
    vertices -= offset
    scale_factor = 2.0 / (vertices.max(dim=0)[0] - vertices.min(dim=0)[0]).max()
    vertices *= scale_factor
    vertices *= target_scale

    return vertices, faces


def export_mesh(vertices, faces, filename, centralize=True, normalize=True):
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()

    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()

    Path(filename).parent.mkdir(exist_ok=True, parents=True)

    mesh = trimesh.Trimesh(vertices, faces)
    if centralize:
        mesh.vertices -= mesh.centroid
    if normalize:
        mesh.vertices /= mesh.extents.max()
    mesh.export(filename)


def write_video(filename, frames, **kwargs):
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().numpy()

    if frames.dtype == np.float32:
        frames = (frames * 255).clip(0, 255).astype(np.uint8)

    Path(filename).parent.mkdir(exist_ok=True, parents=True)
    mediapy.write_video(filename, frames, **kwargs)


def write_image(filename, image, format="hwc", **kwargs):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if format == "chw":
        image = rearrange(image, "c h w -> h w c")
    Path(filename).parent.mkdir(exist_ok=True, parents=True)
    mediapy.write_image(filename, image, **kwargs)


def read_image(filename, normalize=False, chw=False, return_pt=False, device="cpu"):
    image = mediapy.read_image(filename)

    if normalize:
        image = image.astype(np.float32) / 255

    if chw:
        image = rearrange(image, "h w c -> c h w")

    if return_pt:
        image = torch.from_numpy(image).to(device)

    return image


def save_tensor_image(tensor: torch.Tensor, save_path: str):
    if len(os.path.dirname(save_path)) > 0 and not os.path.exists(
        os.path.dirname(save_path)
    ):
        os.makedirs(os.path.dirname(save_path))
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)  # [1, c, h, w]-->[c, h, w]
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    tensor = tensor.permute(1, 2, 0).detach().cpu().numpy()  # [c, h, w]-->[h, w, c]
    Image.fromarray((tensor * 255).astype(np.uint8)).save(save_path)
