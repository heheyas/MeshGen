import torch
import numpy as np
from pathlib import Path
from trimesh.transformations import rotation_matrix
from einops import repeat, rearrange

from meshgen.modules.mesh.render import Renderer
from meshgen.utils.io import load_mesh


def make_zero123pp_grid(data):
    masks = torch.stack(data["masks"], dim=0)
    depths = torch.stack(data["depths"], dim=0)
    normals = torch.stack(data["normals"], dim=0)

    depths = repeat(depths, "b h w 1 -> b h w c", c=3)
    depths = torch.cat([depths, masks], dim=-1)
    normals = torch.cat([normals, masks], dim=-1)

    [depths, normals, masks] = [
        rearrange(img, "(a b) h w c -> (a h) (b w) c", a=3, b=2)
        for img in [depths, normals, masks]
    ]

    ret = dict(depths=depths, normals=normals, masks=masks)

    if "rgbs" in data:
        rgbs = torch.stack(data["rgbs"], dim=0)
        rgbs = rearrange(rgbs, "(a b) h w c -> (a h) (b w) c", a=3, b=2)
        # rgbs = torch.cat([rgbs, masks], dim=-1)
        ret["rgbs"] = rgbs

    return ret


def normalize_vertices_to_cube(vertices: torch.Tensor, scale: float = 1.0):  # V,3
    """shift and resize mesh to fit into a unit cube"""
    offset = (vertices.min(dim=0)[0] + vertices.max(dim=0)[0]) / 2
    vertices -= offset
    scale_factor = 2.0 / (vertices.max(dim=0)[0] - vertices.min(dim=0)[0]).max()
    vertices *= scale_factor
    return vertices * scale


def render_orthogonal_4views(
    mesh_filename,
    reso,
    radius=2.5,
    rotate=False,
    depth_min_val=0.5,
    renderer=None,
):
    normals = []
    depths = []
    masks = []
    device = "cuda"

    if isinstance(mesh_filename, str):
        mesh = load_mesh(mesh_filename, device)
        mesh.vertices = normalize_vertices_to_cube(mesh.vertices)
        if rotate:
            rotmat = rotation_matrix(-np.pi / 2, [1, 0, 0]) @ rotation_matrix(
                -np.pi / 2, [0, 0, 1]
            )
            rotmat = torch.from_numpy(rotmat.astype(np.float32)).to(device)[:3, :3]
            mesh.vertices = mesh.vertices @ rotmat.T
    else:
        mesh = mesh_filename

    if renderer is None:
        renderer = Renderer(mesh.faces.shape[0], device)

    azimuths = [0, np.pi / 2, np.pi, np.pi * 1.5]
    for azi in azimuths:
        rendered = renderer.render_single_view(
            mesh.vertices,
            mesh.faces,
            np.pi / 2,
            azi,
            radius,
            dims=(reso, reso),
            depth_min_val=depth_min_val,
        )
        this_normal = rendered["normal_map"][0] * 0.5 + 0.5
        normals.append(this_normal)
        depths.append(rendered["depth_map"][0])
        masks.append(rendered["mask"][0])

    data = dict()
    data["normals"] = normals
    data["depths"] = depths
    data["masks"] = masks

    return data


def render_zero123pp_6views(
    mesh_filename,
    reso=320,
    radius=2.5,
    rotate=True,
    renderer=None,
    depth_min_val=0.0,
):
    normals = []
    depths = []
    masks = []
    device = "cuda"
    if isinstance(mesh_filename, str):
        mesh = load_mesh(mesh_filename, device)
        mesh.vertices = normalize_vertices_to_cube(mesh.vertices)
        if rotate:
            rotmat = rotation_matrix(-np.pi / 2, [1, 0, 0]) @ rotation_matrix(
                -np.pi / 2, [0, 0, 1]
            )
            rotmat = torch.from_numpy(rotmat.astype(np.float32)).to(device)[:3, :3]
            mesh.vertices = mesh.vertices @ rotmat.T
    else:
        mesh = mesh_filename

    if renderer is None:
        renderer = Renderer(mesh.faces.shape[0], device)
    azimuths = np.deg2rad(np.array([30, 90, 150, 210, 270, 330])).astype(np.float32)
    elevations = (
        -np.deg2rad(np.array([20, -10, 20, -10, 20, -10])) + np.pi / 2
    ).astype(np.float32)

    for azi, ele in zip(azimuths, elevations):
        rendered = renderer.render_single_view(
            mesh.vertices,
            mesh.faces,
            ele,
            azi,
            radius,
            depth_min_val=depth_min_val,
            dims=(reso, reso),
        )
        this_normal = rendered["normal_map"][0] * 0.5 + 0.5
        normals.append(this_normal)
        depths.append(rendered["depth_map"][0])
        masks.append(rendered["mask"][0])

    data = dict()
    data["normals"] = normals
    data["depths"] = depths
    data["masks"] = masks
    zero123pp_grid = make_zero123pp_grid(data)

    return zero123pp_grid


@torch.no_grad()
def render_zero123pp_6views_rgbs(
    mesh,
    reso=320,
    radius=2.5,
    bg="white",
    version="1.2",
    renderer=None,
    depth_min_val=0.0,
    flip_normals=False,
):
    normals = []
    depths = []
    masks = []
    rgbs = []

    if version == "1.2":
        azimuths = np.deg2rad(np.array([30, 90, 150, 210, 270, 330])).astype(np.float32)
        elevations = (
            -np.deg2rad(np.array([20, -10, 20, -10, 20, -10])) + np.pi / 2
        ).astype(np.float32)
    else:
        azimuths = np.deg2rad(np.array([30, 90, 150, 210, 270, 330])).astype(np.float32)
        elevations = (
            -np.deg2rad(np.array([30, -20, 30, -20, 30, -20])) + np.pi / 2
        ).astype(np.float32)

    move_axis = lambda x: rearrange(x, "c h w -> h w c")
    bg = 1.0 if bg == "white" else 0.5
    for azi, ele in zip(azimuths, elevations):
        rendered = mesh.render(
            ele, azi, radius, dims=(reso, reso), depth_min_val=depth_min_val
        )
        # grey background
        rgbs.append(
            move_axis(
                rendered["mask"][0] * rendered["image"][0]
                + (1 - rendered["mask"][0]) * bg
            )
        )
        if flip_normals:
            rendered["normals"][0] = -rendered["normals"][0]
        this_normal = rendered["normals"][0] * 0.5 + 0.5
        normals.append(move_axis(this_normal))
        depths.append(move_axis(rendered["depth"][0]))
        masks.append(move_axis(rendered["mask"][0]))

    data = dict()
    data["normals"] = normals
    data["depths"] = depths
    data["masks"] = masks
    data["rgbs"] = rgbs
    zero123pp_grid = make_zero123pp_grid(data)

    return zero123pp_grid
