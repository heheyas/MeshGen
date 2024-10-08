import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler
import numpy as np
from einops import rearrange, repeat, einsum

from .math_utils import linspace, get_ray_limits_box


def FOV_to_intrinsics(fov, device="cpu"):
    """
    Creates a 3x3 camera intrinsics matrix from the camera field of view, specified in degrees.
    Note the intrinsics are returned as normalized by image size, rather than in pixel units.
    Assumes principal point is at image center.
    """
    focal_length = 0.5 / np.tan(np.deg2rad(fov) * 0.5)
    intrinsics = torch.tensor(
        [[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device
    )
    return intrinsics


class RayGenerator(torch.nn.Module):
    """
    from camera pose and intrinsics to ray origins and directions
    """

    def __init__(self):
        super().__init__()
        (
            self.ray_origins_h,
            self.ray_directions,
            self.depths,
            self.image_coords,
            self.rendering_options,
        ) = (None, None, None, None, None)

    def forward(self, cam2world_matrix, fov, render_size):
        """
        Create batches of rays and return origins and directions.

        cam2world_matrix: (N, 4, 4)
        intrinsics: (N, 3, 3)
        render_size: int

        ray_origins: (N, M, 3)
        ray_dirs: (N, M, 2)
        """
        intrinsics = (
            FOV_to_intrinsics(fov)
            .to(cam2world_matrix)[None]
            .repeat(cam2world_matrix.shape[0], 1, 1)
        )

        N, M = cam2world_matrix.shape[0], render_size**2
        cam_locs_world = cam2world_matrix[:, :3, 3]
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]
        sk = intrinsics[:, 0, 1]

        uv = torch.stack(
            torch.meshgrid(
                torch.arange(
                    render_size, dtype=torch.float32, device=cam2world_matrix.device
                ),
                torch.arange(
                    render_size, dtype=torch.float32, device=cam2world_matrix.device
                ),
                indexing="ij",
            )
        )
        uv = uv.flip(0).reshape(2, -1).transpose(1, 0)
        uv = uv.unsqueeze(0).repeat(cam2world_matrix.shape[0], 1, 1)

        x_cam = uv[:, :, 0].view(N, -1) * (1.0 / render_size) + (0.5 / render_size)
        y_cam = uv[:, :, 1].view(N, -1) * (1.0 / render_size) + (0.5 / render_size)
        z_cam = torch.ones((N, M), device=cam2world_matrix.device)

        x_lift = (
            (
                x_cam
                - cx.unsqueeze(-1)
                + cy.unsqueeze(-1) * sk.unsqueeze(-1) / fy.unsqueeze(-1)
                - sk.unsqueeze(-1) * y_cam / fy.unsqueeze(-1)
            )
            / fx.unsqueeze(-1)
            * z_cam
        )
        y_lift = (y_cam - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z_cam

        cam_rel_points = torch.stack(
            (x_lift, y_lift, z_cam, torch.ones_like(z_cam)), dim=-1
        )

        # NOTE: this should be named _blender2opencv
        _opencv2blender = (
            torch.tensor(
                [
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1],
                ],
                dtype=torch.float32,
                device=cam2world_matrix.device,
            )
            .unsqueeze(0)
            .repeat(N, 1, 1)
        )

        cam2world_matrix = torch.bmm(cam2world_matrix, _opencv2blender)

        world_rel_points = torch.bmm(
            cam2world_matrix, cam_rel_points.permute(0, 2, 1)
        ).permute(0, 2, 1)[:, :, :3]

        ray_dirs = world_rel_points - cam_locs_world[:, None, :]
        ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=2)

        ray_origins = cam_locs_world.unsqueeze(1).repeat(1, ray_dirs.shape[1], 1)

        return ray_origins, ray_dirs


class RaySampler(torch.nn.Module):
    def __init__(
        self,
        num_samples_per_ray,
        bbox_length=1.0,
        near=0.1,
        far=10.0,
        drop_invalid=False,
        disparity=False,
    ):
        super().__init__()
        self.num_samples_per_ray = num_samples_per_ray
        self.bbox_length = bbox_length
        self.near = near
        self.far = far
        self.disparity = disparity
        self.drop_invalid = drop_invalid

    def forward(self, ray_origins, ray_directions):
        if not self.disparity:
            t_start, t_end = get_ray_limits_box(
                ray_origins, ray_directions, self.bbox_length
            )
        else:
            t_start = torch.full_like(ray_origins, self.near)
            t_end = torch.full_like(ray_origins, self.far)
        is_ray_valid = t_end > t_start
        if not self.drop_invalid:
            if torch.any(is_ray_valid).item():
                t_start[~is_ray_valid] = t_start[is_ray_valid].min()
                t_end[~is_ray_valid] = t_start[is_ray_valid].max()
        else:
            is_ray_valid = is_ray_valid[..., 0]
            ray_origins = ray_origins[is_ray_valid]
            ray_directions = ray_directions[is_ray_valid]
            t_start = t_start[is_ray_valid]
            t_end = t_end[is_ray_valid]

        if not self.disparity:
            depths = linspace(t_start, t_end, self.num_samples_per_ray)
            depths += (
                torch.rand_like(depths)
                * (t_end - t_start)
                / (self.num_samples_per_ray - 1)
            )
        else:
            step = 1.0 / self.num_samples_per_ray
            z_steps = torch.linspace(
                0, 1 - step, self.num_samples_per_ray, device=ray_origins.device
            )
            z_steps += torch.rand_like(z_steps) * step
            depths = 1 / (1 / self.near * (1 - z_steps) + 1 / self.far * z_steps)
            depths = depths[..., None, None, None]

        return ray_origins[None] + ray_directions[None] * depths
