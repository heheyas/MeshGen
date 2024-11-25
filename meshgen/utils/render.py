import os
import numpy as np
import torch
import trimesh
from trimesh.transformations import rotation_matrix
import pyrender
import open3d as o3d
from einops import rearrange
from open3d.visualization import rendering
from open3d.visualization.rendering import OffscreenRenderer, MaterialRecord

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

# from meshgen.utils.timer import tic, toc, enable_timing

_render = None


def get_uniform_campos(num_frames, radius, elevation):
    T = num_frames
    azimuths = np.deg2rad(np.linspace(0, 360, T + 1)[:T])
    elevations = np.full_like(azimuths, np.deg2rad(elevation))
    cam_dists = np.full_like(azimuths, radius)

    campos = np.stack(
        [
            cam_dists * np.cos(elevations) * np.cos(azimuths),
            cam_dists * np.cos(elevations) * np.sin(azimuths),
            cam_dists * np.sin(elevations),
        ],
        axis=-1,
    )

    return campos


def init_renderer(reso=512, fov=50.0, force_new_render=False):
    if not force_new_render:
        global _render
        if _render is not None:
            return _render
    _render = OffscreenRenderer(reso, reso)
    _render.scene.set_background([1, 1, 1, 1])
    bottom_plane = trimesh.creation.box([100, 100, 0.01])
    bottom_plane.apply_translation([0, 0, -0.55])
    bottom_plane = bottom_plane.as_open3d
    bottom_plane.compute_vertex_normals()
    bottom_plane.paint_uniform_color([1, 1, 1])
    plane_mat = MaterialRecord()
    plane_mat.base_color = [1, 1, 1, 1]
    plane_mat.shader = "defaultLit"
    _render.scene.add_geometry("bottom_plane", bottom_plane, plane_mat)

    _render.scene.set_lighting(
        _render.scene.LightingProfile.MED_SHADOWS, (0.577, -0.577, -0.577)
    )
    _render.scene.set_lighting(
        _render.scene.LightingProfile.MED_SHADOWS, (-0.577, -0.577, -0.577)
    )
    old_camera = _render.scene.camera
    _render.scene.camera.set_projection(
        fov,
        1,
        old_camera.get_near(),
        old_camera.get_far(),
        old_camera.get_field_of_view_type(),
    )

    return _render


@torch.no_grad()
def render_mesh_spiral_offscreen(
    vertices,
    faces,
    reso=512,
    num_frames=90,
    elevation=0,
    radius=2.0,
    normalize=True,
    rotate=True,
    color=None,
    fov=50.0,
    force_new_render=False,
):
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.cpu().numpy()
        faces = faces.cpu().numpy()

    if len(vertices) == 0:
        return np.ones((num_frames, reso, reso, 3), dtype=np.uint8) * 255

    mesh = trimesh.Trimesh(vertices, faces)
    if normalize:
        mesh.apply_translation(-mesh.centroid)
        scale = max(mesh.extents)
        scale_T = np.eye(4)
        scale_T[0, 0] = scale_T[1, 1] = scale_T[2, 2] = 1.0 / scale
        mesh.apply_transform(scale_T)

    if rotate:
        yz_rotation = rotation_matrix(np.pi / 2, [1, 0, 0])
        mesh.apply_transform(yz_rotation)

    # init_renderer(reso, fov)
    render = init_renderer(reso, fov, force_new_render=force_new_render)

    # global _render
    # render = _render

    # color = np.array([58, 75, 101], dtype=np.float64) / 255
    if color is None:
        color = np.array([0.034, 0.294, 0.5], dtype=np.float64) * 1.2
    # color = np.array([144.0 / 255, 210.0 / 255, 236.0 / 255], dtype=np.float64)

    mesh = mesh.as_open3d
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    mat = MaterialRecord()
    mat.base_color = [1, 1, 1, 1]
    mat.base_metallic = 0.0
    mat.base_roughness = 1.0
    mat.shader = "defaultLit"
    render.scene.add_geometry("model", mesh, mat)

    campos = get_uniform_campos(num_frames, radius, elevation)

    # render = OffscreenRenderer(reso, reso)

    # render.scene.set_background([1, 1, 1, 1])
    # render.scene.add_geometry("model", mesh, mat)

    # bottom_plane = trimesh.creation.box([100, 100, 0.01])
    # bottom_plane.apply_translation([0, 0, -0.55])
    # bottom_plane = bottom_plane.as_open3d
    # bottom_plane.compute_vertex_normals()
    # bottom_plane.paint_uniform_color([1, 1, 1])
    # plane_mat = MaterialRecord()
    # plane_mat.base_color = [1, 1, 1, 1]
    # plane_mat.shader = "defaultLit"
    # render.scene.add_geometry("bottom_plane", bottom_plane, plane_mat)

    # # render.scene.scene.enable_sun_light(False)
    # light_dir = np.array([1, 1, 1])
    # # render.scene.scene.add_spot_light(
    # #     "light", [1, 1, 1], -3 * light_dir, light_dir, 1e8, 1e2, 0.1, 0.1, True
    # # )

    # render.scene.set_lighting(
    #     render.scene.LightingProfile.MED_SHADOWS, (0.577, -0.577, -0.577)
    # )

    frames = []
    render.scene.camera.look_at([0, 0, 0], campos[0], [0, 0, 1])
    for i in range(num_frames):
        azimuth = i / num_frames * 2 * np.pi
        render.scene.set_geometry_transform(
            "model", rotation_matrix(azimuth, [0, 0, 1])
        )
        frame = np.asarray(render.render_to_image())
        frames.append(frame)

    render.scene.remove_geometry("model")
    return np.stack(frames, axis=0)


def render_point_cloud_spiral_offscreen(
    vertices,
    reso=512,
    num_frames=90,
    elevation=0,
    radius=2.0,
    point_radius=0.01,
    rotate=True,
    **kwargs,
):
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.cpu().numpy()

    if rotate:
        yz_rotation = rotation_matrix(np.pi / 2, [1, 0, 0])
        vertices = np.dot(vertices, yz_rotation[:3, :3].T)

    colors = np.array([125, 151, 250, 255]) / 255
    colors = np.tile(colors, (vertices.shape[0], 1))

    sm = []
    for v in vertices:
        this_point = trimesh.creation.uv_sphere(radius=point_radius)
        this_point.apply_translation(v)
        sm.append(this_point)

    sm = trimesh.util.concatenate(sm)

    return render_mesh_spiral_offscreen(
        sm.vertices, sm.faces, reso, num_frames, elevation, radius, **kwargs
    )


import torch
import torch.nn.functional as F
import numpy as np


def pad_camera_extrinsics_4x4(extrinsics):
    if extrinsics.shape[-2] == 4:
        return extrinsics
    padding = torch.tensor([[0, 0, 0, 1]]).to(extrinsics)
    if extrinsics.ndim == 3:
        padding = padding.unsqueeze(0).repeat(extrinsics.shape[0], 1, 1)
    extrinsics = torch.cat([extrinsics, padding], dim=-2)
    return extrinsics


def center_looking_at_camera_pose(
    camera_position: torch.Tensor,
    look_at: torch.Tensor = None,
    up_world: torch.Tensor = None,
):
    """
    Create OpenGL camera extrinsics from camera locations and look-at position.

    camera_position: (M, 3) or (3,)
    look_at: (3)
    up_world: (3)
    return: (M, 3, 4) or (3, 4)
    """
    # by default, looking at the origin and world up is z-axis
    if look_at is None:
        look_at = torch.tensor([0, 0, 0], dtype=torch.float32)
    if up_world is None:
        up_world = torch.tensor([0, 0, 1], dtype=torch.float32)
    if camera_position.ndim == 2:
        look_at = look_at.unsqueeze(0).repeat(camera_position.shape[0], 1)
        up_world = up_world.unsqueeze(0).repeat(camera_position.shape[0], 1)

    # OpenGL camera: z-backward, x-right, y-up
    z_axis = camera_position - look_at
    z_axis = F.normalize(z_axis, dim=-1).float()
    x_axis = torch.linalg.cross(up_world, z_axis, dim=-1)
    x_axis = F.normalize(x_axis, dim=-1).float()
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    y_axis = F.normalize(y_axis, dim=-1).float()

    extrinsics = torch.stack([x_axis, y_axis, z_axis, camera_position], dim=-1)
    extrinsics = pad_camera_extrinsics_4x4(extrinsics)
    return extrinsics


def spherical_camera_pose(azimuths: np.ndarray, elevations: np.ndarray, radius=2.5):
    azimuths = np.deg2rad(azimuths)
    elevations = np.deg2rad(elevations)

    xs = radius * np.cos(elevations) * np.cos(azimuths)
    ys = radius * np.cos(elevations) * np.sin(azimuths)
    zs = radius * np.sin(elevations)

    cam_locations = np.stack([xs, ys, zs], axis=-1)
    cam_locations = torch.from_numpy(cam_locations).float()

    c2ws = center_looking_at_camera_pose(cam_locations)
    return c2ws


def get_circular_camera_poses(M=120, radius=2.5, elevation=30.0):
    # M: number of circular views
    # radius: camera dist to center
    # elevation: elevation degrees of the camera
    # return: (M, 4, 4)
    assert M > 0 and radius > 0

    elevation = np.deg2rad(elevation)

    camera_positions = []
    for i in range(M):
        azimuth = 2 * np.pi * i / M
        x = radius * np.cos(elevation) * np.cos(azimuth)
        y = radius * np.cos(elevation) * np.sin(azimuth)
        z = radius * np.sin(elevation)
        camera_positions.append([x, y, z])
    camera_positions = np.array(camera_positions)
    camera_positions = torch.from_numpy(camera_positions).float()
    extrinsics = center_looking_at_camera_pose(camera_positions)
    return extrinsics


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


def get_render_cameras(
    batch_size=1, M=120, radius=2.0, elevation=20.0, is_flexicubes=False
):
    """
    Get the rendering camera parameters.
    """
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = (
            FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        )
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def render_instant_frames(
    model,
    planes,
    reso=512,
    num_frames=90,
    elevation=0,
    radius=2.0,
    chunk_size=1,
    is_flexicubes=False,
    **kwargs,
):
    """
    Render frames from triplanes.
    """
    render_size = reso
    render_cameras = get_render_cameras(
        1, num_frames, radius, elevation, is_flexicubes
    ).to(planes)
    frames = []
    for i in range(0, render_cameras.shape[1], chunk_size):
        if is_flexicubes:
            frame = model.forward_geometry(
                planes,
                render_cameras[:, i : i + chunk_size],
                render_size=render_size,
            )["img"]
        else:
            frame = model.forward_synthesizer(
                planes,
                render_cameras[:, i : i + chunk_size],
                render_size=render_size,
            )["images_rgb"]
        frames.append(frame)

    frames = torch.cat(frames, dim=1)[0]  # we suppose batch size is always 1
    frames = frames.cpu().numpy()
    frames = rearrange(frames, "T C H W -> T H W C")
    return frames
