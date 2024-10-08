import os
import numpy as np
import contextlib
import io
import sys
import torch
import tempfile
import pyacvd
import pyvista as pv

from dataclasses import dataclass
from vedo import Mesh
import trimesh
from meshgen.utils.io import load_mesh


class Vertex:
    @dataclass
    class _V:
        x: float
        y: float
        z: float

    def __init__(self, x: float, y: float, z: float):
        self.co = self._V(float(x), float(y), float(z))


class Face:
    @dataclass
    class _F:
        vertices: list[int]

        def __len__(self):
            return len(self.vertices)

        def __getitem__(self, key):
            return self.vertices[key]

    def __init__(self, vertices) -> None:
        self.vertices = self._F(vertices)


def vf_from_np(vertices, faces):
    blender_V = []
    blender_F = []
    for vv in vertices:
        blender_V.append(Vertex(vv[0], vv[1], vv[2]))

    for ff in faces:
        blender_F.append(Face([int(f_) for f_ in ff.tolist()]))

    return blender_V, blender_F


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout


class suppress_stdout_stderr(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def auto_remesh(vertices, faces, triangulate=True, density=0.0, scaling=2.0):
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.cpu().numpy()
        faces = faces.cpu().numpy()

    vv, ff = vf_from_np(vertices, faces)
    new_v, new_f = generate_quad_mesh(vv, ff, density, scaling, 0)
    # with suppress_stdout_stderr():
    #     new_v, new_f = generate_quad_mesh(vv, ff, density, scaling, 0)
    vedo_mesh = Mesh([new_v, new_f])
    if triangulate:
        vedo_mesh.triangulate()

    return vedo_mesh.vertices, vedo_mesh.cells


def instantmesh_remesh(vertices, faces, target_num_faces=10000):
    """
    remeshing using InstantMeshes
    """
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.cpu().numpy()
        faces = faces.cpu().numpy()
    mesh = trimesh.Trimesh(vertices, faces)
    mesh_file = tempfile.NamedTemporaryFile(suffix=f"_original.obj", delete=False).name
    mesh.export(mesh_file, include_normals=True)

    remeshed_file = tempfile.NamedTemporaryFile(
        suffix=f"_remeshed.obj", delete=False
    ).name

    command = f"tmp/InstantMeshes {mesh_file} -f {target_num_faces} -o {remeshed_file}"
    os.system(command)

    v, f = load_mesh(remeshed_file)

    del mesh_file, remeshed_file

    return v, f


def pyacvd_remesh(vertices, faces, target_num_faces=50000):
    cells = np.zeros((faces.shape[0], 4), dtype=int)
    cells[:, 1:] = faces
    cells[:, 0] = 3
    mesh = pv.PolyData(vertices, cells)
    clus = pyacvd.Clustering(mesh)
    clus.cluster(target_num_faces)
    remesh = clus.create_mesh()

    vertices = remesh.points
    faces = remesh.faces.reshape(-1, 4)[:, 1:]
    return vertices, faces
