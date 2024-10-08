import os
import cv2
import json
import torch
import trimesh
import kaolin as kal
from loguru import logger
import numpy as np
from trimesh.transformations import rotation_matrix
import torch.nn.functional as F

from meshgen.utils.ops import mesh_simplification


def dot(x, y, dim=-1):
    return torch.sum(x * y, dim, keepdim=True)


class Mesh:
    def __init__(
        self,
        mesh_path,
        device,
        target_scale=1.0,
        mesh_dy=0.0,
        remove_mesh_part_names=None,
        remove_unsupported_buffers=None,
        intermediate_dir=None,
        rotate=False,
        simplify_if_necessary=True,
    ):
        # from https://github.com/threedle/text2mesh
        self.material_cvt, self.material_num, org_mesh_path, is_convert = (
            None,
            1,
            mesh_path,
            False,
        )
        # if not mesh_path.endswith(".obj") and not mesh_path.endswith(".off"):
        #     if mesh_path.endswith(".gltf"):
        #         mesh_path = self.preprocess_gltf(
        #             mesh_path, remove_mesh_part_names, remove_unsupported_buffers
        #         )
        #     mesh_temp = trimesh.load(
        #         mesh_path, force="mesh", process=True, maintain_order=True
        #     )
        #     mesh_path = os.path.splitext(mesh_path)[0] + "_cvt.obj"
        #     mesh_temp.export(mesh_path)
        #     merge_texture_path = os.path.join(
        #         os.path.dirname(mesh_path), "material_0.png"
        #     )
        #     if os.path.exists(merge_texture_path):
        #         self.material_cvt = cv2.imread(merge_texture_path)
        #         self.material_num = (
        #             self.material_cvt.shape[1] // self.material_cvt.shape[0]
        #         )
        #     logger.info(
        #         "Converting current mesh model to obj file with {} material~".format(
        #             self.material_num
        #         )
        #     )
        #     is_convert = True

        if ".obj" in mesh_path:
            try:
                mesh = kal.io.obj.import_mesh(
                    mesh_path,
                    with_normals=True,
                    with_materials=True,
                    heterogeneous_mesh_handler=kal.io.utils.mesh_handler_naive_triangulate,
                )
            except:
                mesh = kal.io.obj.import_mesh(
                    mesh_path,
                    with_normals=True,
                    with_materials=False,
                    heterogeneous_mesh_handler=kal.io.utils.mesh_handler_naive_triangulate,
                )
        elif ".off" in mesh_path:
            mesh = kal.io.off.import_mesh(mesh_path)
        else:
            raise ValueError(f"{mesh_path} extension not implemented in mesh reader.")

        self.vertices = mesh.vertices.to(device)
        self.faces = mesh.faces.to(device)
        try:
            self.vt = mesh.uvs
            self.ft = mesh.face_uvs_idx
        except AttributeError:
            self.vt = None
            self.ft = None
        self.mesh_path = mesh_path
        self.normalize_mesh(target_scale=target_scale, mesh_dy=mesh_dy)

        if rotate:
            rotmat = rotation_matrix(-np.pi / 2, [1, 0, 0]) @ rotation_matrix(
                -np.pi / 2, [0, 0, 1]
            )
            rotmat = torch.from_numpy(rotmat.astype(np.float32)).to(device)[:3, :3]
            self.vertices = self.vertices @ rotmat.T

        self.vn = self._compute_normals()

        if self.faces.shape[0] > 20000 and simplify_if_necessary:
            self.vertices, self.faces = mesh_simplification(
                self.vertices, self.faces, 20000
            )
            self.vertices = self.vertices.to(torch.float32)
            self.faces = self.faces.to(torch.int64)
            self.vn = self._compute_normals()

        if is_convert and intermediate_dir is not None:
            if not os.path.exists(intermediate_dir):
                os.makedirs(intermediate_dir)
            if os.path.exists(os.path.splitext(org_mesh_path)[0] + "_removed.gltf"):
                os.system(
                    "mv {} {}".format(
                        os.path.splitext(org_mesh_path)[0] + "_removed.gltf",
                        intermediate_dir,
                    )
                )
            if mesh_path.endswith("_cvt.obj"):
                os.system("mv {} {}".format(mesh_path, intermediate_dir))
            os.system(
                "mv {} {}".format(
                    os.path.join(os.path.dirname(mesh_path), "material.mtl"),
                    intermediate_dir,
                )
            )
            # if os.path.exists(merge_texture_path):
            #     os.system(
            #         "mv {} {}".format(
            #             os.path.join(os.path.dirname(mesh_path), "material_0.png"),
            #             intermediate_dir,
            #         )
            #     )

    def preprocess_gltf(
        self, mesh_path, remove_mesh_part_names, remove_unsupported_buffers
    ):
        with open(mesh_path, "r") as fr:
            gltf_json = json.load(fr)
            if remove_mesh_part_names is not None:
                temp_primitives = []
                for primitive in gltf_json["meshes"][0]["primitives"]:
                    if_append, material_id = True, primitive["material"]
                    material_name = gltf_json["materials"][material_id]["name"]
                    for remove_mesh_part_name in remove_mesh_part_names:
                        if material_name.find(remove_mesh_part_name) >= 0:
                            if_append = False
                            break
                    if if_append:
                        temp_primitives.append(primitive)
                gltf_json["meshes"][0]["primitives"] = temp_primitives
                logger.info(
                    "Deleting mesh with materials named '{}' from gltf model ~".format(
                        remove_mesh_part_names
                    )
                )

            if remove_unsupported_buffers is not None:
                temp_buffers = []
                for buffer in gltf_json["buffers"]:
                    if_append = True
                    for unsupported_buffer in remove_unsupported_buffers:
                        if buffer["uri"].find(unsupported_buffer) >= 0:
                            if_append = False
                            break
                    if if_append:
                        temp_buffers.append(buffer)
                gltf_json["buffers"] = temp_buffers
                logger.info(
                    "Deleting unspported buffers within uri {} from gltf model ~".format(
                        remove_unsupported_buffers
                    )
                )
            updated_mesh_path = os.path.splitext(mesh_path)[0] + "_removed.gltf"
            with open(updated_mesh_path, "w") as fw:
                json.dump(gltf_json, fw, indent=4)
        return updated_mesh_path

    def normalize_mesh(self, target_scale=1.0, mesh_dy=0.0):
        # verts = self.vertices
        # center = verts.mean(dim=0)
        # verts = verts - center
        # scale = torch.max(torch.norm(verts, p=2, dim=1))
        # verts = verts / scale
        # verts *= target_scale
        # verts[:, 1] += mesh_dy
        # self.vertices = verts
        vertices = self.vertices
        """shift and resize mesh to fit into a unit cube"""
        offset = (vertices.min(dim=0)[0] + vertices.max(dim=0)[0]) / 2
        vertices -= offset
        scale_factor = 2.0 / (vertices.max(dim=0)[0] - vertices.min(dim=0)[0]).max()
        vertices *= scale_factor

        self.vertices = vertices * target_scale

    def _compute_normals(self):
        i0 = self.faces[:, 0]
        i1 = self.faces[:, 1]
        i2 = self.faces[:, 2]

        v0 = self.vertices[i0, :]
        v1 = self.vertices[i1, :]
        v2 = self.vertices[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)

        # Splat face normals to vertices
        v_nrm = torch.zeros_like(self.vertices)
        v_nrm.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        v_nrm.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        # Normalize, replace zero (degenerated) normals with some default value
        v_nrm = torch.where(
            dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.as_tensor([0.0, 0.0, 1.0]).to(v_nrm)
        )
        v_nrm = F.normalize(v_nrm, dim=1)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(v_nrm))

        return v_nrm
