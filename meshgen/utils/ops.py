import cv2
from PIL import Image
import math
import numpy as np
import torch
import kaolin
import pyrender
from einops import rearrange
import numbers
import torch.nn.functional as F
import pyvista as pv
import pyacvd
from einops import repeat
import itertools
from tqdm import tqdm

import pyfqmr

from meshgen.utils.birefnet import run_model as rembg_birefnet


def sample_on_surface(vertices, faces, num_samples):
    """
    sample on surface for a single mesh
    """
    input_np = False
    if isinstance(vertices, np.ndarray):
        input_np = True
        vertices = torch.from_numpy(vertices)

    samples = kaolin.ops.mesh.sample_points(vertices[None], faces, num_samples)[0][0]

    if input_np:
        samples = samples.cpu().numpy()

    return samples


# def fps_sample(vertices, num_samples):
#     return kaolin.ops.mesh.farthest_point_sample(vertices[None], num_samples)[0]


def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.

    Bugfix reference: https://github.com/NVlabs/eg3d/issues/67
    """
    return torch.tensor(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
            [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
        ],
        dtype=torch.float32,
    )


def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = (
        coordinates.unsqueeze(1)
        .expand(-1, n_planes, -1, -1)
        .reshape(N * n_planes, M, 3)
    )
    inv_planes = (
        torch.linalg.inv(planes)
        .unsqueeze(0)
        .expand(N, -1, -1, -1)
        .reshape(N * n_planes, 3, 3)
    )
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]


def sample_from_planes(
    plane_axes,
    plane_features,
    coordinates,
    mode="bilinear",
    padding_mode="zeros",
    box_warp=None,
):
    assert padding_mode == "zeros"
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.reshape(N * n_planes, C, H, W)
    dtype = plane_features.dtype

    coordinates = (2 / box_warp) * coordinates  # add specific box bounds

    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    with torch.autocast("cuda", enabled=False):
        output_features = (
            torch.nn.functional.grid_sample(
                plane_features.float(),
                projected_coordinates.float(),
                mode=mode,
                padding_mode=padding_mode,
                align_corners=False,
            )
            .permute(0, 3, 2, 1)
            .reshape(N, n_planes, M, C)
        )
    return output_features


def logit_normal(mu, sigma, shape, device, dtype):
    z = torch.randn(*shape, device=device, dtype=dtype)
    z = mu + sigma * z
    t = torch.sigmoid(z)
    return t


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def dot(x, y):
    return torch.sum(x * y, -1, keepdim=True)


def length(x, eps=1e-20):
    return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)


def get_projection_matrix(fov, reso, flip_y=False):
    # flip_y is used in nvdiffrast
    fov = np.deg2rad(fov)
    cam = pyrender.PerspectiveCamera(yfov=fov)

    proj_mat = cam.get_projection_matrix(reso, reso)
    if flip_y:
        proj_mat[1, :] *= -1

    return proj_mat


def calc_normal(v, f):
    i0, i1, i2 = (
        f[:, 0].long(),
        f[:, 1].long(),
        f[:, 2].long(),
    )
    v0, v1, v2 = v[i0, :], v[i1, :], v[i2, :]

    face_normals = torch.cross(v1 - v0, v2 - v0)
    face_normals = safe_normalize(face_normals)

    vn = torch.zeros_like(v)
    vn.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
    vn.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
    vn.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

    vn = torch.where(
        torch.sum(vn * vn, -1, keepdim=True) > 1e-20,
        vn,
        torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device),
    )

    return vn


def compute_tv_loss(planes):
    # planes are with shape [b, 3, c, h, w]
    planes = rearrange(planes, "b n c h w -> (b n) c h w")
    batch_size, c, h, w = planes.shape
    count_h = batch_size * c * (h - 1) * w
    count_w = batch_size * c * h * (w - 1)
    h_tv = torch.square(planes[..., 1:, :] - planes[..., : h - 1, :]).sum()
    w_tv = torch.square(planes[..., :, 1:] - planes[..., :, : w - 1]).sum()
    return 2 * (
        h_tv / count_h + w_tv / count_w
    )  # This is summing over batch and c instead of avg


class GaussianSmoothing(torch.nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence):
            Number of channels of the input tensors.
            Output will have this number of channels as well.
        kernel_size (int, sequence):
            Size of the gaussian kernel.
        sigma (float, sequence):
            Standard deviation of the gaussian kernel.
        dim (int, optional):
            The number of dimensions of the data.
            Default value is 2 (spatial).
        stride (int, sequence, optional):
            Stride for Conv module.
        padding (int, sequence, optional):
            Padding for Conv module.
        padding_mode (str, optional):
            Padding mode for Conv module.
    """

    def __init__(self, channels, kernel_size, sigma, dim=2, stride=1, padding=0):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # Used for Conv module
        self.stride = stride
        self.padding = padding

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size],
            indexing="ij",
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                f"Only 1, 2 and 3 dimensions are supported. Received {dim}."
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(
            input,
            weight=self.weight,
            groups=self.groups,
            stride=self.stride,
            padding=self.padding,
        )


def preprocess_ip(ip, ref_mask, ref_image, ignore_alpha=True):
    ref_h, ref_w = ref_mask.shape
    # if ip.mode != "RGBA":
    #     ip = ip.convert("RGB")
    #     ip = remove(ip, alpha_matting=False, session=get_rembg_session())
    if ip.mode != "RGBA" or ignore_alpha:
        ip = ip.convert("RGB")
        ip = rembg_birefnet(ip)
    # ip = remove(ip, alpha_matting=False, session=get_rembg_session())
    # blend with white background
    ip = np.asarray(ip)
    ip_mask = ip[:, :, 3].astype(np.float32) / 255
    ip_rgb = ip[:, :, :3].astype(np.float32) / 255
    ip_rgb = ip_rgb * ip_mask[:, :, None].astype(np.float32) + (
        1 - ip_mask[:, :, None].astype(np.float32)
    )
    ip = Image.fromarray((ip_rgb * 255).astype(np.uint8))

    coords = np.nonzero(ip_mask)
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    len_x, len_y = x_max - x_min, y_max - y_min
    center = (x_min + x_max) // 2, (y_min + y_max) // 2

    ref_coords = np.nonzero(ref_mask)
    ref_y_min, ref_y_max = ref_coords[0].min(), ref_coords[0].max()
    ref_x_min, ref_x_max = ref_coords[1].min(), ref_coords[1].max()
    ref_len_x, ref_len_y = ref_x_max - ref_x_min, ref_y_max - ref_y_min
    ref_center = (ref_x_min + ref_x_max) // 2, (ref_y_min + ref_y_max) // 2

    ip = ip.crop((x_min, y_min, x_max, y_max)).resize((ref_len_x, ref_len_y))
    new_ip = Image.new("RGB", (ref_h, ref_w), (255, 255, 255))
    new_ip.paste(ip, (ref_x_min, ref_y_min, ref_x_max, ref_y_max))

    return new_ip


def mesh_simplification(v, f, target=50000, backend="pyacvd"):
    is_torch = False
    if isinstance(v, torch.Tensor):
        is_torch = True
        device = v.device
        dtype = v.dtype
        v = v.detach().cpu().numpy()
        f = f.detach().cpu().numpy()

    if backend == "pyfqmr":
        mesh_simplifier = pyfqmr.Simplify()
        mesh_simplifier.setMesh(v, f)
        mesh_simplifier.simplify_mesh(
            target_count=target,
            aggressiveness=7,
            preserve_border=True,
        )
        ret = mesh_simplifier.getMesh()[:2]
    elif backend == "pyacvd":
        cells = np.zeros((f.shape[0], 4), dtype=int)
        cells[:, 1:] = f
        cells[:, 0] = 3
        mesh = pv.PolyData(v, cells)
        clus = pyacvd.Clustering(mesh)
        clus.cluster(target)
        remesh = clus.create_mesh()

        vertices = remesh.points
        faces = remesh.faces.reshape(-1, 4)[:, 1:]
        ret = [vertices, faces]
    else:
        raise ValueError(f"Backend {backend} not implemented in mesh simplification")

    if is_torch:
        ret = list(map(lambda x: torch.from_numpy(x).to(device=device), ret))

    return ret


def dilate_depth_outline(depth, iters=5, dilate_kernel=3):
    # ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)

    img = np.asarray(depth)
    for i in range(iters):
        _, mask = cv2.threshold(img, thresh=0, maxval=255, type=cv2.THRESH_BINARY)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
        mask = mask / 255

        img_dilate = cv2.dilate(img, np.ones((dilate_kernel, dilate_kernel), np.uint8))

        img = (mask * img + (1 - mask) * img_dilate).astype(np.uint8)
    return Image.fromarray(img)


def dilate_mask(mask, dilate_kernel=10):
    mask = np.asarray(mask)
    mask = cv2.dilate(mask, np.ones((dilate_kernel, dilate_kernel), np.uint8))

    return Image.fromarray(mask)


def extract_bg_mask(img, mask_color=[204, 25, 204], dilate_kernel=5):
    """
    :param mask_color:  BGR
    :return:
    """
    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    mask = (img == mask_color).all(axis=2).astype(np.float32)
    mask = mask[:, :, np.newaxis]

    mask = cv2.dilate(mask, np.ones((dilate_kernel, dilate_kernel), np.uint8))[
        :, :, np.newaxis
    ]
    mask = (mask * 255).astype(np.uint8)
    mask = repeat(mask, "h w 1 -> h w c", c=3)
    return Image.fromarray(mask)


@torch.no_grad()
def dilate_mask(mask, kernel_size=10, format="hwc"):
    is_torch = False
    is_pil = False
    if format == "chw":
        mask = rearrange(mask, "c h w -> h w c")
    if isinstance(mask, torch.Tensor):
        is_torch = True
        dtype = mask.dtype
        device = mask.device
        mask = mask.detach().cpu().numpy()
    if isinstance(mask, Image.Image):
        is_pil = True
        mask = np.asarray(mask) / 255
    mask = cv2.dilate(mask, np.ones((kernel_size, kernel_size)))

    if is_torch:
        mask = torch.from_numpy(mask).to(device=device, dtype=dtype)

    if is_pil:
        mask = Image.fromarray((mask * 255).astype(np.uint8))

    if format == "chw":
        if mask.dim() == 2:
            mask = rearrange(mask, "h w -> 1 h w")
        else:
            mask = rearrange(mask, "h w c -> c h w")

    return mask


@torch.no_grad()
def latent_preview(x):
    # adapted from https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204/7
    v1_4_latent_rgb_factors = torch.tensor(
        [
            #   R        G        B
            [0.298, 0.207, 0.208],  # L1
            [0.187, 0.286, 0.173],  # L2
            [-0.158, 0.189, 0.264],  # L3
            [-0.184, -0.271, -0.473],  # L4
        ],
        dtype=x.dtype,
        device=x.device,
    )
    image = x.permute(0, 2, 3, 1) @ v1_4_latent_rgb_factors
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.float()
    image = image.cpu()
    image = image.numpy()
    return image


@torch.no_grad()
def get_canny_edge(image, threshold1=100, threshold2=200):
    image = np.asarray(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    return Image.fromarray(edges)


def uv_padding(image, hole_mask, padding=2, uv_padding_block=4):
    uv_padding_size = padding
    image1 = (image[0].detach().cpu().numpy() * 255).astype(np.uint8)
    hole_mask = (hole_mask[0].detach().cpu().numpy() * 255).astype(np.uint8)
    block = uv_padding_block
    res = image1.shape[0]
    chunk = res // block
    inpaint_image = np.zeros_like(image1)
    prods = list(itertools.product(range(block), range(block)))
    for i, j in tqdm(prods):
        patch = cv2.inpaint(
            image1[i * chunk : (i + 1) * chunk, j * chunk : (j + 1) * chunk],
            hole_mask[i * chunk : (i + 1) * chunk, j * chunk : (j + 1) * chunk],
            uv_padding_size,
            cv2.INPAINT_TELEA,
        )
        inpaint_image[i * chunk : (i + 1) * chunk, j * chunk : (j + 1) * chunk] = patch
    inpaint_image = inpaint_image / 255.0
    return torch.from_numpy(inpaint_image).to(image)
