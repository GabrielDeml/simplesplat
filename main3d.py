"""
3D Gaussian Splatting — educational, pure-PyTorch implementation.

Matches the original paper (Kerbl et al., SIGGRAPH 2023) as closely as
possible without custom CUDA kernels.

Usage:
    python main3d.py path/to/nerf_synthetic/lego

Download data:
    https://www.kaggle.com/datasets/nguyenhung1903/nerf-synthetic-dataset
    unzip nerf_synthetic.zip
"""

import time
import json
import math
import random
import os
import sys

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from PIL import Image
import numpy as np


# ============================================================================
# Device & Utilities
# ============================================================================

def get_device() -> torch.device:
    """Pick the best available device: CUDA GPU > Apple MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_image(tensor: torch.Tensor, path: str) -> None:
    """Save a (H, W, 3) float tensor as an image file."""
    img = (tensor.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)
    print(f"Saved {path}")


def save_ply(gaussians: "Gaussians3D", path: str) -> None:
    """Save Gaussians to a .ply file in the standard 3DGS format.

    Compatible with SuperSplat, the official 3DGS viewer, and most web viewers.
    Values are stored in their raw (pre-activation) form:
      - scale in log-space (viewer applies exp)
      - opacity in logit-space (viewer applies sigmoid)
      - SH coefficients as-is (viewer evaluates SH basis functions)
    """
    N = gaussians.position.shape[0]

    pos = gaussians.position.detach().cpu().numpy()                   # (N, 3)
    normals = np.zeros((N, 3), dtype=np.float32)                      # (N, 3) placeholder
    f_dc = gaussians.sh_coeffs[:, 0, :].detach().cpu().numpy()        # (N, 3) degree-0 SH
    # Higher-degree SH: (N, K, 3) → transpose to (N, 3, K) → flatten to (N, 3K)
    # This groups by color channel: [R_1..R_K, G_1..G_K, B_1..B_K]
    f_rest = gaussians.sh_coeffs[:, 1:, :].detach().cpu().numpy()     # (N, 15, 3)
    f_rest = np.transpose(f_rest, (0, 2, 1)).reshape(N, -1)          # (N, 45)
    opacity = gaussians.opacity.detach().cpu().numpy()[:, None]       # (N, 1)
    scale = gaussians.scale.detach().cpu().numpy()                    # (N, 3)
    rotation = gaussians.rotation.detach().cpu().numpy()              # (N, 4)

    # Build PLY header
    n_rest = f_rest.shape[1]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
    )
    for i in range(n_rest):
        header += f"property float f_rest_{i}\n"
    header += (
        "property float opacity\n"
        "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
        "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
        "end_header\n"
    )

    # Write binary data — all float32, concatenated per-vertex
    data = np.concatenate([pos, normals, f_dc, f_rest, opacity, scale, rotation], axis=1)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.astype(np.float32).tobytes())

    print(f"Saved {path} ({N} Gaussians, {data.shape[1]} properties)")


# ============================================================================
# Spherical Harmonics (degree 3 — 16 basis functions)
# ============================================================================
#
# Degree 0 (1 function):  constant color regardless of direction
# Degree 1 (3 functions): linear variation — basic directionality
# Degree 2 (5 functions): quadratic — specular-like highlights
# Degree 3 (7 functions): cubic — sharper specular effects
# Total: 1 + 3 + 5 + 7 = 16 basis functions

SH_C0 = 0.28209479177387814       # sqrt(1 / (4π))
SH_C1 = 0.4886025119029199        # sqrt(3 / (4π))
SH_C2_0 = 1.0925484305920792      # sqrt(15 / (4π))
SH_C2_1 = 0.31539156525252005     # sqrt(5 / (16π))
SH_C2_2 = 0.5462742152960396      # sqrt(15 / (16π))
SH_C3_0 = 0.5900435899266435      # sqrt(35 / (32π)) × sqrt(2)
SH_C3_1 = 2.890611442640554       # sqrt(105 / (4π))
SH_C3_2 = 0.4570457994644658      # sqrt(21 / (32π)) × sqrt(2)
SH_C3_3 = 0.3731763325901154      # sqrt(7 / (16π))
SH_C3_4 = 0.4570457994644658      # same as SH_C3_2
SH_C3_5 = 1.4453057213202769      # sqrt(105 / (16π))
SH_C3_6 = 0.5900435899266435      # same as SH_C3_0


def eval_sh(
    sh_coeffs: torch.Tensor, directions: torch.Tensor, active_degree: int = 3
) -> torch.Tensor:
    """Evaluate spherical harmonics for given viewing directions.

    sh_coeffs:     (N, 16, 3) — learned SH coefficients (degree 3)
    directions:    (N, 3)     — unit viewing directions (Gaussian → camera)
    active_degree: int        — max SH degree to evaluate (for progressive activation)
    Returns:       (N, 3)     — RGB color per Gaussian, clamped to [0, 1]
    """
    x = directions[:, 0:1]  # (N, 1)
    y = directions[:, 1:2]
    z = directions[:, 2:3]

    # L=0 (1 function)
    result = SH_C0 * sh_coeffs[:, 0]  # (N, 3)

    if active_degree >= 1:
        # L=1 (3 functions)
        result = result + SH_C1 * (
            y * sh_coeffs[:, 1] + z * sh_coeffs[:, 2] + x * sh_coeffs[:, 3]
        )

    if active_degree >= 2:
        # L=2 (5 functions)
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        result = result + (
            SH_C2_0 * xy * sh_coeffs[:, 4]
            + SH_C2_0 * yz * sh_coeffs[:, 5]
            + SH_C2_1 * (3.0 * zz - 1.0) * sh_coeffs[:, 6]
            + SH_C2_0 * xz * sh_coeffs[:, 7]
            + SH_C2_2 * (xx - yy) * sh_coeffs[:, 8]
        )

    if active_degree >= 3:
        # L=3 (7 functions)
        result = result + (
            SH_C3_0 * y * (3.0 * xx - yy) * sh_coeffs[:, 9]
            + SH_C3_1 * x * y * z * sh_coeffs[:, 10]
            + SH_C3_2 * y * (5.0 * zz - 1.0) * sh_coeffs[:, 11]
            + SH_C3_3 * z * (5.0 * zz - 3.0) * sh_coeffs[:, 12]
            + SH_C3_4 * x * (5.0 * zz - 1.0) * sh_coeffs[:, 13]
            + SH_C3_5 * z * (xx - yy) * sh_coeffs[:, 14]
            + SH_C3_6 * x * (xx - 3.0 * yy) * sh_coeffs[:, 15]
        )

    # +0.5 centers output so zero coefficients give grey, then clamp to valid range
    return (result + 0.5).clamp(0.0, 1.0)


# ============================================================================
# SSIM Loss
# ============================================================================

def _gaussian_window(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create a 1D Gaussian window for SSIM computation."""
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def ssim(
    img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11
) -> torch.Tensor:
    """Compute structural similarity index (SSIM) between two images.

    img1, img2: (H, W, 3) float tensors in [0, 1]
    Returns: scalar SSIM value (higher = more similar, max 1.0)
    """
    device = img1.device
    C = 3  # color channels

    # Build 2D Gaussian window from outer product of 1D windows
    w1d = _gaussian_window(window_size, 1.5, device)
    window = (w1d[:, None] * w1d[None, :]).unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)
    window = window.expand(C, 1, window_size, window_size)            # (3, 1, K, K)

    # Reshape to (B, C, H, W) for conv2d
    i1 = img1.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    i2 = img2.permute(2, 0, 1).unsqueeze(0)

    pad = window_size // 2
    mu1 = F.conv2d(i1, window, padding=pad, groups=C)
    mu2 = F.conv2d(i2, window, padding=pad, groups=C)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(i1 * i1, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(i2 * i2, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(i1 * i2, window, padding=pad, groups=C) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


# ============================================================================
# Camera Model
# ============================================================================

@dataclass
class Camera:
    """Pinhole camera with extrinsics (pose) and intrinsics (lens).

    Convention: OpenCV — camera looks down +z, y points down, x points right.
    A point in front of the camera has positive z in camera space.
    """
    R: torch.Tensor   # (3, 3)  world-to-camera rotation
    t: torch.Tensor   # (3,)    world-to-camera translation: p_cam = R @ p_world + t
    fx: float         # focal length in pixels (horizontal)
    fy: float         # focal length in pixels (vertical)
    cx: float         # principal point x (typically width / 2)
    cy: float         # principal point y (typically height / 2)
    width: int
    height: int


# ============================================================================
# Dataset Loading (NeRF-Synthetic / Blender format)
# ============================================================================

def load_nerf_dataset(
    data_dir: str, image_size: int = 128
) -> tuple[list[torch.Tensor], list[Camera]]:
    """Load a NeRF-synthetic (Blender) dataset.

    Expected directory layout:
        data_dir/
            transforms_train.json   — camera FOV + per-frame 4×4 camera-to-world matrices
            train/r_0.png, r_1.png, ...  — RGBA images

    The transform_matrix in the JSON is a 4×4 camera-to-world matrix in OpenGL
    convention (y up, -z forward).  We convert to OpenCV convention (y down,
    +z forward) by negating the y and z columns, then invert to get
    world-to-camera R, t.

    Returns: (images, cameras)
        images[i]  — (H, W, 3) float tensor in [0, 1], RGBA composited over white
        cameras[i] — Camera with world-to-camera R, t and intrinsics
    """
    json_path = os.path.join(data_dir, "transforms_train.json")
    with open(json_path) as f:
        meta = json.load(f)

    # Focal length from horizontal field of view
    fov_x = meta["camera_angle_x"]  # radians
    fx = image_size / (2.0 * math.tan(fov_x / 2.0))
    fy = fx   # square pixels
    cx = image_size / 2.0
    cy = image_size / 2.0

    images: list[torch.Tensor] = []
    cameras: list[Camera] = []

    for frame in meta["frames"]:
        # --- Load image ---
        img_path = frame["file_path"]
        if not img_path.endswith(".png"):
            img_path += ".png"
        img_path = os.path.join(data_dir, img_path)

        img = Image.open(img_path).convert("RGBA").resize((image_size, image_size))
        img = np.array(img, dtype=np.float32) / 255.0  # (H, W, 4)

        # Composite RGBA over white background:
        # rgb_out = rgb × alpha + white × (1 − alpha)
        alpha = img[:, :, 3:4]
        rgb = img[:, :, :3] * alpha + (1.0 - alpha)
        images.append(torch.tensor(rgb))

        # --- Parse camera pose ---
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)  # (4, 4)

        # OpenGL → OpenCV: negate y and z columns of the rotation part
        # This flips "y up, z backward" → "y down, z forward"
        c2w[:3, 1] *= -1
        c2w[:3, 2] *= -1

        # Invert camera-to-world → world-to-camera
        w2c = np.linalg.inv(c2w)
        R = torch.tensor(w2c[:3, :3])
        t = torch.tensor(w2c[:3, 3])

        cameras.append(Camera(R=R, t=t, fx=fx, fy=fy, cx=cx, cy=cy,
                              width=image_size, height=image_size))

    print(f"Loaded {len(images)} views at {image_size}×{image_size} from {data_dir}")
    return images, cameras


# ============================================================================
# Gaussians3D
# ============================================================================

@dataclass
class Gaussians3D:
    """A collection of N 3D Gaussians with learnable parameters.

    position: (N, 3)     — xyz centers in world space
    scale:    (N, 3)     — log-space sizes; exp(scale) = actual spread in x, y, z
    rotation: (N, 4)     — unit quaternions (w, x, y, z) encoding 3D orientation
    sh_dc:    (N, 1, 3)  — degree-0 SH (base color), separate LR from rest
    sh_rest:  (N, 15, 3) — degree 1-3 SH (view-dependent color), 20x lower LR
    opacity:  (N,)       — logit-space; sigmoid(opacity) = actual transparency [0, 1]
    """
    position: torch.Tensor
    scale: torch.Tensor
    rotation: torch.Tensor
    sh_dc: torch.Tensor
    sh_rest: torch.Tensor
    opacity: torch.Tensor

    @property
    def sh_coeffs(self) -> torch.Tensor:
        """Combined SH coefficients (N, 16, 3) for rendering."""
        return torch.cat([self.sh_dc, self.sh_rest], dim=1)

    def params(self) -> list[torch.Tensor]:
        return [self.position, self.scale, self.rotation, self.sh_dc, self.sh_rest, self.opacity]


def create_random_gaussians_3d(
    n: int, scene_extent: float = 2.0, device: torch.device = torch.device("cpu")
) -> Gaussians3D:
    """Create N random 3D Gaussians scattered in a cube of side 2×scene_extent."""
    return Gaussians3D(
        position=((torch.rand(n, 3, device=device) - 0.5) * 2 * scene_extent
                  ).requires_grad_(True),
        scale=(torch.ones(n, 3, device=device) * math.log(scene_extent / 10.0)
               ).requires_grad_(True),
        rotation=F.normalize(torch.randn(n, 4, device=device), dim=1
                             ).requires_grad_(True),
        sh_dc=torch.zeros(n, 1, 3, device=device).requires_grad_(True),
        sh_rest=torch.zeros(n, 15, 3, device=device).requires_grad_(True),
        opacity=torch.zeros(n, device=device).requires_grad_(True),
    )


# ============================================================================
# Math Helpers
# ============================================================================

def quat_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternions to 3×3 rotation matrices.

    q: (N, 4) — quaternions as (w, x, y, z)
    Returns: (N, 3, 3) rotation matrices
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    # fmt: off
    R = torch.stack([
        1 - 2*(yy + zz),  2*(xy - wz),      2*(xz + wy),
        2*(xy + wz),      1 - 2*(xx + zz),   2*(yz - wx),
        2*(xz - wy),      2*(yz + wx),        1 - 2*(xx + yy),
    ], dim=-1).reshape(-1, 3, 3)
    # fmt: on

    return R


def build_covariance_3d(
    scale: torch.Tensor, rotation: torch.Tensor
) -> torch.Tensor:
    """Build 3D covariance matrices from scale and rotation.

    Σ = R S Sᵀ Rᵀ  where R is from the quaternion and S = diag(sx, sy, sz).

    scale:    (N, 3) — actual scales (already exp'd from log-space)
    rotation: (N, 4) — unit quaternions
    Returns:  (N, 3, 3) — symmetric positive semi-definite covariance matrices
    """
    R = quat_to_rotation_matrix(rotation)  # (N, 3, 3)
    S = torch.diag_embed(scale)            # (N, 3, 3)
    M = R @ S                              # (N, 3, 3)
    return M @ M.transpose(-1, -2)         # Σ = R S Sᵀ Rᵀ


# ============================================================================
# Rendering
# ============================================================================

@dataclass
class RenderOutput:
    """Outputs from render(), used for both the image and densification stats."""
    image: torch.Tensor         # (H, W, 3)
    means_2d: torch.Tensor      # (V, 2) — viewspace 2D positions (has retain_grad)
    radii: torch.Tensor         # (V,)   — screen-space radius per visible Gaussian
    visible: torch.Tensor       # (N,)   — boolean mask of which Gaussians were visible


def render(
    gaussians: Gaussians3D, camera: Camera, bg_color: float = 1.0,
    active_sh_degree: int = 3,
) -> RenderOutput:
    """Render 3D Gaussians from a camera viewpoint.

    Returns a RenderOutput with the image plus viewspace stats needed for
    the paper's densification (2D projected position gradients, screen-space radii).
    """
    device = gaussians.position.device
    N = gaussians.position.shape[0]
    H, W = camera.height, camera.width

    scales = torch.exp(gaussians.scale)                     # (N, 3)
    opacities = torch.sigmoid(gaussians.opacity)            # (N,)
    rotations = F.normalize(gaussians.rotation, dim=1)      # (N, 4)

    cov3d = build_covariance_3d(scales, rotations)          # (N, 3, 3)

    R_cam = camera.R.to(device)
    t_cam = camera.t.to(device)
    pos_cam = (R_cam @ gaussians.position.T).T + t_cam      # (N, 3)

    visible = pos_cam[:, 2] > 0.1                           # (N,)
    if not visible.any():
        empty = torch.full((H, W, 3), bg_color, device=device)
        return RenderOutput(empty, torch.zeros(0, 2, device=device),
                            torch.zeros(0, device=device), visible)

    pos_cam   = pos_cam[visible]
    cov3d     = cov3d[visible]
    opacities = opacities[visible]
    sh_coeffs = gaussians.sh_coeffs[visible]
    pos_world = gaussians.position[visible]
    V = pos_cam.shape[0]

    tx, ty, tz = pos_cam[:, 0], pos_cam[:, 1], pos_cam[:, 2]
    fx, fy = camera.fx, camera.fy
    cx, cy = camera.cx, camera.cy

    # Viewspace 2D projected positions — retain_grad so we can read the
    # gradient after backward (the paper uses this for densification).
    means_2d = torch.stack([fx * tx / tz + cx, fy * ty / tz + cy], dim=1)
    means_2d.retain_grad()

    cov_cam = R_cam[None] @ cov3d @ R_cam.T[None]

    J = torch.zeros(V, 2, 3, device=device)
    J[:, 0, 0] = fx / tz
    J[:, 0, 2] = -fx * tx / (tz * tz)
    J[:, 1, 1] = fy / tz
    J[:, 1, 2] = -fy * ty / (tz * tz)

    cov2d = J @ cov_cam @ J.transpose(-1, -2) + 0.3 * torch.eye(2, device=device)

    a = cov2d[:, 0, 0]
    b = cov2d[:, 0, 1]
    d = cov2d[:, 1, 1]
    det = (a * d - b * b).clamp(min=1e-6)
    inv_a = d / det
    inv_b = -b / det
    inv_d = a / det

    # Screen-space radius from max eigenvalue of 2D covariance (for pruning)
    trace = a + d
    sqrt_disc = torch.sqrt((trace * trace - 4 * det).clamp(min=0))
    lambda_max = 0.5 * (trace + sqrt_disc)
    radii = 3.0 * torch.sqrt(lambda_max)                    # (V,) 3-sigma radius in pixels

    cam_center = -(R_cam.T @ t_cam)
    view_dirs = F.normalize(cam_center[None] - pos_world, dim=1)
    colors = eval_sh(sh_coeffs, view_dirs, active_degree=active_sh_degree)

    # Sort by depth — means_2d was already created with retain_grad above,
    # gradients flow through the index operation back to the unsorted means_2d
    depth_order = pos_cam[:, 2].argsort()
    sorted_means = means_2d[depth_order]
    inv_a     = inv_a[depth_order]
    inv_b     = inv_b[depth_order]
    inv_d     = inv_d[depth_order]
    opacities = opacities[depth_order]
    colors    = colors[depth_order]

    ys = torch.arange(H, dtype=torch.float32, device=device)
    xs = torch.arange(W, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    dx = xx[:, :, None] - sorted_means[None, None, :, 0]
    dy = yy[:, :, None] - sorted_means[None, None, :, 1]

    mahal = (dx * dx * inv_a[None, None, :]
             + 2 * dx * dy * inv_b[None, None, :]
             + dy * dy * inv_d[None, None, :])

    alpha = (opacities[None, None, :] * torch.exp(-0.5 * mahal)).clamp(max=0.99)

    one_minus_alpha = 1 - alpha
    T = torch.cat([
        torch.ones(H, W, 1, device=device),
        one_minus_alpha[:, :, :-1].cumprod(dim=2),
    ], dim=2)

    weights = alpha * T
    rendered = (weights.unsqueeze(-1) * colors[None, None, :, :]).sum(dim=2)

    T_final = one_minus_alpha.prod(dim=2, keepdim=True)
    rendered = rendered + bg_color * T_final

    return RenderOutput(rendered.clamp(0, 1), means_2d, radii, visible)


# ============================================================================
# Adaptive Density Control
# ============================================================================

def prune_3d(gaussians: Gaussians3D, min_opacity: float = 0.005,
             scene_extent: float = 1.0) -> Gaussians3D:
    """Remove Gaussians that are nearly transparent or excessively large.

    Matches the paper: prune if sigmoid(opacity) < ε_α or max scale > scene_extent.
    """
    actual_opacity = torch.sigmoid(gaussians.opacity.detach())
    actual_scale = torch.exp(gaussians.scale.detach()).max(dim=1).values
    keep = (actual_opacity > min_opacity) & (actual_scale < scene_extent)
    if keep.all():
        return gaussians, keep

    n_before = gaussians.position.shape[0]
    gaussians = Gaussians3D(
        position=gaussians.position.detach()[keep].requires_grad_(True),
        scale=gaussians.scale.detach()[keep].requires_grad_(True),
        rotation=gaussians.rotation.detach()[keep].requires_grad_(True),
        sh_dc=gaussians.sh_dc.detach()[keep].requires_grad_(True),
        sh_rest=gaussians.sh_rest.detach()[keep].requires_grad_(True),
        opacity=gaussians.opacity.detach()[keep].requires_grad_(True),
    )
    print(f"    Pruned: {n_before} → {gaussians.position.shape[0]}")
    return gaussians, keep


def reset_opacities(gaussians: Gaussians3D) -> Gaussians3D:
    """Reset all opacities to near-transparent (sigmoid → ~0.01).

    The paper resets every 3000 steps (out of 30000).  This forces every
    Gaussian to re-earn its opacity.  Useful ones quickly recover;
    useless ones stay transparent and get pruned next cycle.
    """
    new_opacity = torch.full_like(gaussians.opacity.detach(), -4.6)  # logit(0.01)
    gaussians = Gaussians3D(
        position=gaussians.position.detach().requires_grad_(True),
        scale=gaussians.scale.detach().requires_grad_(True),
        rotation=gaussians.rotation.detach().requires_grad_(True),
        sh_dc=gaussians.sh_dc.detach().requires_grad_(True),
        sh_rest=gaussians.sh_rest.detach().requires_grad_(True),
        opacity=new_opacity.requires_grad_(True),
    )
    print(f"    Reset opacities → sigmoid(-4.6) ≈ 0.01")
    return gaussians


def split_3d(gaussians: Gaussians3D, avg_grad: torch.Tensor,
             grad_threshold: float = 0.0002,
             percent_dense: float = 0.01,
             scene_extent: float = 1.0) -> Gaussians3D:
    """Split large Gaussians with high accumulated gradients.

    Matches the paper:
    - Uses accumulated mean gradient (not per-step)
    - Scale threshold = percent_dense × scene_extent
    - Children sampled from parent Gaussian's PDF (using its rotation + scale)
    - Children scaled to parent_scale / 1.6
    """
    actual_scale = torch.exp(gaussians.scale.detach())       # (N, 3)
    max_scale = actual_scale.max(dim=1).values               # (N,)
    scale_threshold = percent_dense * scene_extent

    should_split = (avg_grad > grad_threshold) & (max_scale > scale_threshold)
    if not should_split.any():
        return gaussians

    n_splits = should_split.sum().item()
    keep = ~should_split

    # Sample child positions from the parent Gaussian's distribution:
    # pos_child = R @ (scale * z) + pos_parent,  where z ~ N(0, I)
    split_pos   = gaussians.position.detach()[should_split]     # (S, 3)
    split_scale = gaussians.scale.detach()[should_split]        # (S, 3) log-space
    split_rot   = gaussians.rotation.detach()[should_split]     # (S, 4)
    split_sh_dc   = gaussians.sh_dc.detach()[should_split]      # (S, 1, 3)
    split_sh_rest = gaussians.sh_rest.detach()[should_split]    # (S, 15, 3)
    split_opa   = gaussians.opacity.detach()[should_split]      # (S,)

    stds = torch.exp(split_scale)                               # (S, 3) actual scale
    R_mat = quat_to_rotation_matrix(F.normalize(split_rot, dim=1))  # (S, 3, 3)

    # Sample two children per parent
    samples_a = torch.randn_like(split_pos) * stds             # (S, 3)
    samples_b = torch.randn_like(split_pos) * stds
    offset_a = torch.bmm(R_mat, samples_a.unsqueeze(-1)).squeeze(-1)  # (S, 3)
    offset_b = torch.bmm(R_mat, samples_b.unsqueeze(-1)).squeeze(-1)
    pos_a = split_pos + offset_a
    pos_b = split_pos + offset_b

    # Children are scaled to parent / 1.6  (paper uses 0.8 * N with N=2)
    new_scale = split_scale - math.log(1.6)

    gaussians = Gaussians3D(
        position=torch.cat([gaussians.position.detach()[keep], pos_a, pos_b]).requires_grad_(True),
        scale=torch.cat([gaussians.scale.detach()[keep], new_scale, new_scale]).requires_grad_(True),
        rotation=torch.cat([gaussians.rotation.detach()[keep], split_rot, split_rot]).requires_grad_(True),
        sh_dc=torch.cat([gaussians.sh_dc.detach()[keep], split_sh_dc, split_sh_dc]).requires_grad_(True),
        sh_rest=torch.cat([gaussians.sh_rest.detach()[keep], split_sh_rest, split_sh_rest]).requires_grad_(True),
        opacity=torch.cat([gaussians.opacity.detach()[keep], split_opa, split_opa]).requires_grad_(True),
    )
    print(f"    Split {n_splits} → now {gaussians.position.shape[0]}")
    return gaussians


def duplicate_3d(gaussians: Gaussians3D, avg_grad: torch.Tensor,
                 grad_threshold: float = 0.0002,
                 percent_dense: float = 0.01,
                 scene_extent: float = 1.0) -> Gaussians3D:
    """Clone small, high-gradient Gaussians (keeps original + adds a copy).

    Matches the paper: small Gaussians (below the scale threshold) with high
    accumulated gradients get duplicated with identical parameters.
    """
    actual_scale = torch.exp(gaussians.scale.detach())
    max_s = actual_scale.max(dim=1).values
    scale_threshold = percent_dense * scene_extent

    should_dup = (avg_grad > grad_threshold) & (max_s <= scale_threshold)
    if not should_dup.any():
        return gaussians

    n_dups = should_dup.sum().item()
    # Paper duplicates with identical parameters (no random offset)
    dup_pos   = gaussians.position.detach()[should_dup]
    dup_scale = gaussians.scale.detach()[should_dup]
    dup_rot   = gaussians.rotation.detach()[should_dup]
    dup_sh_dc   = gaussians.sh_dc.detach()[should_dup]
    dup_sh_rest = gaussians.sh_rest.detach()[should_dup]
    dup_opa   = gaussians.opacity.detach()[should_dup]

    gaussians = Gaussians3D(
        position=torch.cat([gaussians.position.detach(), dup_pos]).requires_grad_(True),
        scale=torch.cat([gaussians.scale.detach(), dup_scale]).requires_grad_(True),
        rotation=torch.cat([gaussians.rotation.detach(), dup_rot]).requires_grad_(True),
        sh_dc=torch.cat([gaussians.sh_dc.detach(), dup_sh_dc]).requires_grad_(True),
        sh_rest=torch.cat([gaussians.sh_rest.detach(), dup_sh_rest]).requires_grad_(True),
        opacity=torch.cat([gaussians.opacity.detach(), dup_opa]).requires_grad_(True),
    )
    print(f"    Duplicated {n_dups} → now {gaussians.position.shape[0]}")
    return gaussians


# ============================================================================
# Training
# ============================================================================

def get_position_lr(step: int, lr_init: float, lr_final: float,
                    max_steps: int) -> float:
    """Exponential learning rate decay for positions (matches the paper)."""
    if step >= max_steps:
        return lr_final
    t = step / max_steps
    return lr_init * math.exp(t * math.log(lr_final / lr_init))


def build_optimizer(gaussians: Gaussians3D, position_lr: float) -> torch.optim.Adam:
    """Create Adam optimizer with the paper's per-parameter learning rates.

    Paper values: position=0.00016×extent (decayed), scale=0.005, rotation=0.001,
    SH DC=0.0025, SH rest=0.000125 (20× lower), opacity=0.05.
    """
    return torch.optim.Adam([
        {"params": [gaussians.position], "lr": position_lr},
        {"params": [gaussians.scale],    "lr": 0.005},
        {"params": [gaussians.rotation], "lr": 0.001},
        {"params": [gaussians.sh_dc],    "lr": 0.0025},
        {"params": [gaussians.sh_rest],  "lr": 0.000125},
        {"params": [gaussians.opacity],  "lr": 0.05},
    ])


def train(
    images: list[torch.Tensor], cameras: list[Camera],
    n_gaussians: int = 500, n_iters: int = 30000,
) -> Gaussians3D:
    """Train 3D Gaussians to reconstruct a scene from multiple camera views.

    Follows the original paper's schedule:
      - Loss: 0.8 × L1 + 0.2 × (1 - SSIM)
      - Densification every 100 steps, from step 500 to n_iters/2
      - Opacity reset every 3000 steps (within densification window)
      - Position LR: exponential decay from 0.00016×extent to 0.0000016×extent
      - SH degree progressively activated: 0 → 1 → 2 → 3 every 1000 steps
      - Gradient accumulation between densification steps
    """
    device = get_device()
    images_dev = [img.to(device) for img in images]
    H, W = images[0].shape[0], images[0].shape[1]

    # Estimate scene extent from camera positions
    cam_positions = [-(cam.R.T @ cam.t).numpy() for cam in cameras]
    avg_cam_dist = float(np.mean([np.linalg.norm(c) for c in cam_positions]))
    scene_extent = avg_cam_dist * 0.5

    gaussians = create_random_gaussians_3d(n_gaussians, scene_extent, device=device)

    # Paper's position learning rates (scaled by scene extent)
    pos_lr_init = 0.00016 * scene_extent
    pos_lr_final = 0.0000016 * scene_extent
    optimizer = build_optimizer(gaussians, pos_lr_init)

    # Schedule (matching the paper, scaled to our iteration count)
    densify_from = 500
    densify_until = n_iters // 2
    densify_every = 100
    opacity_reset_every = 3000
    sh_upgrade_every = 1000

    # Gradient accumulation for densification decisions
    N = gaussians.position.shape[0]
    grad_accum = torch.zeros(N, device=device)
    grad_count = torch.zeros(N, device=device)

    # Progressive SH — start with degree 0 (constant color only)
    active_sh_degree = 0

    print(f"Device: {device}")
    print(f"Scene extent: {scene_extent:.2f} (avg camera distance: {avg_cam_dist:.2f})")
    print(f"Training {n_gaussians} Gaussians on {W}×{H} images "
          f"({len(images)} views) for {n_iters} steps...")
    print(f"Densification: steps {densify_from}–{densify_until}, "
          f"opacity reset every {opacity_reset_every}\n")

    t_start = time.perf_counter()

    for step in range(n_iters):
        # Update position learning rate (exponential decay)
        pos_lr = get_position_lr(step, pos_lr_init, pos_lr_final, n_iters)
        for param_group in optimizer.param_groups:
            if param_group["params"][0] is gaussians.position:
                param_group["lr"] = pos_lr
                break

        optimizer.zero_grad()

        # Pick a random training view
        view_idx = random.randint(0, len(images) - 1)
        target = images_dev[view_idx]
        camera = cameras[view_idx]

        # Render and compute loss: 0.8 × L1 + 0.2 × D-SSIM (paper's formula)
        rendered = render(gaussians, camera, active_sh_degree=active_sh_degree)
        l1_loss = (rendered - target).abs().mean()
        ssim_val = ssim(rendered, target)
        loss = 0.8 * l1_loss + 0.2 * (1.0 - ssim_val)

        loss.backward()

        # Accumulate position gradients for densification decisions
        if gaussians.position.grad is not None:
            grad_norm = gaussians.position.grad.detach().norm(dim=1)  # (N,)
            grad_accum[:grad_norm.shape[0]] += grad_norm
            grad_count[:grad_norm.shape[0]] += 1

        optimizer.step()

        with torch.no_grad():
            gaussians.rotation.data = F.normalize(gaussians.rotation.data, dim=1)

        if step % 100 == 0:
            print(f"  Step {step:5d}/{n_iters}  Loss: {loss.item():.6f}  "
                  f"L1: {l1_loss.item():.4f}  SSIM: {ssim_val.item():.4f}  "
                  f"N: {gaussians.position.shape[0]}  SH: {active_sh_degree}")

        # --- Save checkpoint PLY every 1000 steps ---
        if step > 0 and step % 1000 == 0:
            save_ply(gaussians, f"output_{step}.ply")

        # --- Progressive SH activation ---
        if step > 0 and step % sh_upgrade_every == 0 and active_sh_degree < 3:
            active_sh_degree += 1
            print(f"    SH degree → {active_sh_degree}")

        # --- Adaptive density control ---
        in_densify = densify_from <= step < densify_until
        if in_densify and step % densify_every == 0 and step > 0:
            # Compute mean accumulated gradient per Gaussian
            mask = grad_count > 0
            avg_grad = torch.zeros_like(grad_accum)
            avg_grad[mask] = grad_accum[mask] / grad_count[mask]

            # Original paper order: clone → split → prune.
            # Clone only appends, so we extend avg_grad for the new entries.
            # Split and prune don't need avg_grad after they run.
            gaussians = duplicate_3d(gaussians, avg_grad, scene_extent=scene_extent)
            n_added = gaussians.position.shape[0] - avg_grad.shape[0]
            if n_added > 0:
                avg_grad = torch.cat([avg_grad, torch.zeros(n_added, device=device)])
            gaussians = split_3d(gaussians, avg_grad, scene_extent=scene_extent)
            gaussians, _ = prune_3d(gaussians, scene_extent=scene_extent)

            # Rebuild optimizer and gradient accumulators for new Gaussian count
            pos_lr = get_position_lr(step, pos_lr_init, pos_lr_final, n_iters)
            optimizer = build_optimizer(gaussians, pos_lr)
            N = gaussians.position.shape[0]
            grad_accum = torch.zeros(N, device=device)
            grad_count = torch.zeros(N, device=device)

        # --- Opacity reset (paper does every 3000 steps) ---
        if in_densify and step % opacity_reset_every == 0 and step > 0:
            gaussians = reset_opacities(gaussians)
            pos_lr = get_position_lr(step, pos_lr_init, pos_lr_final, n_iters)
            optimizer = build_optimizer(gaussians, pos_lr)

    elapsed = time.perf_counter() - t_start
    print(f"\nDone in {elapsed:.2f}s ({elapsed / n_iters * 1000:.1f} ms/step)")
    return gaussians


# ============================================================================
# Entry Point
# ============================================================================

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python main3d.py <path/to/nerf_synthetic/scene>")
        print("\nDownload NeRF-synthetic data:")
        print("  https://www.kaggle.com/datasets/nguyenhung1903/nerf-synthetic-dataset")
        print("  unzip nerf_synthetic.zip")
        print("  python main3d.py nerf_synthetic/lego")
        return

    data_dir = sys.argv[1]
    images, cameras = load_nerf_dataset(data_dir, image_size=128)

    gaussians = train(images, cameras, n_gaussians=500, n_iters=30000)

    # Render a training view and save alongside the ground truth for comparison
    with torch.no_grad():
        result = render(gaussians, cameras[0])
    save_image(result, "result3d.png")
    save_image(images[0], "gt3d.png")

    # Export as .ply for use in 3DGS viewers (SuperSplat, web viewers, etc.)
    save_ply(gaussians, "output.ply")


if __name__ == "__main__":
    main()
