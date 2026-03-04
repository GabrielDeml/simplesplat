import torch 
from dataclasses import dataclass
from PIL import Image
from torchvision import transforms
import numpy as np

# The goal of this is to write a simple 2d splat algroithem from scratch using pytorch.
# It should be able to take in an image a train the gausians

def get_device() -> torch.device:
    """Pick the best available device: CUDA GPU > Apple MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@dataclass
class Gaussians2D: 
    position: torch.Tensor # (N, 2) e.g. [[50.0, 30.0], [80.0, 60.0], ...]  — x, y center
    scale: torch.Tensor    # (N, 2) e.g. [[10.0, 5.0], [8.0, 12.0], ...]   — spread in x, y
    rotation: torch.Tensor # (N,)   e.g. [0.0, 1.57, ...]                   — angle in radians
    color: torch.Tensor    # (N, 3) e.g. [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], ...] — RGB
    opacity: torch.Tensor  # (N,)   e.g. [0.8, 0.5, ...]                    — 0=transparent, 1=solid

    def params(self) -> list[torch.Tensor]:
        return [self.position, self.scale, self.rotation, self.color, self.opacity]

def render(gaussians: Gaussians2D, width: int, height: int) -> torch.Tensor:
    # Create a grid of pixel coordinates — meshgrid gives us the x and y
    # position of every pixel as two (H, W) tensors, so we can do math
    # on all pixels at once instead of looping.
    # For a 4x3 image, xx would be:  [[0,1,2,3],   and yy:  [[0,0,0,0],
    #                                  [0,1,2,3],             [1,1,1,1],
    #                                  [0,1,2,3]]             [2,2,2,2]]
    device = gaussians.position.device
    ys = torch.arange(height, dtype=torch.float32, device=device)  # e.g. [0, 1, 2] for H=3
    xs = torch.arange(width, dtype=torch.float32, device=device)   # e.g. [0, 1, 2, 3] for W=4
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")   # both (H, W)

    # Distance from every pixel to every Gaussian center.
    # The None/broadcasting trick computes all pixel × Gaussian pairs at once:
    #   (H, W, 1) - (1, 1, N) → (H, W, N)
    dx = xx[:, :, None] - gaussians.position[None, None, :, 0]  # (H, W, N)
    dy = yy[:, :, None] - gaussians.position[None, None, :, 1]  # (H, W, N)
    sx = gaussians.scale[:, 0]  # (N,) e.g. [10.0, 8.0, ...]
    sy = gaussians.scale[:, 1]  # (N,) e.g. [5.0, 12.0, ...]

    # Turn distance into strength using the Gaussian formula.
    # This is like a soft spotlight: 1.0 at the center, fading smoothly to 0.
    # sx/sy control how wide the spotlight is in each direction.
    # e.g. 5 pixels away with sx=10: exp(-0.5 * 25/100) = 0.88  (still strong)
    #      30 pixels away with sx=10: exp(-0.5 * 900/100) = 0.01 (basically gone)
    strength = torch.exp(-0.5 * (dx**2 / (sx**2 + 1e-6) + dy**2 / (sy**2 + 1e-6)))  # (H, W, N)

    # Scale strength by opacity — how transparent each Gaussian is
    alpha = gaussians.opacity * strength  # (H, W, N)

    # Multiply each Gaussian's alpha by its color, then sum all N Gaussians
    # together to get the final color at each pixel.
    # (H, W, N, 1) * (1, 1, N, 3) → (H, W, N, 3) → sum over N → (H, W, 3)
    image = (alpha.unsqueeze(-1) * gaussians.color[None, None, :, :]).sum(dim=2)

    return image.clamp(0, 1)



def load_image(image_path: str, size: int = 128) -> torch.Tensor:
    """Load an image, resize it, and return as a float tensor in [0, 1]."""
    image = Image.open(image_path).convert("RGB").resize((size, size))  # type: ignore
    return torch.tensor(np.array(image), dtype=torch.float32) / 255.0  # (H, W, 3)


def create_random_gaussians(n: int, width: int, height: int, device: torch.device = torch.device("cpu")) -> Gaussians2D:
    """Create N Gaussians with random parameters, all with requires_grad=True."""
    return Gaussians2D(
        # Random positions spread across the image
        position=(torch.rand(n, 2, device=device) * torch.tensor([width, height], dtype=torch.float32, device=device)).requires_grad_(True),
        # Start with a reasonable spread (image_size / 20)
        scale=(torch.ones(n, 2, device=device) * (min(width, height) / 20.0)).requires_grad_(True),
        # No rotation to start
        rotation=torch.zeros(n, device=device).requires_grad_(True),
        # Random colors
        color=torch.rand(n, 3, device=device).requires_grad_(True),
        # Start semi-transparent
        opacity=(torch.ones(n, device=device) * 0.5).requires_grad_(True),
    )


def prune(gaussians: Gaussians2D, min_opacity: float = 0.01) -> Gaussians2D:
    """Remove Gaussians that are nearly transparent — they aren't contributing anything
    to the image, so we delete them to save computation."""

    # Check each Gaussian's opacity against the threshold.
    # opacity is (N,) e.g. [0.8, 0.002, 0.5, 0.001, ...]
    # keep is (N,) of booleans e.g. [True, False, True, False, ...]
    keep = gaussians.opacity.detach() > min_opacity

    if keep.all():
        return gaussians  # nothing to prune

    # Use the boolean mask to keep only the Gaussians above the threshold.
    # If we had 200 Gaussians and 10 were below min_opacity, we now have 190.
    # e.g. position was (200, 2), now becomes (190, 2)
    #      color was (200, 3), now becomes (190, 3)
    # .detach() = copy the values, forget the gradient history
    # .requires_grad_(True) = make the new tensors trainable again
    n_before = gaussians.position.shape[0]
    gaussians = Gaussians2D(
        position=gaussians.position.detach()[keep].requires_grad_(True),
        scale=gaussians.scale.detach()[keep].requires_grad_(True),
        rotation=gaussians.rotation.detach()[keep].requires_grad_(True),
        color=gaussians.color.detach()[keep].requires_grad_(True),
        opacity=gaussians.opacity.detach()[keep].requires_grad_(True),
    )
    print(f"    Pruned: {n_before} → {gaussians.position.shape[0]} Gaussians")
    return gaussians


def split(gaussians: Gaussians2D, grad_threshold: float = 0.002, min_scale: float = 8.0) -> Gaussians2D:
    """Split large Gaussians that have high position gradients.

    Why? A large Gaussian covering a big area can't capture fine detail.
    If its gradient is high, it means the optimizer is "struggling" — it wants
    to move the Gaussian but it's too big to fit what it sees.
    Solution: replace it with two smaller Gaussians, each half the size,
    nudged apart from the original center.
    """
    if gaussians.position.grad is None:
        return gaussians

    # How much the optimizer wanted to move each Gaussian's position.
    # .grad is (N, 2) — the gradient for x and y.
    # .norm(dim=1) collapses (N, 2) → (N,) — one magnitude per Gaussian.
    # e.g. grad = [[0.1, -0.05], [0.003, 0.001], ...] → grad_mag = [0.112, 0.003, ...]
    grad_mag = gaussians.position.grad.detach().norm(dim=1)  # (N,)

    # Average scale of each Gaussian (mean of sx and sy).
    # scale is (N, 2) e.g. [[10.0, 8.0], [3.0, 4.0], ...]
    # avg_scale is (N,) e.g. [9.0, 3.5, ...]
    avg_scale = gaussians.scale.detach().mean(dim=1)  # (N,)

    # Split if: high gradient (struggling) AND large scale (covering too much).
    # should_split is (N,) of booleans e.g. [True, False, False, True, ...]
    should_split = (grad_mag > grad_threshold) & (avg_scale > min_scale)

    if not should_split.any():
        return gaussians

    n_splits = should_split.sum().item()

    # The ones we DON'T split — keep them as-is
    keep = ~should_split  # (N,) e.g. [False, True, True, False, ...]

    # Extract the parameters of the Gaussians we're splitting.
    # S = number of Gaussians being split.
    split_pos = gaussians.position.detach()[should_split]       # (S, 2)
    split_scale = gaussians.scale.detach()[should_split] / 2.0  # (S, 2) — half the size!
    split_rot = gaussians.rotation.detach()[should_split]        # (S,)
    split_color = gaussians.color.detach()[should_split]         # (S, 3)
    split_opacity = gaussians.opacity.detach()[should_split]     # (S,)

    # Offset the two children in opposite directions.
    # e.g. if split_scale is [5.0, 4.0], offset is [2.5, 2.0]
    # child A moves to [center_x + 2.5, center_y + 2.0]
    # child B moves to [center_x - 2.5, center_y - 2.0]
    offset = split_scale * 0.5  # (S, 2)
    pos_a = split_pos + offset   # (S, 2)
    pos_b = split_pos - offset   # (S, 2)

    # Combine: the kept originals + two children per split.
    # If we had 200 Gaussians, kept 190, and split 10 → 190 + 10 + 10 = 210
    gaussians = Gaussians2D(
        position=torch.cat([gaussians.position.detach()[keep], pos_a, pos_b]).requires_grad_(True),
        scale=torch.cat([gaussians.scale.detach()[keep], split_scale, split_scale]).requires_grad_(True),
        rotation=torch.cat([gaussians.rotation.detach()[keep], split_rot, split_rot]).requires_grad_(True),
        color=torch.cat([gaussians.color.detach()[keep], split_color, split_color]).requires_grad_(True),
        opacity=torch.cat([gaussians.opacity.detach()[keep], split_opacity, split_opacity]).requires_grad_(True),
    )
    print(f"    Split {n_splits} large Gaussians → now {gaussians.position.shape[0]} total")
    return gaussians


def duplicate(gaussians: Gaussians2D, grad_threshold: float = 0.002, max_scale: float = 8.0) -> Gaussians2D:
    """Duplicate small Gaussians that have high position gradients.

    Why? A small Gaussian in a high-error area means this region needs
    more coverage — one small Gaussian isn't enough. Unlike split (which
    replaces a big one), duplicate KEEPS the original and adds a clone
    nearby with a small random offset.
    """
    if gaussians.position.grad is None:
        return gaussians

    # Same gradient magnitude as split — how much each Gaussian wants to move.
    grad_mag = gaussians.position.grad.detach().norm(dim=1)  # (N,)
    avg_scale = gaussians.scale.detach().mean(dim=1)  # (N,)

    # Duplicate if: high gradient AND small scale (the opposite of split).
    # Small scale = this Gaussian is already detailed, it just needs a friend.
    should_dup = (grad_mag > grad_threshold) & (avg_scale <= max_scale)

    if not should_dup.any():
        return gaussians

    n_dups = should_dup.sum().item()  # e.g. 15 Gaussians to duplicate

    # Clone the selected Gaussians' parameters.
    # Add a small random offset to position so the clone isn't exactly on top.
    # e.g. original at [50.0, 30.0] → clone at [51.3, 28.7]
    dup_pos = gaussians.position.detach()[should_dup] + torch.randn_like(gaussians.position.detach()[should_dup]) * 2.0  # (D, 2)
    dup_scale = gaussians.scale.detach()[should_dup]      # (D, 2) — same size as original
    dup_rot = gaussians.rotation.detach()[should_dup]      # (D,)
    dup_color = gaussians.color.detach()[should_dup]        # (D, 3)
    dup_opacity = gaussians.opacity.detach()[should_dup]    # (D,)

    # Append the clones to ALL existing Gaussians (we keep everything).
    # If we had 200 Gaussians and duplicated 15 → 200 + 15 = 215
    gaussians = Gaussians2D(
        position=torch.cat([gaussians.position.detach(), dup_pos]).requires_grad_(True),
        scale=torch.cat([gaussians.scale.detach(), dup_scale]).requires_grad_(True),
        rotation=torch.cat([gaussians.rotation.detach(), dup_rot]).requires_grad_(True),
        color=torch.cat([gaussians.color.detach(), dup_color]).requires_grad_(True),
        opacity=torch.cat([gaussians.opacity.detach(), dup_opacity]).requires_grad_(True),
    )
    print(f"    Duplicated {n_dups} small Gaussians → now {gaussians.position.shape[0]} total")
    return gaussians


def train(target: torch.Tensor, n_gaussians: int = 200, n_iters: int = 1500, lr: float = 0.01) -> Gaussians2D:
    """
    Train Gaussians to reconstruct the target image.

    target: (H, W, 3) float tensor in [0, 1]
    """
    device = get_device()
    target = target.to(device)
    H, W = target.shape[0], target.shape[1]

    # Create random Gaussians on the same device as the target
    gaussians = create_random_gaussians(n_gaussians, W, H, device=device)
    print(f"Device: {device}")

    # Adam optimizer — uses gaussians.params() to get all the tensors
    optimizer = torch.optim.Adam(gaussians.params(), lr=lr)

    print(f"Training {n_gaussians} Gaussians on {W}x{H} image for {n_iters} steps...\n")

    for step in range(n_iters):
        # Reset gradients from the last step
        optimizer.zero_grad()

        # Render the current Gaussians into an image
        rendered = render(gaussians, W, H)

        # Compare to target — mean squared error (how far off are we?)
        loss = ((rendered - target) ** 2).mean()

        # Backprop — compute how each parameter should change to reduce the loss
        loss.backward()

        # Update parameters in the direction that reduces the loss
        optimizer.step()

        if step % 100 == 0:
            print(f"  Step {step:5d}/{n_iters}  Loss: {loss.item():.6f}  N: {gaussians.position.shape[0]}")

        # Every 100 steps, adapt the Gaussians: prune, split, duplicate
        if step > 0 and step % 100 == 0 and step < n_iters - 100:
            gaussians = prune(gaussians)
            gaussians = split(gaussians)
            gaussians = duplicate(gaussians)
            # Rebuild optimizer since the tensors changed
            optimizer = torch.optim.Adam(gaussians.params(), lr=lr)

    return gaussians


def save_image(tensor: torch.Tensor, path: str) -> None:
    """Save a (H, W, 3) float tensor as an image file."""
    img = (tensor.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)
    print(f"Saved {path}")


def main() -> None:
    import sys

    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path>")
        return

    target = load_image(sys.argv[1], size=256)
    print(f"Loaded {sys.argv[1]}")

    print(f"Target: {target.shape[1]}x{target.shape[0]} pixels\n")

    # Train Gaussians to match the target
    gaussians = train(target, n_gaussians=200, n_iters=1500, lr=0.01)

    # Render final result and save
    with torch.no_grad():
        result = render(gaussians, target.shape[1], target.shape[0])
    save_image(result, "result.png")


if __name__ == "__main__":
    main()