import torch 
from dataclasses import dataclass
from PIL import Image
from torchvision import transforms

# The goal of this is to write a simple 2d splat algroithem from scratch using pytorch.
# It should be able to take in an image a train the gausians 

## Goals: 
# 1. Be very simple and easy to understand. 
# 2. Be able to actually work with a real image. 


# Steps:
# [X] Load an image into pytorch tensor
# [x] Define Gaussian parameters (position, scale, rotation, color, opacity)
# [ ] Write a render() function that draws Gaussians into an image
# [ ] Write a training loop (Adam + MSE loss)
# [ ] Save the result

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
    ys = torch.arange(height, dtype=torch.float32)  # e.g. [0, 1, 2] for H=3
    xs = torch.arange(width, dtype=torch.float32)   # e.g. [0, 1, 2, 3] for W=4
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
    import numpy as np
    image = Image.open(image_path).convert("RGB").resize((size, size))  # type: ignore
    return torch.tensor(np.array(image), dtype=torch.float32) / 255.0  # (H, W, 3)


def create_random_gaussians(n: int, width: int, height: int) -> Gaussians2D:
    """Create N Gaussians with random parameters, all with requires_grad=True."""
    return Gaussians2D(
        # Random positions spread across the image
        position=(torch.rand(n, 2) * torch.tensor([width, height], dtype=torch.float32)).requires_grad_(True),
        # Start with a reasonable spread (image_size / 20)
        scale=(torch.ones(n, 2) * (min(width, height) / 20.0)).requires_grad_(True),
        # No rotation to start
        rotation=torch.zeros(n).requires_grad_(True),
        # Random colors
        color=torch.rand(n, 3).requires_grad_(True),
        # Start semi-transparent
        opacity=(torch.ones(n) * 0.5).requires_grad_(True),
    )


def train(target: torch.Tensor, n_gaussians: int = 200, n_iters: int = 1500, lr: float = 0.01) -> Gaussians2D:
    """
    Train Gaussians to reconstruct the target image.

    target: (H, W, 3) float tensor in [0, 1]
    """
    H, W = target.shape[0], target.shape[1]

    # Create random Gaussians
    gaussians = create_random_gaussians(n_gaussians, W, H)

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
            print(f"  Step {step:5d}/{n_iters}  Loss: {loss.item():.6f}")

    return gaussians


def save_image(tensor: torch.Tensor, path: str) -> None:
    """Save a (H, W, 3) float tensor as an image file."""
    import numpy as np
    img = (tensor.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)
    print(f"Saved {path}")


def main() -> None:
    import os

    # Load target image
    if os.path.exists("image.jpg"):
        target = load_image("image.jpg")
        print("Loaded image.jpg")
    elif os.path.exists("image.png"):
        target = load_image("image.png")
        print("Loaded image.png")
    else:
        print("No image found! Put an image.jpg or image.png in this folder.")
        return

    print(f"Target: {target.shape[1]}x{target.shape[0]} pixels\n")

    # Train Gaussians to match the target
    gaussians = train(target, n_gaussians=200, n_iters=1500, lr=0.01)

    # Render final result and save
    with torch.no_grad():
        result = render(gaussians, target.shape[1], target.shape[0])
    save_image(result, "result.png")


if __name__ == "__main__":
    main()