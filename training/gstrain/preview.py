from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .model_torch import TorchGaussianModel
from .render.renderer_torch import render_gaussians
from .train import TrainConfig, TrainView

LABEL_HEIGHT = 14
GAP = 6


def to_pixels(image: torch.Tensor) -> np.ndarray:
    """(H, W, 3) float in [0, 1] -> (H, W, 3) uint8."""
    values = image.detach().clamp(0.0, 1.0).cpu().numpy()

    return (values * 255.0 + 0.5).astype(np.uint8)


def save_image(image: torch.Tensor, path: str | Path, quality: int = 95) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    picture = Image.fromarray(to_pixels(image))

    if path.suffix.lower() in {".jpg", ".jpeg"}:
        picture.save(path, quality=quality)
    else:
        picture.save(path)

    return path


def error_map(rendered: torch.Tensor, target: torch.Tensor, gain: float = 4.0) -> torch.Tensor:
    """Absolute difference, amplified so small errors are actually visible."""
    return ((rendered - target).abs() * gain).clamp(0.0, 1.0)


def _label(picture: Image.Image, panels: list[tuple[str, int]]) -> None:
    draw = ImageDraw.Draw(picture)

    for text, left in panels:
        draw.rectangle([left, 0, left + 6 * len(text) + 6, LABEL_HEIGHT], fill=(0, 0, 0))
        draw.text((left + 3, 2), text, fill=(255, 255, 255))


def compose(panels: list[tuple[str, torch.Tensor]], labels: bool = True) -> Image.Image:
    """Lay panels out left to right on a dark strip."""
    arrays = [to_pixels(image) for _, image in panels]

    height = max(array.shape[0] for array in arrays)
    width = sum(array.shape[1] for array in arrays) + GAP * (len(arrays) - 1)
    top = LABEL_HEIGHT if labels else 0

    canvas = np.zeros((height + top, width, 3), dtype=np.uint8)

    positions = []
    left = 0
    for array in arrays:
        canvas[top : top + array.shape[0], left : left + array.shape[1]] = array
        positions.append(left)
        left += array.shape[1] + GAP

    picture = Image.fromarray(canvas)

    if labels:
        _label(picture, [(name, x) for (name, _), x in zip(panels, positions)])

    return picture


def save_comparison(
    rendered: torch.Tensor,
    target: torch.Tensor,
    path: str | Path,
    include_error: bool = True,
    gain: float = 4.0,
    quality: int = 95,
) -> Path:
    """Write [target | render | error] as one image."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    target = target.to(rendered.device)

    panels = [("target", target), ("render", rendered)]
    if include_error:
        panels.append((f"error x{gain:g}", error_map(rendered, target, gain)))

    picture = compose(panels)

    if path.suffix.lower() in {".jpg", ".jpeg"}:
        picture.save(path, quality=quality)
    else:
        picture.save(path)

    return path


def save_view_previews(
    model: TorchGaussianModel,
    views: list[TrainView],
    directory: str | Path,
    prefix: str = "view",
    limit: int | None = None,
    suffix: str = ".jpg",
    gain: float = 4.0,
    **render_kwargs,
) -> list[Path]:
    """One comparison image per view."""
    directory = Path(directory)
    written: list[Path] = []

    for index, view in enumerate(views[:limit]):
        with torch.no_grad():
            rendered = render_gaussians(view.camera, model, **render_kwargs)

        written.append(
            save_comparison(
                rendered,
                view.image,
                directory / f"{prefix}_{index:03d}{suffix}",
                gain=gain,
            )
        )

    return written


def save_progress_preview(
    model: TorchGaussianModel,
    view: TrainView,
    directory: str | Path,
    step: int,
    prefix: str = "step",
    config: TrainConfig = TrainConfig() # basic config
) -> Path:
    """Snapshot one view mid-training; call from the on_iteration callback."""
    with torch.no_grad():
        rendered = render_gaussians(
            view.camera, 
            model, 
            tile_size=config.tile_size,
            tile_chunk=config.tile_chunk, 
            max_per_tile=config.max_per_tile
        )

    return save_comparison(rendered, view.image, Path(directory) / f"{prefix}_{step:06d}.jpg")
