"""Per-checkpoint preview renders, contact sheets and metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .losses import l1_loss, psnr, ssim
from .render import render


def to_image(tensor: torch.Tensor):
    from PIL import Image

    array = (tensor.detach().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    return Image.fromarray(array.permute(1, 2, 0).cpu().numpy())


def _hstack(images):
    from PIL import Image

    width = sum(i.width for i in images)
    height = max(i.height for i in images)
    sheet = Image.new("RGB", (width, height))
    x = 0
    for image in images:
        sheet.paste(image, (x, 0))
        x += image.width
    return sheet


def _vstack(images):
    from PIL import Image

    width = max(i.width for i in images)
    height = sum(i.height for i in images)
    sheet = Image.new("RGB", (width, height))
    y = 0
    for image in images:
        sheet.paste(image, (0, y))
        y += image.height
    return sheet


class PreviewWriter:
    """Renders a fixed set of views so checkpoints are visually comparable."""

    def __init__(self, out_dir: Path, cameras, max_views: int = 4) -> None:
        self.root = Path(out_dir) / "previews"
        self.root.mkdir(parents=True, exist_ok=True)
        if not cameras:
            self.cameras = []
        else:
            step = max(1, len(cameras) // max_views)
            self.cameras = list(cameras[::step])[:max_views]
        self.timeline: list[Path] = []

    @property
    def enabled(self) -> bool:
        return bool(self.cameras)


    @torch.no_grad()
    def write_current_frame(self, model, bg_color: torch.Tensor):
        if not self.enabled:
            return None
        
        folder = self.root / "current"
        folder.mkdir(parents=True, exist_ok=True)

        camera = self.cameras[0]
        cam = camera.to(model.device, model.get_xyz.dtype)
        out = render(cam, model, bg_color)
        rendered = to_image(out.image)
        rendered.save(folder / "current.png")

    @torch.no_grad()
    def write(self, model, iteration: int, bg_color: torch.Tensor, extra: dict | None = None) -> Path | None:
        if not self.enabled:
            return None

        folder = self.root / f"iter_{iteration:06d}"
        folder.mkdir(parents=True, exist_ok=True)

        rows, l1s, psnrs, ssims = [], [], [], []
        first_render = None
        for i, camera in enumerate(self.cameras):
            cam = camera.to(model.device, model.get_xyz.dtype)
            out = render(cam, model, bg_color)
            rendered = to_image(out.image)
            if first_render is None:
                first_render = rendered
            if cam.image is not None:
                l1s.append(float(l1_loss(out.image, cam.image)))
                psnrs.append(float(psnr(out.image, cam.image)))
                ssims.append(float(ssim(out.image, cam.image)))
                rows.append(_hstack([rendered, to_image(cam.image)]))
            else:
                rows.append(rendered)
            rows[-1].save(folder / f"view_{i:02d}.png")

        sheet = _vstack(rows)
        sheet.save(folder / "contact_sheet.png")
        sheet.save(self.root / "latest.png")

        metrics = {
            "iteration": iteration,
            "num_gaussians": len(model),
            "l1": float(np.mean(l1s)) if l1s else None,
            "psnr": float(np.mean(psnrs)) if psnrs else None,
            "ssim": float(np.mean(ssims)) if ssims else None,
        }
        metrics.update(extra or {})
        (folder / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        timeline_frame = folder / "timeline_frame.png"
        first_render.save(timeline_frame)
        self.timeline.append(timeline_frame)
        self._write_timeline()
        return folder

    def _write_timeline(self, max_columns: int = 12) -> None:
        from PIL import Image

        frames = self.timeline
        if len(frames) > max_columns:
            step = len(frames) / max_columns
            frames = [frames[int(i * step)] for i in range(max_columns)]
        images = [Image.open(p).convert("RGB") for p in frames]
        _hstack(images).save(self.root / "timeline.png")
        for image in images:
            image.close()
