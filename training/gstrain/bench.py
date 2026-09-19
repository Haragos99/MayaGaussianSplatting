"""Rasterizer benchmark: forward and backward timing, no images required."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .dataset import Scene
from .model import GaussianModel
from .render import render


@dataclass
class BenchResult:
    device: str
    width: int
    height: int
    splats: int
    forward_ms: float
    backward_ms: float

    @property
    def total_ms(self) -> float:
        return self.forward_ms + self.backward_ms

    def __str__(self) -> str:
        return (
            f"{self.device:<5} {self.width}x{self.height} {self.splats:>8} splats  "
            f"forward {self.forward_ms:7.1f} ms   +backward {self.total_ms:7.1f} ms   "
            f"({1000.0 / self.total_ms:5.2f} it/s)"
        )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark(
    source,
    resolution: int = 8,
    max_points: int = 0,
    device: str = "cpu",
    steps: int = 5,
    warmup: int = 2,
    sh_degree: int = 3,
    element_budget: int = 4_000_000,
) -> BenchResult:
    dev = torch.device(device)
    scene = Scene.load(
        Path(source),
        resolution=resolution,
        eval_hold=0,
        sh_degree=sh_degree,
        max_points=max_points,
        max_views=1,
        load_images=False,
    )
    model = GaussianModel.from_params(scene.init_params, scene.extent, dev)
    model.active_sh_degree = sh_degree
    for tensor in (model._xyz, model._features_dc, model._features_rest, model._opacity,
                   model._scaling, model._rotation):
        tensor.requires_grad_(True)

    camera = scene.train_cameras[0].to(dev)
    bg = torch.zeros(3, device=dev)

    def one(backward: bool) -> float:
        for tensor in model._xyz, model._scaling:
            tensor.grad = None
        _sync(dev)
        start = time.perf_counter()
        out = render(camera, model, bg, element_budget=element_budget)
        if backward:
            out.image.sum().backward()
        _sync(dev)
        return (time.perf_counter() - start) * 1000.0

    for _ in range(warmup):
        one(True)

    forward = min(one(False) for _ in range(steps))
    total = min(one(True) for _ in range(steps))

    return BenchResult(
        device=str(dev),
        width=camera.width,
        height=camera.height,
        splats=len(model),
        forward_ms=forward,
        backward_ms=max(0.0, total - forward),
    )
