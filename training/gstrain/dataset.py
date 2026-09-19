"""Torch cameras and scene loading."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from . import cameras as cam_math
from . import colmap
from .params import GaussianParams, create_from_point_cloud
from .undistort import undistort_image


@dataclass
class Camera:
    uid: int
    name: str
    width: int
    height: int
    fov_x: float
    fov_y: float
    world_view_transform: torch.Tensor  # (4,4) row-vector world -> camera
    camera_center: torch.Tensor  # (3,)
    image: torch.Tensor | None  # (3,H,W) in [0,1]
    is_test: bool = False
    cx: float | None = None
    cy: float | None = None

    @property
    def focal_x(self) -> float:
        return self.width / (2.0 * math.tan(self.fov_x * 0.5))

    @property
    def focal_y(self) -> float:
        return self.height / (2.0 * math.tan(self.fov_y * 0.5))

    @property
    def principal_x(self) -> float:
        return self.width * 0.5 if self.cx is None else self.cx

    @property
    def principal_y(self) -> float:
        return self.height * 0.5 if self.cy is None else self.cy

    def to(self, device, dtype=None) -> "Camera":
        return replace(
            self,
            world_view_transform=self.world_view_transform.to(device=device, dtype=dtype),
            camera_center=self.camera_center.to(device=device, dtype=dtype),
            image=None if self.image is None else self.image.to(device=device, dtype=dtype),
        )


def _load_image(info: cam_math.CameraInfo) -> torch.Tensor:
    from PIL import Image

    with Image.open(info.image_path) as img:
        img = img.convert("RGB")
        if img.size != (info.width, info.height):
            # Downscale first: distortion is defined on normalised coordinates, so
            # rectifying the smaller image is equivalent and far cheaper.
            img = img.resize((info.width, info.height), Image.LANCZOS)
        array = np.asarray(img, dtype=np.float32) / 255.0

    if not info.distortion.is_identity:
        array = undistort_image(
            array, info.focal_x, info.focal_y, info.principal_x, info.principal_y, info.distortion
        )
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).contiguous()


def _to_camera(info: cam_math.CameraInfo, image: torch.Tensor | None) -> Camera:
    return Camera(
        uid=info.uid,
        name=info.name,
        width=info.width,
        height=info.height,
        fov_x=float(info.fov_x),
        fov_y=float(info.fov_y),
        world_view_transform=torch.from_numpy(cam_math.world_view_transform(info.R, info.T)),
        camera_center=torch.from_numpy(info.camera_center.astype(np.float32)),
        image=image,
        is_test=info.is_test,
        cx=float(info.principal_x),
        cy=float(info.principal_y),
    )


class Scene:
    def __init__(
        self,
        train_cameras: list[Camera],
        test_cameras: list[Camera],
        init_params: GaussianParams,
        extent: float,
    ) -> None:
        self.train_cameras = train_cameras
        self.test_cameras = test_cameras
        self.init_params = init_params
        self.extent = extent

    @classmethod
    def load(
        cls,
        source,
        images=None,
        resolution: int = 1,
        eval_hold: int = 8,
        sh_degree: int = 3,
        data_device: str = "cpu",
        require_images: bool = True,
        load_images: bool = True,
        max_views: int = 0,
        max_points: int = 0,
        seed: int = 0,
        reporter=None,
    ) -> "Scene":
        from .progress import Reporter

        report = reporter or Reporter(quiet=True)
        source = Path(source)

        report.stage("reading COLMAP model")
        model = colmap.read_model(source)
        images_dir = Path(images) if images is not None else _default_images_dir(source, model.path)

        infos = cam_math.cameras_from_model(model, images_dir, eval_hold=eval_hold)
        _, extent = cam_math.scene_normalization(infos)
        if max_views and len(infos) > max_views:
            # Evenly spaced so the rig still surrounds the scene.
            step = len(infos) / max_views
            infos = [infos[int(i * step)] for i in range(max_views)]
        report.detail(
            f"{len(model.images)} images, {len(model.points.ids)} points, scene extent {extent:.3f}"
        )
        report.done()

        missing = [i.image_path for i in infos if not i.image_path.exists()]
        if missing and require_images:
            listed = "\n  ".join(str(p) for p in missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} of {len(infos)} images are missing under {images_dir}:\n  {listed}"
            )

        reading = load_images and not missing
        sample = infos[0].scaled(resolution)
        if reading:
            report.stage(f"loading images ({len(infos)} at {sample.width}x{sample.height})")
            if not sample.distortion.is_identity:
                report.detail("lens distortion rectified on load")
        else:
            report.stage("preparing cameras (images not loaded)")

        train, test = [], []
        for info in report.track(infos, "decode", total=len(infos)) if reading else infos:
            scaled = info.scaled(resolution)
            image = None
            if reading:
                image = _load_image(scaled).to(data_device)
            camera = _to_camera(scaled, image)
            (test if scaled.is_test else train).append(camera)
        report.done(f"{len(train)} train / {len(test)} held out")

        if not train:
            raise ValueError("no training views; lower --eval-hold or add more images")

        # Sized from the full cloud first: subsampling afterwards would otherwise
        # inflate the kNN spacing and start every splat far too large.
        report.stage("initialising gaussians")
        params = create_from_point_cloud(model.points.xyz, model.points.rgb, sh_degree=sh_degree)
        if max_points and len(params) > max_points:
            pick = np.random.default_rng(seed).choice(len(params), max_points, replace=False)
            params = params.select(np.sort(pick))
        report.done(f"{len(params)} splats, sh degree {sh_degree}")
        return cls(train, test, params, extent)


def _default_images_dir(source: Path, model_path: Path) -> Path:
    for candidate in (source / "images", model_path.parent.parent / "images", source.parent / "images"):
        if candidate.is_dir():
            return candidate
    return source / "images"
