"""The optimizable Gaussian model: parameters, learning rates, density control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .optim_utils import cat_tensors_to_optimizer, prune_optimizer, replace_tensor_to_optimizer
from .params import GaussianParams
from .torch_utils import build_rotation, get_expon_lr_func, inverse_sigmoid

PARAM_NAMES = ("xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation")


@dataclass
class OptimizationConfig:
    position_lr_init: float = 1.6e-4
    position_lr_final: float = 1.6e-6
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    feature_lr: float = 2.5e-3
    opacity_lr: float = 0.05
    scaling_lr: float = 5e-3
    rotation_lr: float = 1e-3
    percent_dense: float = 0.01
    densify_grad_threshold: float = 2e-4
    min_opacity: float = 0.005
    max_screen_size: int = 20


class GaussianModel:
    def __init__(self, sh_degree: int = 3, device: str | torch.device = "cpu") -> None:
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0
        self.device = torch.device(device)
        self.spatial_lr_scale = 1.0
        self.optimizer: torch.optim.Optimizer | None = None
        self.cfg = OptimizationConfig()

        empty = torch.empty(0, device=self.device)
        self._xyz = nn.Parameter(empty.reshape(0, 3).clone())
        self._features_dc = nn.Parameter(empty.reshape(0, 1, 3).clone())
        self._features_rest = nn.Parameter(empty.reshape(0, 0, 3).clone())
        self._opacity = nn.Parameter(empty.reshape(0, 1).clone())
        self._scaling = nn.Parameter(empty.reshape(0, 3).clone())
        self._rotation = nn.Parameter(empty.reshape(0, 4).clone())

        self.xyz_gradient_accum = torch.zeros((0, 1), device=self.device)
        self.denom = torch.zeros((0, 1), device=self.device)
        self.max_radii2D = torch.zeros((0,), device=self.device)

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_params(
        cls, params: GaussianParams, spatial_lr_scale: float, device="cpu", dtype=torch.float32
    ) -> "GaussianModel":
        model = cls(params.sh_degree, device)
        model.spatial_lr_scale = float(spatial_lr_scale)

        def p(array):
            return nn.Parameter(torch.from_numpy(np.ascontiguousarray(array)).to(model.device, dtype))

        model._xyz = p(params.xyz)
        model._features_dc = p(params.features_dc)
        model._features_rest = p(params.features_rest)
        model._opacity = p(params.opacity)
        model._scaling = p(params.scaling)
        model._rotation = p(params.rotation)
        model._reset_stats()
        return model

    def to_params(self) -> GaussianParams:
        """Raw parameters as numpy, ready for export_ply.write_ply."""

        def a(tensor):
            return tensor.detach().cpu().float().numpy()

        return GaussianParams(
            xyz=a(self._xyz),
            features_dc=a(self._features_dc),
            features_rest=a(self._features_rest),
            opacity=a(self._opacity),
            scaling=a(self._scaling),
            rotation=a(self._rotation),
        )

    # ------------------------------------------------------------------ #
    # activated views
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return int(self._xyz.shape[0])

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_scaling(self) -> torch.Tensor:
        return torch.exp(self._scaling)

    @property
    def get_rotation(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self._rotation, dim=-1)

    @property
    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self._opacity)

    @property
    def get_features(self) -> torch.Tensor:
        return torch.cat([self._features_dc, self._features_rest], dim=1)

    def oneup_sh_degree(self) -> None:
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # ------------------------------------------------------------------ #
    # optimizer
    # ------------------------------------------------------------------ #

    def training_setup(self, cfg: OptimizationConfig | None = None) -> None:
        if cfg is not None:
            self.cfg = cfg
        c = self.cfg
        self._reset_stats()
        groups = [
            {"params": [self._xyz], "lr": c.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {"params": [self._features_dc], "lr": c.feature_lr, "name": "f_dc"},
            {"params": [self._features_rest], "lr": c.feature_lr / 20.0, "name": "f_rest"},
            {"params": [self._opacity], "lr": c.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": c.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": c.rotation_lr, "name": "rotation"},
        ]
        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        self.xyz_scheduler = get_expon_lr_func(
            lr_init=c.position_lr_init * self.spatial_lr_scale,
            lr_final=c.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=c.position_lr_delay_mult,
            max_steps=c.position_lr_max_steps,
        )

    def update_learning_rate(self, iteration: int) -> float:
        for group in self.optimizer.param_groups:
            if group["name"] == "xyz":
                group["lr"] = self.xyz_scheduler(iteration)
                return group["lr"]
        return 0.0

    def _reset_stats(self) -> None:
        n = len(self)
        self.xyz_gradient_accum = torch.zeros((n, 1), device=self.device)
        self.denom = torch.zeros((n, 1), device=self.device)
        self.max_radii2D = torch.zeros((n,), device=self.device)

    def _adopt(self, tensors: dict[str, nn.Parameter]) -> None:
        self._xyz = tensors["xyz"]
        self._features_dc = tensors["f_dc"]
        self._features_rest = tensors["f_rest"]
        self._opacity = tensors["opacity"]
        self._scaling = tensors["scaling"]
        self._rotation = tensors["rotation"]

    # ------------------------------------------------------------------ #
    # adaptive density control
    # ------------------------------------------------------------------ #

    def add_densification_stats(self, viewspace_grad_ndc: torch.Tensor, visible: torch.Tensor) -> None:
        grad = torch.norm(viewspace_grad_ndc[visible, :2], dim=-1, keepdim=True)
        self.xyz_gradient_accum[visible] += grad
        self.denom[visible] += 1

    def reset_opacity(self) -> None:
        capped = torch.min(self.get_opacity, torch.full_like(self._opacity, 0.01))
        self._opacity = replace_tensor_to_optimizer(self.optimizer, inverse_sigmoid(capped), "opacity")

    def prune_points(self, mask: torch.Tensor) -> None:
        keep = ~mask
        self._adopt(prune_optimizer(self.optimizer, keep))
        self.xyz_gradient_accum = self.xyz_gradient_accum[keep]
        self.denom = self.denom[keep]
        self.max_radii2D = self.max_radii2D[keep]

    def _densification_postfix(self, tensors: dict[str, torch.Tensor]) -> None:
        self._adopt(cat_tensors_to_optimizer(self.optimizer, tensors))
        self._reset_stats()

    def _densify_and_clone(self, grads: torch.Tensor, max_grad: float, extent: float) -> None:
        selected = torch.norm(grads, dim=-1) >= max_grad
        selected &= self.get_scaling.max(dim=1).values <= self.cfg.percent_dense * extent
        if not bool(selected.any()):
            return
        self._densification_postfix(
            {
                "xyz": self._xyz[selected],
                "f_dc": self._features_dc[selected],
                "f_rest": self._features_rest[selected],
                "opacity": self._opacity[selected],
                "scaling": self._scaling[selected],
                "rotation": self._rotation[selected],
            }
        )

    def _densify_and_split(self, grads: torch.Tensor, max_grad: float, extent: float, n_split: int = 2) -> None:
        n_before = len(self)
        # Cloning already appended rows, so the gradient vector is shorter than the model.
        padded = torch.zeros((n_before,), device=self.device)
        padded[: grads.shape[0]] = grads.squeeze(-1)
        selected = padded >= max_grad
        selected &= self.get_scaling.max(dim=1).values > self.cfg.percent_dense * extent
        if not bool(selected.any()):
            return

        stds = self.get_scaling[selected].repeat(n_split, 1)
        samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
        rotations = build_rotation(self._rotation[selected]).repeat(n_split, 1, 1)
        new_xyz = torch.bmm(rotations, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected].repeat(
            n_split, 1
        )
        new_scaling = torch.log(self.get_scaling[selected].repeat(n_split, 1) / (0.8 * n_split))

        self._densification_postfix(
            {
                "xyz": new_xyz,
                "f_dc": self._features_dc[selected].repeat(n_split, 1, 1),
                "f_rest": self._features_rest[selected].repeat(n_split, 1, 1),
                "opacity": self._opacity[selected].repeat(n_split, 1),
                "scaling": new_scaling,
                "rotation": self._rotation[selected].repeat(n_split, 1),
            }
        )
        prune = torch.cat(
            [selected, torch.zeros(n_split * int(selected.sum()), dtype=torch.bool, device=self.device)]
        )
        self.prune_points(prune)

    def densify_and_prune(self, extent: float, max_screen_size: int | None) -> None:
        grads = self.xyz_gradient_accum / self.denom.clamp(min=1.0)
        grads = torch.nan_to_num(grads, nan=0.0)

        self._densify_and_clone(grads, self.cfg.densify_grad_threshold, extent)
        self._densify_and_split(grads, self.cfg.densify_grad_threshold, extent)

        prune = (self.get_opacity < self.cfg.min_opacity).squeeze(-1)
        if max_screen_size is not None:
            prune |= self.max_radii2D > max_screen_size
            prune |= self.get_scaling.max(dim=1).values > 0.1 * extent
        if bool(prune.any()):
            self.prune_points(prune)

    # ------------------------------------------------------------------ #
    # checkpoints
    # ------------------------------------------------------------------ #

    def capture(self, iteration: int) -> dict:
        return {
            "iteration": iteration,
            "active_sh_degree": self.active_sh_degree,
            "max_sh_degree": self.max_sh_degree,
            "spatial_lr_scale": self.spatial_lr_scale,
            "xyz": self._xyz.detach().cpu(),
            "f_dc": self._features_dc.detach().cpu(),
            "f_rest": self._features_rest.detach().cpu(),
            "opacity": self._opacity.detach().cpu(),
            "scaling": self._scaling.detach().cpu(),
            "rotation": self._rotation.detach().cpu(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "xyz_gradient_accum": self.xyz_gradient_accum.cpu(),
            "denom": self.denom.cpu(),
            "max_radii2D": self.max_radii2D.cpu(),
        }

    @classmethod
    def restore(cls, state: dict, cfg: OptimizationConfig, device="cpu") -> tuple["GaussianModel", int]:
        model = cls(state["max_sh_degree"], device)
        model.active_sh_degree = state["active_sh_degree"]
        model.spatial_lr_scale = state["spatial_lr_scale"]
        for attr, key in (
            ("_xyz", "xyz"),
            ("_features_dc", "f_dc"),
            ("_features_rest", "f_rest"),
            ("_opacity", "opacity"),
            ("_scaling", "scaling"),
            ("_rotation", "rotation"),
        ):
            setattr(model, attr, nn.Parameter(state[key].to(model.device)))
        model.training_setup(cfg)
        if state["optimizer"] is not None:
            model.optimizer.load_state_dict(state["optimizer"])
        model.xyz_gradient_accum = state["xyz_gradient_accum"].to(model.device)
        model.denom = state["denom"].to(model.device)
        model.max_radii2D = state["max_radii2D"].to(model.device)
        return model, int(state["iteration"])
