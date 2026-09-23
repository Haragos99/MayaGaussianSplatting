"""The 3DGS training loop.

Everything this writes is the contract for any external UI:
    progress.jsonl                            one JSON object per logged iteration
    status.json                               running | done | error
    point_cloud/iteration_<N>/point_cloud.ply loadable by the Maya plugin
    chkpnt<N>.pt                              resumable checkpoints
    previews/                                 PNG previews + metrics
"""

from __future__ import annotations

import json
import os
import random
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .dataset import Scene
from .export_ply import write_ply
from .losses import photometric_loss, psnr
from .model import GaussianModel, OptimizationConfig
from .preview import PreviewWriter
from .render import render


@dataclass
class TrainConfig:
    source: Path
    model: Path
    images: Path | None = None
    resolution: int = 1
    iterations: int = 30_000
    sh_degree: int = 3
    device: str = "auto"
    data_device: str = "cpu"
    eval_hold: int = 8
    background: str = "black"
    seed: int = 0
    up_axis: str = "colmap"
    element_budget: int = 4_000_000
    log_every: int = 10
    preview_every: int = 500
    preview_views: int = 4
    save_iterations: tuple[int, ...] = (7_000, 30_000)
    checkpoint_every: int = 0
    resume: Path | None = None
    quiet: bool = False
    max_views: int = 0
    max_points: int = 0
    # adaptive density control
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densification_interval: int = 100
    opacity_reset_interval: int = 3_000
    densify_grad_threshold: float = 2e-4
    percent_dense: float = 0.01
    min_opacity: float = 0.005
    max_screen_size: int = 20
    sh_degree_interval: int = 1_000
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def background_tensor(name: str, device) -> torch.Tensor:
    if name == "white":
        return torch.ones(3, device=device)
    if name == "black":
        return torch.zeros(3, device=device)
    raise ValueError(f"unknown background {name!r}, expected 'black' or 'white'")


def _progress_bar(cfg: "TrainConfig", start_iteration: int):
    if cfg.quiet:
        return None
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    return tqdm(total=cfg.iterations, initial=start_iteration - 1, unit="it", dynamic_ncols=True)


class RunWriter:
    """Append-only progress log the UI can tail while training runs."""

    def __init__(self, model_dir: Path) -> None:
        self.dir = Path(model_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.progress = open(self.dir / "progress.jsonl", "a", encoding="utf-8")

    def log(self, record: dict) -> None:
        self.progress.write(json.dumps(record) + "\n")
        self.progress.flush()
        os.fsync(self.progress.fileno())

    def status(self, state: str, **fields) -> None:
        payload = {"state": state, "time": time.time(), **fields}
        (self.dir / "status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def close(self) -> None:
        self.progress.close()


def export_checkpoint_ply(model: GaussianModel, model_dir: Path, iteration: int, up_axis: str):
    """Writes the versioned checkpoint and refreshes <model_dir>/point_cloud.ply."""
    model_dir = Path(model_dir)
    params = model.to_params()
    versioned = write_ply(
        model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply", params, up_axis=up_axis
    )
    root = write_ply(model_dir / "point_cloud.ply", params, up_axis=up_axis)
    return versioned, root


def train(cfg: TrainConfig) -> Path:
    from .progress import Reporter

    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    model_dir = Path(cfg.model)
    writer = RunWriter(model_dir)
    (model_dir / "cfg_args.json").write_text(
        json.dumps({k: str(v) for k, v in asdict(cfg).items()}, indent=2), encoding="utf-8"
    )
    writer.status("running", iteration=0)

    report = Reporter(total_stages=4, quiet=cfg.quiet)
    report.banner(f"gstrain: {cfg.source}\n     -> {model_dir}")
    iteration = 0

    try:
        scene = Scene.load(
            cfg.source,
            images=cfg.images,
            resolution=cfg.resolution,
            eval_hold=cfg.eval_hold,
            sh_degree=cfg.sh_degree,
            data_device=cfg.data_device,
            max_views=cfg.max_views,
            max_points=cfg.max_points,
            seed=cfg.seed,
            reporter=report,
        )

        opt = cfg.optimization
        opt.position_lr_max_steps = cfg.iterations
        opt.percent_dense = cfg.percent_dense
        opt.densify_grad_threshold = cfg.densify_grad_threshold
        opt.min_opacity = cfg.min_opacity
        opt.max_screen_size = cfg.max_screen_size

        start_iteration = 1
        if cfg.resume is not None:
            state = torch.load(cfg.resume, map_location=device, weights_only=False)
            model, done = GaussianModel.restore(state, opt, device)
            start_iteration = done + 1
        else:
            model = GaussianModel.from_params(scene.init_params, scene.extent, device)
            model.training_setup(opt)

        bg = background_tensor(cfg.background, device)
        preview = PreviewWriter(model_dir, scene.test_cameras or scene.train_cameras, cfg.preview_views)

        report.stage(f"training {cfg.iterations} iterations on {device}")
        camera0 = scene.train_cameras[0]
        report.detail(
            f"{camera0.width}x{camera0.height}, densify {cfg.densify_from_iter}-{cfg.densify_until_iter}"
            f" every {cfg.densification_interval}, preview every {cfg.preview_every or 'never'}"
        )

        view_stack: list = []
        started = time.time()
        save_at = set(cfg.save_iterations) - {cfg.iterations}
        bar = _progress_bar(cfg, start_iteration)

        def note(message: str) -> None:
            if cfg.quiet:
                return
            # bar.write keeps the progress bar from being torn apart by prints.
            (bar.write if bar is not None else print)(f"      {message}")

        for iteration in range(start_iteration, cfg.iterations + 1):
            lr = model.update_learning_rate(iteration)
            if iteration % cfg.sh_degree_interval == 0 and model.active_sh_degree < model.max_sh_degree:
                model.oneup_sh_degree()
                note(f"iter {iteration}: sh degree -> {model.active_sh_degree}")

            if not view_stack:
                view_stack = list(scene.train_cameras)
                rng.shuffle(view_stack)
            camera = view_stack.pop().to(device)

            out = render(camera, model, bg, element_budget=cfg.element_budget)
            loss, l1 = photometric_loss(out.image, camera.image)
            loss.backward()

            with torch.no_grad():
                visible = out.visibility_filter
                model.max_radii2D[visible] = torch.maximum(
                    model.max_radii2D[visible], out.radii[visible].to(model.max_radii2D.dtype)
                )
                grad = out.viewspace_points.grad
                if grad is not None:
                    # The reference implementation accumulates NDC gradients, so the
                    # 2e-4 threshold only holds once pixel gradients are rescaled.
                    scale = torch.tensor(
                        [camera.width * 0.5, camera.height * 0.5], device=grad.device, dtype=grad.dtype
                    )
                    model.add_densification_stats(grad * scale, visible)

                if iteration < cfg.densify_until_iter:
                    if (
                        iteration > cfg.densify_from_iter
                        and iteration % cfg.densification_interval == 0
                    ):
                        size_threshold = (
                            cfg.max_screen_size if iteration > cfg.opacity_reset_interval else None
                        )
                        model.densify_and_prune(scene.extent, size_threshold)
                    if iteration % cfg.opacity_reset_interval == 0:
                        model.reset_opacity()

                model.optimizer.step()
                model.optimizer.zero_grad(set_to_none=True)

            if iteration % cfg.log_every == 0 or iteration == cfg.iterations:
                with torch.no_grad():
                    quality = float(psnr(out.image, camera.image))
                record = {
                    "iter": iteration,
                    "loss": float(loss.detach()),
                    "l1": float(l1.detach()),
                    "psnr": quality,
                    "num_gaussians": len(model),
                    "lr_xyz": lr,
                    "elapsed_s": round(time.time() - started, 2),
                }
                writer.log(record)
                writer.status("running", iteration=iteration, num_gaussians=len(model))
                if bar is not None:
                    bar.set_postfix(
                        loss=f"{record['loss']:.4f}",
                        psnr=f"{quality:.2f}",
                        splats=len(model),
                        refresh=False,
                    )

            # Save the actual iterion of the model as a .png
            preview.write_current_frame(model, bg)

            if bar is not None:
                bar.update(1)

            if cfg.preview_every and iteration % cfg.preview_every == 0:
                preview.write(model, iteration, bg, {"elapsed_s": round(time.time() - started, 2)})
                note(f"iter {iteration}: preview -> previews/iter_{iteration:06d}")

            if iteration in save_at:
                export_checkpoint_ply(model, model_dir, iteration, cfg.up_axis)
                note(f"iter {iteration}: exported point_cloud/iteration_{iteration}")
            if cfg.checkpoint_every and iteration % cfg.checkpoint_every == 0:
                torch.save(model.capture(iteration), model_dir / f"chkpnt{iteration}.pt")
                note(f"iter {iteration}: checkpoint -> chkpnt{iteration}.pt")

        if bar is not None:
            bar.close()

        versioned, final = export_checkpoint_ply(model, model_dir, cfg.iterations, cfg.up_axis)
        writer.status("done", iteration=cfg.iterations, num_gaussians=len(model), ply=str(final))
        if not cfg.quiet:
            print(
                f"\ndone in {time.time() - started:.1f}s, {len(model)} gaussians"
                f"\n  {final}\n  {versioned}",
                flush=True,
            )
        return final

    except KeyboardInterrupt:
        # BaseException, so the generic handler below never sees it.
        writer.status("cancelled", iteration=iteration)
        report.banner(f"\ncancelled at iteration {iteration}")
        raise
    except Exception as error:  # surfaced through status.json for any external UI
        writer.status("error", message=str(error), traceback=traceback.format_exc())
        raise
    finally:
        writer.close()
