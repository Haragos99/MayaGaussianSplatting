import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .colmap import load_colmap
from .dataset import build_views, split_views
from .densification import DensificationConfig
from .export_ply import UP_AXES, verify_ply, write_ply
from .model import initialize_gaussians
from .model_torch import TorchGaussianModel
from .preview import save_comparison
from .render.renderer_torch import render_gaussians
from .run_log import RunLog
from .strategy_mcmc import MCMCConfig
from .torch_utils import choose_device, describe_device
from .train import BACKGROUNDS, STRATEGIES, TrainConfig, background_color, render_options, scene_extent, train


def _images_dir(data, images: Path | None) -> Path:
    images_dir = images or data.path.parent.parent / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {images_dir} (pass --images)")
    return images_dir


def build_config(args) -> TrainConfig:
    densification = DensificationConfig.for_iterations(
        args.iterations,
        max_gaussians=args.max_gaussians,
        max_screen_fraction=args.max_screen_fraction,
        seed=args.seed,
    )
    mcmc = MCMCConfig(cap_max=args.cap_max, seed=args.seed)
    return TrainConfig(
        iterations=args.iterations,
        background=args.background,
        strategy=args.strategy,
        densification=densification,
        mcmc=mcmc,
        prune_contribution=args.prune_contribution,
        prune_finetune=args.prune_finetune,
        eval_every=args.eval_every,
        log_every=args.log_every,
        sh_degree_interval=args.sh_degree_interval,
        seed=args.seed,
    )


def command_train(args) -> Path:
    out = Path(args.model)
    run_log = RunLog(out)
    run_log.status("starting")
    config = build_config(args)
    run_log.write_json("cfg_args.json", {k: str(v) for k, v in vars(args).items() if k != "func"})

    try:
        device = choose_device()
        print(describe_device(device))

        data = load_colmap(args.source)
        images_dir = _images_dir(data, args.images)
        views = build_views(data, images_dir, max_edge=args.max_edge, device=device, limit=args.max_views or None)
        train_views, test_views = split_views(views, args.eval_hold)
        camera = views[0].camera
        print(
            f"{len(data.images)} images -> {len(train_views)} train / {len(test_views)} test views "
            f"at {camera.width}x{camera.height}; {len(data.points3D.xyz):,} COLMAP points"
        )

        if args.resume:
            model, resume = load_checkpoint(args.resume, device)
            print(f"resuming from {args.resume} at iteration {resume['iteration']}")
        else:
            resume = None
            mcmc = args.strategy == "mcmc"
            initial = initialize_gaussians(
                data.points3D,
                initial_opacity=config.mcmc.initial_opacity if mcmc else 0.1,
                max_points=args.max_points or None,
                seed=args.seed,
            )
            if mcmc:
                initial.scale = initial.scale * config.mcmc.initial_scale_multiplier
            model = TorchGaussianModel.from_numpy(initial, device=device, sh_degree=args.sh_degree)
        print(f"{len(model):,} starting splats, SH degree up to {model.max_sh_degree}")

        preview_view = test_views[0] if test_views else train_views[0]
        save_at = set(args.save_iterations)

        def on_iteration(step: int, _loss: float) -> None:
            iteration = step + 1
            if args.preview_every and iteration % args.preview_every == 0:
                with torch.no_grad():
                    name = "black" if config.background == "random" else config.background
                    rendered = render_gaussians(
                        preview_view.camera, model,
                        background=background_color(name, device), **render_options(config),
                    )
                save_comparison(rendered, preview_view.image, out / "previews" / f"iter_{iteration:06d}.jpg")
            if iteration in save_at:
                write_ply(out / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply", model, args.up_axis)

        def checkpoint(iteration: int, model_, optimizer) -> None:
            save_checkpoint(out / f"chkpnt{iteration}.pt", model_, optimizer, iteration)

        started = time.time()
        history = train(
            model,
            train_views,
            config=config,
            extent=scene_extent(views),
            on_iteration=on_iteration,
            test_views=test_views,
            run_log=run_log,
            checkpoint=checkpoint,
            checkpoint_every=args.checkpoint_every,
            resume=resume,
        )
        elapsed = time.time() - started

        final_iteration = len(history) + (resume["iteration"] if resume else 0)
        write_ply(out / "point_cloud" / f"iteration_{final_iteration}" / "point_cloud.ply", model, args.up_axis)
        final = write_ply(out / "point_cloud.ply", model, args.up_axis)

        from .train import evaluate
        metrics = {
            "iterations": final_iteration,
            "splats": len(model),
            "seconds": round(elapsed, 1),
            "train_views": len(train_views),
            "test_views": len(test_views),
            "resolution": [camera.width, camera.height],
            "strategy": config.strategy,
            **({f"test_{k}": v for k, v in evaluate(model, test_views, config).items()} if test_views else {}),
        }
        run_log.write_json("metrics.json", metrics)
        run_log.status("done", iteration=final_iteration, splats=len(model), ply=str(final))
        print(json.dumps(metrics, indent=2))
        print(f"wrote {final}")
        return final
    except KeyboardInterrupt:
        run_log.status("cancelled")
        raise
    except Exception as error:
        run_log.status("error", message=str(error), traceback=traceback.format_exc())
        raise
    finally:
        run_log.close()


def command_verify(args) -> None:
    info = verify_ply(args.ply)
    opacity = info["opacity"]
    scale = info["scale"].max(axis=1)
    print(f"{args.ply}: OK")
    print(f"  splats       {info['count']:,}")
    print(f"  SH degree    {info['sh_degree']}")
    print(f"  opacity      median {np.median(opacity):.3f}, < 0.1: {(opacity < 0.1).mean():.1%}")
    print(f"  scale        median {np.median(scale):.4g}, p95 {np.percentile(scale, 95):.4g}")
    low, high = info["xyz"].min(axis=0), info["xyz"].max(axis=0)
    print(f"  bounds       {np.round(low, 3)} .. {np.round(high, 3)}")


def command_stats(args) -> None:
    data = load_colmap(args.source)
    cameras = {camera.model for camera in data.cameras.values()}
    first = next(iter(data.cameras.values()))
    views = build_views(data, _images_dir(data, args.images), max_edge=64, device=torch.device("cpu"))
    train_views, test_views = split_views(views, args.eval_hold)
    print(f"COLMAP model   {data.path}")
    print(f"cameras        {len(data.cameras)} ({', '.join(sorted(cameras))}), {first.width}x{first.height}")
    print(f"images         {len(data.images)} -> {len(train_views)} train / {len(test_views)} test")
    print(f"points         {len(data.points3D.xyz):,}")
    print(f"scene extent   {scene_extent(views):.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gstrain", description="3D Gaussian Splatting trainer")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("train", help="train a scene")
    p.add_argument("-s", "--source", type=Path, required=True, help="COLMAP project root or sparse/0 folder")
    p.add_argument("-m", "--model", type=Path, required=True, help="output directory")
    p.add_argument("--images", type=Path, help="image folder (default: <project>/images)")
    p.add_argument("--iterations", type=int, default=7_000)
    p.add_argument("--max-edge", type=int, default=480, help="long image side in pixels")
    p.add_argument("--max-views", type=int, default=0, help="use N evenly spaced views (0 = all)")
    p.add_argument("--eval-hold", type=int, default=8, help="hold out every Nth view for testing (0 = none)")
    p.add_argument("--max-points", type=int, default=0, help="starting splats from COLMAP points (0 = all)")
    p.add_argument("--sh-degree", type=int, default=3, choices=[0, 1, 2, 3])
    p.add_argument("--sh-degree-interval", type=int, default=1_000, help="iterations per extra SH band")
    p.add_argument("--background", default="black", choices=BACKGROUNDS)
    p.add_argument("--strategy", default="default", choices=STRATEGIES,
                   help="default = 3DGS clone/split/prune; mcmc = 3DGS-MCMC with a splat cap")
    p.add_argument("--max-gaussians", type=int, default=2_000_000, help="splat budget (default strategy)")
    p.add_argument("--cap-max", type=int, default=300_000, help="splat cap (mcmc strategy)")
    p.add_argument("--max-screen-fraction", type=float, default=None,
                   help="default strategy: after the first opacity reset also prune splats wider than "
                        "this fraction of the image (off by default, e.g. 0.15)")
    p.add_argument("--prune-contribution", type=float, default=0.0,
                   help="after training drop this fraction of least-contributing splats (e.g. 0.3)")
    p.add_argument("--prune-finetune", type=int, default=500, help="fine-tune iterations after pruning")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--preview-every", type=int, default=500, help="0 disables previews")
    p.add_argument("--save-iterations", type=int, nargs="*", default=[7_000, 30_000])
    p.add_argument("--checkpoint-every", type=int, default=0)
    p.add_argument("--resume", type=Path, help="checkpoint (.pt) to continue from")
    p.add_argument("--up-axis", default="colmap", choices=UP_AXES)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=command_train)

    p = commands.add_parser("verify", help="decode a .ply the way the Maya plugin does")
    p.add_argument("ply", type=Path)
    p.set_defaults(func=command_verify)

    p = commands.add_parser("stats", help="what does a COLMAP project contain?")
    p.add_argument("-s", "--source", type=Path, required=True)
    p.add_argument("--images", type=Path)
    p.add_argument("--eval-hold", type=int, default=8)
    p.set_defaults(func=command_stats)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
