"""Command line entry point. Phase A commands need only numpy + scipy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import cameras as cam_math
from . import colmap, verify_ply
from .export_ply import write_ply
from .params import create_from_point_cloud


def _resolve_images_dir(source: Path, override: Path | None) -> Path:
    if override is not None:
        return override
    for candidate in (source / "images", source.parent / "images", source.parent.parent / "images"):
        if candidate.is_dir():
            return candidate
    return source / "images"


def cmd_stats(args: argparse.Namespace) -> int:
    model = colmap.read_model(args.source)
    print(f"model dir     : {model.path}")
    print(f"cameras       : {len(model.cameras)}")
    for cam in sorted(model.cameras.values(), key=lambda c: c.id):
        try:
            note = "" if cam.is_pinhole else "  <-- distortion, rectified on load"
            cam.intrinsics()
        except ValueError as error:
            note = f"  <-- UNSUPPORTED: {error}"
        print(f"  id {cam.id:<4} {cam.model:<22} {cam.width}x{cam.height} params={cam.params}{note}")
    print(f"images        : {len(model.images)}")
    print(f"points3D      : {len(model.points.ids)}")
    if len(model.points.ids):
        print(f"  bbox min    : {np.array2string(model.points.xyz.min(axis=0), precision=4)}")
        print(f"  bbox max    : {np.array2string(model.points.xyz.max(axis=0), precision=4)}")
        print(f"  mean error  : {model.points.error.mean():.4f} px")

    images_dir = _resolve_images_dir(Path(args.source), args.images)
    print(f"images dir    : {images_dir}{'' if images_dir.is_dir() else '  <-- MISSING'}")
    try:
        infos = cam_math.cameras_from_model(model, images_dir, 8)
    except ValueError as error:
        print(f"cameras       : cannot be used -> {error}")
        return 1
    center, radius = cam_math.scene_normalization(infos)
    print(f"scene centre  : {np.array2string(center, precision=4)}")
    print(f"scene extent  : {radius:.4f}")
    print(f"train / test  : {sum(not c.is_test for c in infos)} / {sum(c.is_test for c in infos)}")
    return 0


def cmd_export_init(args: argparse.Namespace) -> int:
    model = colmap.read_model(args.source)
    params = create_from_point_cloud(model.points.xyz, model.points.rgb, sh_degree=args.sh_degree)
    out = Path(args.model) / "point_cloud" / "iteration_0" / "point_cloud.ply"
    write_ply(out, params, up_axis=args.up_axis)
    print(f"wrote {len(params)} splats -> {out}")
    print(verify_ply.summarize(out))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    print(verify_ply.summarize(args.ply))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    # Imported lazily so stats/export-init/verify keep working without torch.
    from .train import TrainConfig, train

    cfg = TrainConfig(
        source=args.source,
        model=args.model,
        images=args.images,
        resolution=args.resolution,
        iterations=args.iterations,
        sh_degree=args.sh_degree,
        device=args.device,
        data_device=args.data_device,
        eval_hold=args.eval_hold,
        background=args.background,
        seed=args.seed,
        up_axis=args.up_axis,
        element_budget=args.element_budget,
        log_every=args.log_every,
        preview_every=args.preview_every,
        preview_views=args.preview_views,
        save_iterations=tuple(args.save_iterations),
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        densify_from_iter=args.densify_from_iter,
        densify_until_iter=args.densify_until_iter,
        densification_interval=args.densification_interval,
        opacity_reset_interval=args.opacity_reset_interval,
        densify_grad_threshold=args.densify_grad_threshold,
        sh_degree_interval=args.sh_degree_interval,
        quiet=args.quiet,
        max_views=args.max_views,
        max_points=args.max_points,
    )
    final = train(cfg)
    if not args.quiet:
        print(f"finished -> {final}")
        print(verify_ply.summarize(final))
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .bench import benchmark

    for device in args.devices:
        try:
            print(
                benchmark(
                    args.source,
                    resolution=args.resolution,
                    max_points=args.max_points,
                    device=device,
                    steps=args.steps,
                    sh_degree=args.sh_degree,
                    element_budget=args.element_budget,
                )
            )
        except (RuntimeError, AssertionError) as error:
            print(f"{device:<5} unavailable: {error}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gstrain", description="Gaussian Splatting trainer")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_source(p: argparse.ArgumentParser) -> None:
        p.add_argument("-s", "--source", type=Path, required=True, help="COLMAP project or sparse model dir")
        p.add_argument("--images", type=Path, default=None, help="image folder (default: <source>/images)")

    p_stats = sub.add_parser("stats", help="print what the COLMAP model contains")
    add_source(p_stats)
    p_stats.set_defaults(func=cmd_stats)

    p_init = sub.add_parser("export-init", help="export the initial Gaussians as a 3DGS .ply")
    add_source(p_init)
    p_init.add_argument("-m", "--model", type=Path, required=True, help="output dir")
    p_init.add_argument("--sh-degree", type=int, default=3, choices=[0, 1, 2, 3])
    p_init.add_argument("--up-axis", default="colmap", choices=["colmap", "maya-y"])
    p_init.set_defaults(func=cmd_export_init)

    p_verify = sub.add_parser("verify", help="decode a .ply the way the Maya plugin does")
    p_verify.add_argument("ply", type=Path)
    p_verify.set_defaults(func=cmd_verify)

    p_train = sub.add_parser("train", help="train a Gaussian Splatting model")
    add_source(p_train)
    p_train.add_argument("-m", "--model", type=Path, required=True, help="output dir")
    p_train.add_argument("-r", "--resolution", type=int, default=1, help="image downscale factor")
    p_train.add_argument("--iterations", type=int, default=30_000)
    p_train.add_argument("--sh-degree", type=int, default=3, choices=[0, 1, 2, 3])
    p_train.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p_train.add_argument("--data-device", default="cpu", help="where to cache the training images")
    p_train.add_argument("--eval-hold", type=int, default=8, help="hold out every Nth view (0 = none)")
    p_train.add_argument("--background", default="black", choices=["black", "white"])
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--up-axis", default="colmap", choices=["colmap", "maya-y"])
    p_train.add_argument("--element-budget", type=int, default=4_000_000, help="rasterizer tile chunk size")
    p_train.add_argument("--log-every", type=int, default=10)
    p_train.add_argument("--preview-every", type=int, default=500, help="0 disables previews")
    p_train.add_argument("--preview-views", type=int, default=4)
    p_train.add_argument("--save-iterations", type=int, nargs="*", default=[7_000, 30_000])
    p_train.add_argument("--checkpoint-every", type=int, default=0)
    p_train.add_argument("--resume", type=Path, default=None)
    p_train.add_argument("--densify-from-iter", type=int, default=500)
    p_train.add_argument("--densify-until-iter", type=int, default=15_000)
    p_train.add_argument("--densification-interval", type=int, default=100)
    p_train.add_argument("--opacity-reset-interval", type=int, default=3_000)
    p_train.add_argument("--densify-grad-threshold", type=float, default=2e-4)
    p_train.add_argument("--sh-degree-interval", type=int, default=1_000, help="iters per SH band")
    p_train.add_argument("--max-views", type=int, default=0, help="use at most N views (0 = all)")
    p_train.add_argument("--max-points", type=int, default=0, help="seed with at most N points (0 = all)")
    p_train.add_argument("--quiet", action="store_true", help="no banner or progress bar")
    p_train.set_defaults(func=cmd_train)

    p_bench = sub.add_parser("bench", help="time the rasterizer (no images needed)")
    add_source(p_bench)
    p_bench.add_argument("-r", "--resolution", type=int, default=8)
    p_bench.add_argument("--max-points", type=int, default=0)
    p_bench.add_argument("--sh-degree", type=int, default=3, choices=[0, 1, 2, 3])
    p_bench.add_argument("--steps", type=int, default=5)
    p_bench.add_argument("--element-budget", type=int, default=4_000_000)
    p_bench.add_argument("--devices", nargs="*", default=["cpu", "cuda"])
    p_bench.set_defaults(func=cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
