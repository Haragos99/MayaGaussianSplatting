import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from gstrain import verify_ply  # noqa: E402
from gstrain.dataset import Scene  # noqa: E402
from gstrain.train import TrainConfig, train  # noqa: E402


def small_config(source, out, **overrides) -> TrainConfig:
    defaults = dict(
        source=source,
        model=out,
        iterations=40,
        sh_degree=1,
        device="cpu",
        eval_hold=0,
        log_every=1,
        preview_every=0,
        save_iterations=(40,),
        densify_from_iter=10,
        densify_until_iter=30,
        densification_interval=10,
        opacity_reset_interval=1_000_000,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def read_progress(out):
    lines = (out / "progress.jsonl").read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines]


def test_scene_loads_images(colmap_with_images):
    scene = Scene.load(colmap_with_images, eval_hold=0, sh_degree=1)
    assert len(scene.train_cameras) == 6
    assert scene.extent > 0
    camera = scene.train_cameras[0]
    assert camera.image.shape == (3, 48, 64)
    assert 0.0 <= float(camera.image.min()) and float(camera.image.max()) <= 1.0


def test_scene_reports_missing_images(colmap_binary):
    with pytest.raises(FileNotFoundError, match="images are missing"):
        Scene.load(colmap_binary, eval_hold=0)


def test_training_reduces_loss_and_exports_a_loadable_ply(colmap_with_images, tmp_path):
    out = tmp_path / "run"
    final = train(small_config(colmap_with_images, out))

    records = read_progress(out)
    assert len(records) == 40
    assert all(r["loss"] == r["loss"] for r in records), "loss went NaN"
    first = sum(r["loss"] for r in records[:5]) / 5
    last = sum(r["loss"] for r in records[-5:]) / 5
    assert last < first

    assert final.exists()
    decoded = verify_ply.decode(final)
    assert len(decoded) == records[-1]["num_gaussians"]
    assert decoded.opacity.min() >= 0.0

    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "done"


def test_densification_changes_the_gaussian_count(colmap_with_images, tmp_path):
    out = tmp_path / "run"
    train(small_config(colmap_with_images, out, densify_grad_threshold=0.0))
    counts = [r["num_gaussians"] for r in read_progress(out)]
    assert counts[-1] != counts[0]


def test_previews_are_written(colmap_with_images, tmp_path):
    out = tmp_path / "run"
    train(small_config(colmap_with_images, out, iterations=20, save_iterations=(20,), preview_every=10))
    assert (out / "previews" / "iter_000010" / "contact_sheet.png").exists()
    assert (out / "previews" / "latest.png").exists()
    assert (out / "previews" / "timeline.png").exists()
    metrics = json.loads((out / "previews" / "iter_000020" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["psnr"] > 0


def test_failure_is_recorded_in_status(tmp_path):
    out = tmp_path / "run"
    with pytest.raises(FileNotFoundError):
        train(small_config(tmp_path / "does-not-exist", out))
    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "error"
    assert "traceback" in status


def test_interrupt_is_recorded_as_cancelled(colmap_with_images, tmp_path, monkeypatch):
    """KeyboardInterrupt is a BaseException, so it must be caught separately."""
    from gstrain import train as train_module

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(train_module, "render", interrupt)
    out = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        train(small_config(colmap_with_images, out))

    status = json.loads((out / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "cancelled"


def test_quiet_suppresses_all_console_output(colmap_with_images, tmp_path, capsys):
    train(small_config(colmap_with_images, tmp_path / "run", iterations=5, save_iterations=(5,), quiet=True))
    captured = capsys.readouterr()
    assert captured.out == ""


def test_stages_are_reported(colmap_with_images, tmp_path, capsys):
    train(small_config(colmap_with_images, tmp_path / "run", iterations=5, save_iterations=(5,)))
    out = capsys.readouterr().out
    assert "[1/4] reading COLMAP model" in out
    assert "[2/4] loading images" in out
    assert "[3/4] initialising gaussians" in out
    assert "[4/4] training 5 iterations" in out
