import json
import numpy as np
import pytest
from gstrain import cli
from gstrain.export_ply import load_ply, read_ply

from conftest import SCENE_HEIGHT, SCENE_IMAGES, SCENE_WIDTH

pytestmark = pytest.mark.slow

FAST = ["--max-edge", "32", "--preview-every", "0", "--log-every", "1", "--eval-every", "10",
        "--save-iterations"]


def _train(scene, out, iterations: int = 12, *extra: str) -> None:
    cli.main(["train", "-s", str(scene), "-m", str(out), "--iterations", str(iterations), *FAST, *extra])


def _records(out) -> list[dict]:
    return [json.loads(line) for line in (out / "progress.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def trained_run(tiny_scene, tmp_path):
    out = tmp_path / "run"
    _train(tiny_scene, out)
    return out


def test_cli_train_writes_run_files(trained_run, tiny_scene) -> None:
    out = trained_run
    for name in ("point_cloud.ply", "cfg_args.json", "progress.jsonl", "status.json", "metrics.json",
                 "point_cloud/iteration_12/point_cloud.ply"):
        assert (out / name).is_file(), name

    status = json.loads((out / "status.json").read_text())
    assert status["state"] == "done" and status["iteration"] == 12

    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["iterations"] == 12
    assert metrics["train_views"] + metrics["test_views"] == SCENE_IMAGES
    assert metrics["test_views"] == 1  # every 8th view
    assert metrics["resolution"] == [SCENE_WIDTH, SCENE_HEIGHT]
    assert np.isfinite(metrics["test_psnr"])

    iterations = [r["iteration"] for r in _records(out) if "loss" in r]
    assert iterations == list(range(1, 13))

    cfg = json.loads((out / "cfg_args.json").read_text())
    assert cfg["source"] == str(tiny_scene) and cfg["up_axis"] == "colmap"


def test_cli_train_mcmc_runs(tiny_scene, tmp_path) -> None:
    out = tmp_path / "mcmc"
    _train(tiny_scene, out, 6, "--strategy", "mcmc", "--cap-max", "100")
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["strategy"] == "mcmc"
    assert metrics["splats"] == 60  # no refine step before iteration 500
    # MCMC starts splats at opacity 0.5 instead of 0.1.
    opacity = 1 / (1 + np.exp(-read_ply(out / "point_cloud.ply")["opacity"]))
    assert np.median(opacity) > 0.3


def test_cli_clean_reads_cfg_args(trained_run) -> None:
    cli.main(["clean", str(trained_run), "--min-opacity", "0.05"])
    output = trained_run / "point_cloud_clean.ply"
    assert output.is_file()
    assert (trained_run / "point_cloud_clean_preview.jpg").is_file()
    report = json.loads((trained_run / "point_cloud_clean_report.json").read_text())
    assert report["splats_before"] - sum(report["removed"].values()) == report["splats_after"]
    assert len(read_ply(output)["x"]) == report["splats_after"]
    assert "test_psnr_before" in report


def test_cli_clean_without_cfg_args_needs_source(trained_run, tmp_path) -> None:
    lone = tmp_path / "lone.ply"
    lone.write_bytes((trained_run / "point_cloud.ply").read_bytes())
    with pytest.raises(ValueError, match="pass -s"):
        cli.main(["clean", str(lone), "--no-preview"])


def test_cli_clean_up_axis_maya(trained_run, tmp_path) -> None:
    colmap = tmp_path / "colmap.ply"
    maya = tmp_path / "maya.ply"
    cli.main(["clean", str(trained_run), "--no-preview", "-o", str(colmap)])
    cli.main(["clean", str(trained_run), "--no-preview", "-o", str(maya), "--up-axis", "maya-y"])
    a = load_ply(colmap)
    b = load_ply(maya, up_axis="maya-y")
    assert len(a) == len(b)
    assert np.allclose(a.xyz.detach().numpy(), b.xyz.detach().numpy(), atol=1e-5)
    raw = read_ply(maya)
    assert np.allclose(raw["y"], -read_ply(colmap)["y"], atol=1e-5)


def test_cli_verify_and_stats(trained_run, tiny_scene, capsys) -> None:
    ply = trained_run / "point_cloud.ply"
    cli.main(["verify", str(ply)])
    assert "OK" in capsys.readouterr().out

    broken = trained_run / "broken.ply"
    data = ply.read_bytes()
    broken.write_bytes(data[: len(data) // 2])
    with pytest.raises(Exception):
        cli.main(["verify", str(broken)])

    cli.main(["stats", "-s", str(tiny_scene)])
    out = capsys.readouterr().out
    assert f"images         {SCENE_IMAGES} -> 7 train / 1 test" in out
    assert "PINHOLE" in out and "SIMPLE_RADIAL" in out
    assert "points         60" in out


def test_cli_resume_continues_iterations(tiny_scene, tmp_path) -> None:
    first = tmp_path / "first"
    _train(tiny_scene, first, 10, "--checkpoint-every", "6")
    checkpoint = first / "chkpnt6.pt"
    assert checkpoint.is_file()

    second = tmp_path / "second"
    _train(tiny_scene, second, 10, "--resume", str(checkpoint))
    iterations = [r["iteration"] for r in _records(second) if "loss" in r]
    assert iterations == [7, 8, 9, 10]
    assert json.loads((second / "metrics.json").read_text())["iterations"] == 10


def test_paths_with_spaces_and_dollar(tiny_scene, tmp_path) -> None:
    # The PowerShell quoting bug once wrote to `runs$name`; the CLI itself must use the path as given.
    out = tmp_path / "my runs" / "$name test"
    _train(tiny_scene, out, 3)
    assert (out / "point_cloud.ply").is_file()
    assert not (tmp_path / "my runs$name test").exists()


def test_missing_images_folder_is_reported(tiny_scene, tmp_path) -> None:
    out = tmp_path / "bad"
    with pytest.raises(FileNotFoundError, match="image directory"):
        _train(tiny_scene, out, 3, "--images", str(tmp_path / "nope"))
    assert json.loads((out / "status.json").read_text())["state"] == "error"
