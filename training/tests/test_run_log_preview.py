import json
import numpy as np
import torch
from PIL import Image
from gstrain.preview import GAP, LABEL_HEIGHT, compose, error_map, save_comparison, save_image
from gstrain.run_log import RunLog


def test_run_log_lines_and_status(tmp_path) -> None:
    with RunLog(tmp_path / "run") as log:
        log.log({"iteration": 1, "loss": 0.5})
        log.log({"iteration": 2, "loss": 0.4})
        log.status("running", iteration=2)
        log.status("error", message="boom")
        log.write_json("metrics.json", {"psnr": 25.0})
    lines = (tmp_path / "run" / "progress.jsonl").read_text().splitlines()
    assert [json.loads(line)["iteration"] for line in lines] == [1, 2]
    status = json.loads((tmp_path / "run" / "status.json").read_text())
    assert status["state"] == "error" and status["message"] == "boom"
    assert not (tmp_path / "run" / "status.json.tmp").exists()
    assert json.loads((tmp_path / "run" / "metrics.json").read_text()) == {"psnr": 25.0}
    assert log._progress.closed

    # A second run log in the same folder appends (resume keeps the history).
    with RunLog(tmp_path / "run") as log:
        log.log({"iteration": 3})
    assert len((tmp_path / "run" / "progress.jsonl").read_text().splitlines()) == 3


def test_error_map_and_compose_sizes() -> None:
    a = torch.rand(10, 12, 3)
    assert torch.equal(error_map(a, a), torch.zeros_like(a))
    assert error_map(a, (a + 0.5).clamp(0, 1)).max() <= 1.0

    picture = compose([("a", a), ("b", torch.rand(8, 20, 3)), ("c", a)])
    assert picture.size == (12 + 20 + 12 + 2 * GAP, 10 + LABEL_HEIGHT)
    assert compose([("a", a)], labels=False).size == (12, 10)


def test_save_image_and_comparison_roundtrip(tmp_path) -> None:
    image = torch.linspace(0, 1, 16 * 16 * 3).reshape(16, 16, 3)
    path = save_image(image, tmp_path / "deep" / "x.png")
    pixels = np.asarray(Image.open(path))
    assert pixels.shape == (16, 16, 3) and pixels.dtype == np.uint8
    assert np.abs(pixels / 255.0 - image.numpy()).max() <= 0.5 / 255 + 1e-6  # PNG is lossless

    jpg = save_comparison(image, image, tmp_path / "cmp.jpg")
    assert Image.open(jpg).size == (3 * 16 + 2 * GAP, 16 + LABEL_HEIGHT)
