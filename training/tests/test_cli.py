import numpy as np

from gstrain import verify_ply
from gstrain.cli import main


def test_stats_runs(colmap_binary, capsys):
    assert main(["stats", "-s", str(colmap_binary)]) == 0
    out = capsys.readouterr().out
    assert "images        : 6" in out
    assert "points3D      : 40" in out
    assert "scene extent" in out


def test_export_init_writes_a_loadable_ply(colmap_binary, tmp_path, capsys):
    out_dir = tmp_path / "run"
    assert main(["export-init", "-s", str(colmap_binary), "-m", str(out_dir)]) == 0
    ply = out_dir / "point_cloud" / "iteration_0" / "point_cloud.ply"
    assert ply.exists()
    decoded = verify_ply.decode(ply)
    assert len(decoded) == 40
    assert np.isfinite(decoded.xyz).all()
    assert "wrote 40 splats" in capsys.readouterr().out


def test_verify_command(colmap_binary, tmp_path, capsys):
    out_dir = tmp_path / "run"
    main(["export-init", "-s", str(colmap_binary), "-m", str(out_dir)])
    ply = out_dir / "point_cloud" / "iteration_0" / "point_cloud.ply"
    assert main(["verify", str(ply)]) == 0
    assert "splats        : 40" in capsys.readouterr().out
