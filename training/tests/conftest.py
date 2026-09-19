import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fixtures.make_synthetic_colmap import write_images, write_model  # noqa: E402


@pytest.fixture
def colmap_binary(tmp_path) -> Path:
    return write_model(tmp_path / "bin", binary=True)


@pytest.fixture
def colmap_text(tmp_path) -> Path:
    return write_model(tmp_path / "txt", binary=False)


@pytest.fixture
def colmap_with_images(tmp_path) -> Path:
    pytest.importorskip("PIL")
    root = write_model(tmp_path / "full", binary=True)
    write_images(root)
    return root
