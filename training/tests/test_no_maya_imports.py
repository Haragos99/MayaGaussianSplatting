"""The training package must stay usable outside Maya."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

FORBIDDEN = ("maya", "pymel", "PySide", "PySide2", "PySide6", "shiboken", "shiboken2", "shiboken6")

PACKAGE = Path(__file__).resolve().parents[1] / "gstrain"


def _imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_maya_imports_in_source():
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        roots = _imported_roots(ast.parse(path.read_text(encoding="utf-8"), str(path)))
        hits = roots.intersection(FORBIDDEN)
        if hits:
            offenders.append(f"{path.relative_to(PACKAGE.parent)}: {sorted(hits)}")
    assert not offenders, "Maya/Qt imports leaked into the trainer:\n" + "\n".join(offenders)


def test_importing_the_package_loads_no_maya_modules():
    code = (
        "import gstrain, gstrain.cli, gstrain.colmap, gstrain.cameras, "
        "gstrain.params, gstrain.export_ply, gstrain.verify_ply, gstrain.sh, sys;"
        f"bad=[m for m in sys.modules if m.split('.')[0] in {FORBIDDEN!r}];"
        "print(bad); raise SystemExit(1 if bad else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PACKAGE.parent),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
