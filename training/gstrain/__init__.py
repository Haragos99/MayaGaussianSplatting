"""Gaussian Splatting trainer.

Hard rule: nothing under `gstrain/` may import maya, pymel, PySide or shiboken.
The Maya UI lives in src/scripts/ and talks to this package over the CLI and
the files it writes. See tests/test_no_maya_imports.py.
"""

__version__ = "0.1.0"
