"""Convert a selected Gaussian Splat node into a polygon mesh."""

import maya.cmds as cmds
from maya import OpenMayaUI as omui

try:
    from PySide6 import QtCore, QtWidgets
    from shiboken6 import wrapInstance, isValid
except ImportError:
    from PySide2 import QtCore, QtWidgets
    from shiboken2 import wrapInstance, isValid

NODE_TYPE = "GaussianSplattingLocator"
OBJECT_NAME = "gsSplatToMeshWindow"

_window_instance = None


def find_selected_splat_node():
    """Return the first selected splat node, or None."""
    for node in cmds.ls(selection=True, long=True) or []:
        if cmds.objectType(node, isType=NODE_TYPE):
            return node
        # A transform was selected: look for the splat shape underneath.
        shapes = cmds.listRelatives(node, shapes=True, fullPath=True) or []
        for shape in shapes:
            if cmds.objectType(shape, isType=NODE_TYPE):
                return shape
    return None


def convert(resolution=128, iso_level=0.2):
    """Convert the selected splat node. Returns the created mesh name or None."""
    node = find_selected_splat_node()
    if node is None:
        cmds.warning("Select a Gaussian Splat node first.")
        return None

    # The command does not return a result, so detect the new mesh by diffing.
    before = set(cmds.ls(type="mesh", long=True) or [])

    try:
        cmds.gsSplatToMesh(node=node, resolution=resolution, isoLevel=iso_level)
    except RuntimeError as exc:
        cmds.warning("Conversion failed: {0}".format(exc))
        return None

    created = sorted(set(cmds.ls(type="mesh", long=True) or []) - before)
    if not created:
        cmds.warning("No mesh was created. Try a lower iso level.")
        return None

    mesh = cmds.listRelatives(created[0], parent=True, fullPath=True)[0]
    cmds.select(mesh, replace=True)
    cmds.inViewMessage(
        assistMessage="Splat converted to mesh.", position="midCenter", fade=True)
    return mesh


def maya_main_window():
    """Return Maya's main window as a QWidget, so dialogs can parent to it."""
    main_window_ptr = omui.MQtUtil.mainWindow()
    return wrapInstance(int(main_window_ptr), QtWidgets.QWidget)


class SplatToMeshWindow(QtWidgets.QDialog):
    """Custom Qt dialog replacing the old cmds-based window."""

    def __init__(self, parent=None):
        super(SplatToMeshWindow, self).__init__(parent or maya_main_window())

        self.setObjectName(OBJECT_NAME)
        self.setWindowTitle("Splat To Mesh")
        # Explicit flags so the window always gets a proper title bar with a
        # close (X) button, regardless of what Maya's own window inherits.
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowCloseButtonHint)
        # Let Qt actually destroy the C++ object on close, instead of just
        # hiding it -- this is what lets show() re-create it cleanly below.
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setFixedWidth(380)

        self._script_job_id = None

        self._create_widgets()
        self._create_layout()
        self._create_connections()

        self._refresh_selected_label()
        self._start_script_job()

    # -- setup ---------------------------------------------------------

    def _create_widgets(self):
        self.selected_label = QtWidgets.QLabel("<nothing selected>")

        self.resolution_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.resolution_slider.setRange(32, 512)
        self.resolution_spin = QtWidgets.QSpinBox()
        self.resolution_spin.setRange(8, 1024)
        self.resolution_spin.setValue(128)
        self.resolution_spin.setToolTip(
            "Voxel grid resolution. Higher means more detail and slower conversion.")
        self.resolution_slider.setValue(self.resolution_spin.value())

        self.iso_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.iso_slider.setRange(1, 100)  # scaled: slider units map to 0.01-1.0
        self.iso_spin = QtWidgets.QDoubleSpinBox()
        self.iso_spin.setDecimals(3)
        self.iso_spin.setRange(0.01, 1.0)
        self.iso_spin.setSingleStep(0.01)
        self.iso_spin.setValue(0.2)
        self.iso_spin.setToolTip(
            "Density threshold of the surface. Lower means a thicker mesh.")
        self.iso_slider.setValue(int(round(self.iso_spin.value() * 100)))

        self.convert_button = QtWidgets.QPushButton("Convert To Mesh")
        self.convert_button.setFixedHeight(32)

    def _create_layout(self):
        node_layout = QtWidgets.QHBoxLayout()
        node_layout.addWidget(QtWidgets.QLabel("Splat node:"))
        node_layout.addWidget(self.selected_label, 1)

        resolution_layout = QtWidgets.QHBoxLayout()
        resolution_layout.addWidget(QtWidgets.QLabel("Resolution:"))
        resolution_layout.addWidget(self.resolution_slider, 1)
        resolution_layout.addWidget(self.resolution_spin)

        iso_layout = QtWidgets.QHBoxLayout()
        iso_layout.addWidget(QtWidgets.QLabel("Iso Level:"))
        iso_layout.addWidget(self.iso_slider, 1)
        iso_layout.addWidget(self.iso_spin)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)
        main_layout.addLayout(node_layout)
        main_layout.addLayout(resolution_layout)
        main_layout.addLayout(iso_layout)
        main_layout.addSpacing(4)
        main_layout.addWidget(self.convert_button)

    def _create_connections(self):
        self.resolution_slider.valueChanged.connect(self._on_resolution_slider_changed)
        self.resolution_spin.valueChanged.connect(self._on_resolution_spin_changed)
        self.iso_slider.valueChanged.connect(self._on_iso_slider_changed)
        self.iso_spin.valueChanged.connect(self._on_iso_spin_changed)
        self.convert_button.clicked.connect(self._on_convert)

    # -- selection tracking ----------------------------------------------

    def _refresh_selected_label(self, *_args):
        node = find_selected_splat_node()
        self.selected_label.setText(node.split("|")[-1] if node else "<nothing selected>")

    def _start_script_job(self):
        # Parented to this dialog so Maya kills it automatically if the
        # window is destroyed some other way; also explicitly killed below.
        self._script_job_id = cmds.scriptJob(
            event=["SelectionChanged", self._refresh_selected_label], protected=True)

    def closeEvent(self, event):
        if self._script_job_id is not None and cmds.scriptJob(exists=self._script_job_id):
            cmds.scriptJob(kill=self._script_job_id, force=True)
            self._script_job_id = None
        super(SplatToMeshWindow, self).closeEvent(event)

    # -- slider/spinbox sync ----------------------------------------------

    def _on_resolution_slider_changed(self, value):
        self.resolution_spin.blockSignals(True)
        self.resolution_spin.setValue(value)
        self.resolution_spin.blockSignals(False)

    def _on_resolution_spin_changed(self, value):
        clamped = max(self.resolution_slider.minimum(),
                      min(self.resolution_slider.maximum(), value))
        self.resolution_slider.blockSignals(True)
        self.resolution_slider.setValue(clamped)
        self.resolution_slider.blockSignals(False)

    def _on_iso_slider_changed(self, value):
        self.iso_spin.blockSignals(True)
        self.iso_spin.setValue(value / 100.0)
        self.iso_spin.blockSignals(False)

    def _on_iso_spin_changed(self, value):
        clamped = max(self.iso_slider.minimum(),
                      min(self.iso_slider.maximum(), int(round(value * 100))))
        self.iso_slider.blockSignals(True)
        self.iso_slider.setValue(clamped)
        self.iso_slider.blockSignals(False)

    # -- actions -----------------------------------------------------------

    def _on_convert(self):
        convert(self.resolution_spin.value(), self.iso_spin.value())


def show():
    global _window_instance

    if _window_instance is not None and isValid(_window_instance):
        _window_instance.close()

    _window_instance = SplatToMeshWindow()
    _window_instance.show()
    return _window_instance


show()