"""Train a Gaussian Splatting model from Maya.

The trainer itself has no Maya dependency; this panel only launches it as an
external process and reads the files it writes (progress.jsonl, status.json,
previews/latest.png, point_cloud.ply). See training/README.md.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import maya.cmds as cmds
from maya import OpenMayaUI as omui

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from shiboken6 import wrapInstance
except ImportError:
    from PySide2 import QtCore, QtGui, QtWidgets
    from shiboken2 import wrapInstance

NODE_TYPE = "GaussianSplattingLocator"
OBJECT_NAME = "gsTrainWindow"
POLL_MS = 500
STOP_GRACE_MS = 30000

_window_instance = None


def repo_root():
    return Path(r"C:\Users\Geri\Documents\Projects\CG\MayaGaussianSplatting")


def default_trainer_dir():
    return repo_root() / "training"


def default_python():
    candidate = default_trainer_dir() / ".venv" / "Scripts" / "python.exe"
    return candidate if candidate.exists() else Path(sys.executable)


def child_environment():
    """Maya exports Python and Qt variables that would hijack the child interpreter.

    Left in place, the external python picks up Maya's stdlib and Qt plugins and
    either fails to start or crashes on import.
    """
    blocked = (
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONEXECUTABLE",
        "PYTHONNOUSERSITE",
        "QT_PLUGIN_PATH",
        "QT_QPA_PLATFORM_PLUGIN_PATH",
        "QML2_IMPORT_PATH",
        "QT_AUTO_SCREEN_SCALE_FACTOR",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
    )
    return {key: value for key, value in os.environ.items() if key not in blocked}


def maya_main_window():
    return wrapInstance(int(omui.MQtUtil.mainWindow()), QtWidgets.QWidget)


def find_selected_splat_node():
    """Return the first selected splat node, or None."""
    for node in cmds.ls(selection=True, long=True) or []:
        if cmds.objectType(node, isType=NODE_TYPE):
            return node
        for shape in cmds.listRelatives(node, shapes=True, fullPath=True) or []:
            if cmds.objectType(shape, isType=NODE_TYPE):
                return shape
    return None


class PathRow(QtWidgets.QWidget):
    """Line edit plus a browse button, for a file or a directory."""

    def __init__(self, caption, pick_directory=True, file_filter="", parent=None):
        super(PathRow, self).__init__(parent)
        self.caption = caption
        self.pick_directory = pick_directory
        self.file_filter = file_filter

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = QtWidgets.QLineEdit()
        button = QtWidgets.QPushButton("...")
        button.setFixedWidth(30)
        button.clicked.connect(self._browse)
        layout.addWidget(self.edit)
        layout.addWidget(button)

    def _browse(self):
        start = self.edit.text() or str(repo_root())
        if self.pick_directory:
            chosen = QtWidgets.QFileDialog.getExistingDirectory(self, self.caption, start)
        else:
            chosen, _ = QtWidgets.QFileDialog.getOpenFileName(self, self.caption, start, self.file_filter)
        if chosen:
            self.edit.setText(chosen)

    def text(self):
        return self.edit.text().strip()

    def setText(self, value):
        self.edit.setText(value or "")


class TrainWindow(QtWidgets.QDialog):

    def __init__(self, parent=None):
        super(TrainWindow, self).__init__(parent or maya_main_window())

        self.setObjectName(OBJECT_NAME)
        self.setWindowTitle("Gaussian Splatting Training")
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowCloseButtonHint)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setMinimumWidth(520)

        self._process = None
        self._log_handle = None
        self._preview_stamp = None
        self._stop_elapsed = 0

        self._build()
        self._restore_settings()

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(POLL_MS)
        self._timer.timeout.connect(self._poll)

        self._set_running(False)

    # ---------------------------------------------------------------- ui --

    def _build(self):
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self.source = PathRow("COLMAP project folder")
        self.output = PathRow("Training output folder")
        self.python = PathRow("python.exe", pick_directory=False, file_filter="python.exe (python.exe)")
        form.addRow("COLMAP scene", self.source)
        form.addRow("Output", self.output)
        form.addRow("Python", self.python)

        self.iterations = QtWidgets.QSpinBox()
        self.iterations.setRange(10, 200000)
        self.iterations.setSingleStep(1000)
        self.iterations.setValue(7000)
        form.addRow("Iterations", self.iterations)

        self.resolution = QtWidgets.QComboBox()
        for factor in (1, 2, 4, 8, 16, 24):
            self.resolution.addItem("1/{0}".format(factor), factor)
        self.resolution.setCurrentIndex(3)
        form.addRow("Resolution", self.resolution)

        self.device = QtWidgets.QComboBox()
        self.device.addItems(["auto", "cuda", "cpu"])
        form.addRow("Device", self.device)

        self.up_axis = QtWidgets.QComboBox()
        self.up_axis.addItems(["colmap", "maya-y"])
        form.addRow("Up axis", self.up_axis)
        layout.addLayout(form)

        buttons = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.clicked.connect(self.start)
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop)
        self.load_button = QtWidgets.QPushButton("Load latest .ply")
        self.load_button.clicked.connect(self.load_latest)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.load_button)
        layout.addLayout(buttons)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(True)
        layout.addWidget(self.progress)

        self.status = QtWidgets.QLabel("idle")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.metrics = QtWidgets.QLabel("-")
        self.metrics.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.metrics)

        self.preview = QtWidgets.QLabel("no preview yet")
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setMinimumHeight(240)
        self.preview.setStyleSheet("background: #2b2b2b; border: 1px solid #1a1a1a;")
        layout.addWidget(self.preview, 1)

    def _set_running(self, running):
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        for widget in (self.source, self.output, self.python, self.iterations,
                       self.resolution, self.device, self.up_axis):
            widget.setEnabled(not running)

    # ---------------------------------------------------------- settings --

    def _restore_settings(self):
        self.source.setText(self._option("gsTrainSource", ""))
        self.output.setText(self._option("gsTrainOutput", ""))
        self.python.setText(self._option("gsTrainPython", str(default_python())))

    def _store_settings(self):
        cmds.optionVar(stringValue=("gsTrainSource", self.source.text()))
        cmds.optionVar(stringValue=("gsTrainOutput", self.output.text()))
        cmds.optionVar(stringValue=("gsTrainPython", self.python.text()))

    @staticmethod
    def _option(name, fallback):
        if cmds.optionVar(exists=name):
            return cmds.optionVar(query=name)
        return fallback

    # ----------------------------------------------------------- process --

    def start(self):
        source = Path(self.source.text())
        output = Path(self.output.text())
        python = Path(self.python.text())
        trainer = default_trainer_dir()

        if not source.is_dir():
            return self._warn("COLMAP scene folder does not exist:\n{0}".format(source))
        if not python.is_file():
            return self._warn("Python interpreter not found:\n{0}".format(python))
        if not (trainer / "gstrain").is_dir():
            return self._warn("Trainer package not found:\n{0}".format(trainer))
        if not output.name:
            return self._warn("Choose an output folder.")

        output.mkdir(parents=True, exist_ok=True)
        self._store_settings()

        command = [
            str(python), "-m", "gstrain.cli", "train",
            "-s", str(source),
            "-m", str(output),
            "-r", str(self.resolution.currentData()),
            "--iterations", str(self.iterations.value()),
            "--device", self.device.currentText(),
            "--up-axis", self.up_axis.currentText(),
            "--quiet",
        ]

        flags = 0
        if sys.platform == "win32":
            flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP

        self._log_handle = open(str(output / "train_log.txt"), "w", encoding="utf-8")
        try:
            self._process = subprocess.Popen(
                command,
                cwd=str(trainer),
                env=child_environment(),
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                creationflags=flags,
            )
        except OSError as error:
            self._close_log()
            return self._warn("Could not start the trainer:\n{0}".format(error))

        self._progress_offset = 0
        self._stop_elapsed = 0
        self.progress.setMaximum(self.iterations.value())
        self.progress.setValue(0)
        self.status.setText("starting (loading images can take minutes)")
        self._set_running(True)
        self._timer.start()

    def stop(self):
        if self._process is None:
            return
        # Graceful: the trainer finishes the iteration and still exports.
        try:
            (Path(self.output.text()) / "STOP").write_text("", encoding="utf-8")
        except OSError as error:
            cmds.warning("Could not write STOP file: {0}".format(error))
        self.status.setText("stopping after the current iteration...")
        self.stop_button.setEnabled(False)
        self._stop_elapsed = 1

    def _force_stop(self):
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()

    # -------------------------------------------------------------- poll --

    def _poll(self):
        output = Path(self.output.text())
        self._read_progress(output)
        self._read_status(output)
        self._refresh_preview(output)

        if self._stop_elapsed:
            self._stop_elapsed += POLL_MS
            if self._stop_elapsed > STOP_GRACE_MS:
                self.status.setText("trainer did not stop in time; terminating")
                self._force_stop()
                self._stop_elapsed = 0

        if self._process is not None and self._process.poll() is not None:
            self._finish(output)

    def _read_progress(self, output):
        path = output / "progress.jsonl"
        if not path.is_file():
            return
        try:
            lines = path.read_text(encoding="utf-8").strip().splitlines()
        except OSError:
            return
        for line in reversed(lines):
            try:
                record = json.loads(line)
            except ValueError:
                continue  # a partially flushed line
            self.progress.setValue(int(record.get("iter", 0)))
            self.metrics.setText(
                "iter {iter}   loss {loss:.4f}   psnr {psnr:.2f}   splats {num_gaussians}"
                "   {elapsed_s:.0f}s".format(**record)
            )
            return

    def _read_status(self, output):
        path = output / "status.json"
        if not path.is_file():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        state = payload.get("state", "?")
        if state == "error":
            self.status.setText("error: {0}  (see train_log.txt)".format(payload.get("message", "")))
        elif state == "running" and not self._stop_elapsed:
            self.status.setText("training")

    def _refresh_preview(self, output):
        path = output / "previews" / "latest.png"
        if not path.is_file():
            return
        try:
            stamp = path.stat().st_mtime
        except OSError:
            return
        if stamp == self._preview_stamp:
            return
        pixmap = QtGui.QPixmap(str(path))
        if pixmap.isNull():
            return  # caught it mid-write; the next tick will get it
        self._preview_stamp = stamp
        self.preview.setPixmap(
            pixmap.scaled(
                self.preview.width(),
                self.preview.height(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )

    def _finish(self, output):
        code = self._process.returncode
        self._process = None
        self._close_log()
        self._timer.stop()
        self._set_running(False)
        self._stop_elapsed = 0

        state = "?"
        status_path = output / "status.json"
        if status_path.is_file():
            try:
                state = json.loads(status_path.read_text(encoding="utf-8")).get("state", "?")
            except (OSError, ValueError):
                pass

        if state in ("done", "stopped"):
            self.status.setText("{0}; press 'Load latest .ply'".format(state))
        else:
            self.status.setText("exited with code {0} ({1}) - see train_log.txt".format(code, state))

    def _close_log(self):
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    # --------------------------------------------------------------- maya --

    def load_latest(self):
        ply = Path(self.output.text()) / "point_cloud.ply"
        if not ply.is_file():
            return self._warn("No export yet:\n{0}".format(ply))

        node = find_selected_splat_node()
        if node is None:
            node = cmds.createNode(NODE_TYPE)
        # Iteration 2's reload path clears the override buffers for us.
        cmds.setAttr(node + ".fileName", str(ply), type="string")
        cmds.select(node, replace=True)
        cmds.inViewMessage(assistMessage="Loaded {0}".format(ply.name), position="midCenter", fade=True)

    def _warn(self, message):
        cmds.warning(message.replace("\n", " "))
        QtWidgets.QMessageBox.warning(self, "Gaussian Splatting Training", message)

    def closeEvent(self, event):
        self._timer.stop()
        self._close_log()
        if self._process is not None and self._process.poll() is None:
            cmds.warning(
                "Training is still running in the background. Reopen the window "
                "or use 'gstrain.cli stop -m <output>' to end it."
            )
        super(TrainWindow, self).closeEvent(event)


def show():
    global _window_instance
    if _window_instance is not None:
        try:
            _window_instance.close()
            _window_instance.deleteLater()
        except RuntimeError:
            pass
    _window_instance = TrainWindow()
    _window_instance.show()
    return _window_instance


if __name__ == "__main__":
    show()
