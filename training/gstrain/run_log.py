import json
import os
import time
from pathlib import Path


class RunLog:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._progress = open(self.directory / "progress.jsonl", "a", encoding="utf-8")

    def log(self, record: dict) -> None:
        self._progress.write(json.dumps(record) + "\n")
        self._progress.flush()
        os.fsync(self._progress.fileno())

    def status(self, state: str, **fields) -> None:
        payload = {"state": state, "time": time.time(), **fields}
        path = self.directory / "status.json"
        # Write-then-rename so a reader never sees a half-written file.
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, path)

    def write_json(self, name: str, payload: dict) -> Path:
        path = self.directory / name
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def close(self) -> None:
        self._progress.close()

    def __enter__(self) -> "RunLog":
        return self

    def __exit__(self, *_) -> None:
        self.close()
