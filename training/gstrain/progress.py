"""Console stage reporting, so long startup phases are not silent."""

from __future__ import annotations

import time


class Reporter:
    def __init__(self, total_stages: int = 0, quiet: bool = False) -> None:
        self.total_stages = total_stages
        self.quiet = quiet
        self.index = 0
        self._started: float | None = None

    def banner(self, message: str) -> None:
        if not self.quiet:
            print(message, flush=True)

    def stage(self, message: str) -> None:
        self.index += 1
        self._started = time.perf_counter()
        if not self.quiet:
            label = f"[{self.index}/{self.total_stages}]" if self.total_stages else "[*]"
            print(f"{label} {message}", flush=True)

    def detail(self, message: str) -> None:
        if not self.quiet:
            print(f"      {message}", flush=True)

    def done(self, message: str = "") -> None:
        if self.quiet or self._started is None:
            return
        elapsed = time.perf_counter() - self._started
        print(f"      {message + ', ' if message else ''}{elapsed:.1f}s", flush=True)

    def track(self, items, description: str, total: int | None = None):
        if self.quiet:
            return items
        try:
            from tqdm import tqdm
        except ImportError:
            return items
        return tqdm(items, total=total, desc=f"      {description}", unit="img", leave=False)
