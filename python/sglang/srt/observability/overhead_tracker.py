"""Opt-in (SGLANG_TRACK_OVERHEAD=1) non-perturbing overhead accumulators.

perf_counter deltas are summed per bucket; a windowed breakdown is dumped via
logger.warning every N driver events. No per-call I/O — only accumulate + dump.
"""

import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)
perf_counter = time.perf_counter  # bind once: avoid attr lookup on hot path

# Module-const gate read once at import (scheduler sets env at launch).
TRACK_OVERHEAD = envs.SGLANG_TRACK_OVERHEAD.get() > 0


class OverheadTracker:
    """Sum perf_counter deltas per bucket; dump windowed breakdown periodically.

    The window is advanced only by `add(..., driver=True)` events so one logical
    unit (a step, or a request) drives the dump cadence regardless of how many
    sub-buckets fire per unit.
    """

    def __init__(self, name: str, window: Optional[int] = None):
        self.name = name
        self.window = window or envs.SGLANG_TRACK_OVERHEAD_WINDOW.get()
        self._sum = defaultdict(float)  # bucket -> total_s
        self._cnt = defaultdict(int)  # bucket -> n calls
        self._driver = None  # bucket whose count drives the window
        self._driver_seen = 0

    def add(self, bucket: str, dt: float, *, driver: bool = False):
        self._sum[bucket] += dt
        self._cnt[bucket] += 1
        if driver:
            self._driver = bucket
            self._driver_seen += 1
            if self._driver_seen % self.window == 0:
                self._dump()

    def _dump(self):
        n = max(1, self._cnt.get(self._driver, 1))
        tot = sum(self._sum.values())
        lines = [
            f"[OVERHEAD:{self.name}] window={self.window} driver={self._driver} "
            f"n={n} total/n={tot / n * 1e3:.4f}ms"
        ]
        for b, s in sorted(self._sum.items(), key=lambda x: -x[1]):
            lines.append(
                f"  {b:32} calls={self._cnt[b]:8} "
                f"/n={s / n * 1e3:8.4f}ms mean={s / max(1, self._cnt[b]) * 1e6:8.2f}us"
            )
        logger.warning("\n".join(lines))
        self._sum.clear()
        self._cnt.clear()
        self._driver_seen = 0


@contextmanager
def _noop():
    yield


@contextmanager
def _phase(tracker: "OverheadTracker", bucket: str, driver: bool):
    t = perf_counter()
    try:
        yield
    finally:
        tracker.add(bucket, perf_counter() - t, driver=driver)


def phase(tracker: Optional["OverheadTracker"], bucket: str, *, driver: bool = False):
    """Context-manager helper (used by tracker B's hook loops, NOT the scheduler
    hot loop which uses manual gated timing for a true zero-cost off-path)."""
    if tracker is None:
        return _noop()
    return _phase(tracker, bucket, driver)
