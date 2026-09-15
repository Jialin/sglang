#!/usr/bin/env python3
"""Diagnostic only: rerun BasicSanity with the original Rust backend.

The full file preserves preceding accuracy/cache work before concurrency.
This is one-file diagnostic evidence, not validation of the requested stage.
"""

import os
import sys
from pathlib import Path

if __name__ == "__main__":
    test_path = (
        Path(__file__).resolve().parent / "registered" / "core" / "test_basic_sanity.py"
    )
    os.environ["SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND"] = "rust"
    os.environ["SGLANG_TEST_MAX_RETRY"] = "0"
    print(
        "DIAGNOSTIC ONLY: ignoring stage arguments and running the full "
        f"{test_path} with Rust and no retries. The soft watchdog dumps "
        "stacks after 30 seconds without changing the client timeout or "
        "hard watchdog. This is not full-stage validation.",
        flush=True,
    )
    os.execv(sys.executable, [sys.executable, str(test_path)])
