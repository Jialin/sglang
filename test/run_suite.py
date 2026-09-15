#!/usr/bin/env python3
"""Diagnostic only: run NIXL disaggregation with automatic UCX device selection.

This replaces stage selection with one exact test file. Its result is evidence
for this network control only, not validation of the requested stage.
"""

import os
import sys
from pathlib import Path

if __name__ == "__main__":
    test_path = (
        Path(__file__).resolve().parent
        / "registered"
        / "amd"
        / "disaggregation"
        / "test_nixl_transfer_engine_e2e.py"
    )
    os.environ["UCX_NET_DEVICES"] = "all"
    os.environ["SGLANG_TEST_MAX_RETRY"] = "0"
    print(
        "DIAGNOSTIC ONLY: ignoring stage arguments and running "
        f"{test_path} with UCX_NET_DEVICES=all and no retries. "
        "This is not full-stage validation.",
        flush=True,
    )
    os.execv(sys.executable, [sys.executable, str(test_path)])
