#!/usr/bin/env python3
"""Diagnostic only: run the EAGLE canary test with the Python TreeCore.

This branch replaces stage selection with one exact test. A passing workflow
is evidence for this control only, not validation of the requested stage.
"""

import os
import sys
from pathlib import Path


if __name__ == "__main__":
    test_path = (
        Path(__file__).resolve().parent
        / "registered"
        / "mock_model"
        / "test_e2e_spec_eagle.py"
    )
    os.environ["SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND"] = "python"
    os.environ["SGLANG_TEST_MAX_RETRY"] = "0"
    print(
        "DIAGNOSTIC ONLY: ignoring stage arguments and running "
        f"{test_path} with Python TreeCore, first-decode phase/index diagnostics, and no retries. "
        "This is not full-stage validation.",
        flush=True,
    )
    os.execv(sys.executable, [sys.executable, str(test_path)])
