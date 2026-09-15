#!/usr/bin/env python3
"""Diagnostic only: one deterministic remote-weight transfer with Python TreeCore.

This branch replaces stage selection with one exact test in partition zero.
The workflow is control evidence only, not validation of the requested stage.
"""

import argparse
import os
import sys
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-partition-id", type=int, default=0)
    args, _ = parser.parse_known_args()
    if args.auto_partition_id != 0:
        print(
            "DIAGNOSTIC ONLY: no workload in this partition; "
            "the single controlled attempt belongs to partition zero. "
            "This is not full-stage validation.",
            flush=True,
        )
        raise SystemExit(0)

    test_path = (
        Path(__file__).resolve().parent
        / "registered"
        / "model_loading"
        / "test_load_weights_from_remote_instance.py"
    )
    os.environ["SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND"] = "python"
    os.environ["SGLANG_TEST_MAX_RETRY"] = "0"
    print(
        "DIAGNOSTIC ONLY: running "
        f"{test_path} with Python TreeCore, Server/transfer_engine, and no retries. "
        "This is not full-stage validation.",
        flush=True,
    )
    os.execv(sys.executable, [sys.executable, str(test_path)])
