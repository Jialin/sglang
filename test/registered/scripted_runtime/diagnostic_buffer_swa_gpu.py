"""Diagnostic only: run unchanged shared SWA buffer tests on both TreeCores.

Source base: PR 39627 port 87e4ae98ff3978aceb766e746fea6f8edd27df06.
"""

import argparse
import os
import subprocess
import sys
import unittest
from pathlib import Path


def run_backend(backend):
    test_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(test_root / "registered" / "unit" / "mem_cache"))
    import torch
    import test_unified_radix_cache_unittest as shared_suite

    assert torch.cuda.is_available(), "This diagnostic requires a real CUDA GPU"
    shared_suite._TREE_CORE_TEST_BACKEND = backend
    names = []
    for config in ["FULL_SWA_ps1_sw4", "FULL_SWA_ps4_sw4", "FULL_SWA_ps4_sw2"]:
        for method in [
            "test_buffer_only_load_back_uses_full_behind_swa_tombstone",
            "test_buffer_only_load_back_reuses_partial_masked_full",
        ]:
            names.append(f"Test_{config}.{method}")
    for method in [
        "test_buffer_only_load_back_trims_head_published_by_sibling",
        "test_buffer_only_masked_head_is_not_evicted_for_tail_load",
        "test_buffer_only_load_back_fail_stops_on_post_check_overlap",
        "test_buffer_only_swa_window_semantics",
        "test_buffer_only_hit_parks_on_swa_staging_shortfall",
    ]:
        names.append(f"Test_FULL_SWA_ps1_sw4.{method}")
    print(f"Diagnostic backend={backend}; selected={len(names)}; retries=0", flush=True)
    suite = unittest.defaultTestLoader.loadTestsFromNames(names, module=shared_suite)
    result = unittest.TextTestRunner(verbosity=2, failfast=True).run(suite)
    assert result.wasSuccessful(), f"{backend} SWA buffer regression failed"
    assert not result.skipped, f"Selected SWA regressions skipped: {result.skipped}"
    assert result.testsRun == len(names), (result.testsRun, len(names))
    print(f"Diagnostic backend={backend}: {result.testsRun}/{len(names)} passed; zero skips", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", nargs="?", choices=("python", "rust"))
    parser.add_argument("-f", action="store_true")
    args = parser.parse_args()
    if args.backend is not None:
        run_backend(args.backend)
        return
    process_env = dict(os.environ)
    process_env["SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND"] = "rust"
    process_env["SGLANG_TEST_MAX_RETRY"] = "0"
    for backend in ["python", "rust"]:
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), backend],
            env=process_env,
            check=True,
            timeout=1800,
        )
    print("Both backends passed all 11 unchanged SWA buffer regressions.", flush=True)


if __name__ == "__main__":
    main()
