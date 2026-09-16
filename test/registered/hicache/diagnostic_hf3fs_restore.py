"""Diagnostic-only entry point for the unchanged 3FS storage test file."""

import argparse
import os
import unittest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("python", "rust"), default="rust")
    parser.add_argument("-f", "--failfast", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()

    os.environ["SGLANG_TEST_MAX_RETRY"] = "0"
    os.environ["SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND"] = args.backend

    import test_hicache_storage_3fs_backend as original_tests

    suite = unittest.defaultTestLoader.loadTestsFromModule(original_tests)
    assert suite.countTestCases() == 3, suite.countTestCases()
    print(
        f"HF3FS restore diagnostic: backend={args.backend}, original_tests=3, "
        "method_retries=0, failfast=True",
        flush=True,
    )
    if args.collect_only:
        return

    result = unittest.TextTestRunner(verbosity=2, failfast=True).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    assert result.testsRun == 3, result.testsRun
    assert not result.skipped, result.skipped


if __name__ == "__main__":
    main()
