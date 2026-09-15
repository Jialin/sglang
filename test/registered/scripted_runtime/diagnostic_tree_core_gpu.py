"""Diagnostic only: verify the three repaired scripted TreeCore GPU files.

Source base: PR 39627 tested merge 39d3887751b54dd0a0e7c28c58475f6562225cae.
This wrapper is isolated from the PR branch and changes no test assertions.
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    test_root = Path(__file__).resolve().parents[2]
    process_env = dict(os.environ)
    process_env["SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND"] = "rust"
    process_env["SGLANG_TEST_MAX_RETRY"] = "0"
    files = (
        "registered/chunked_prefill/test_scripted_core_1gpu.py",
        "registered/scripted_runtime/test_scripted_runtime_core.py",
        "registered/chunked_prefill/test_scripted_swa_1gpu.py",
    )
    for filename in files:
        print(f"Diagnostic: {filename}; Rust selected, test retries disabled", flush=True)
        subprocess.run(
            [sys.executable, filename, "-v", "-f"],
            cwd=test_root,
            env=process_env,
            check=True,
            timeout=1800,
        )
    print("All three unchanged scripted GPU test files passed.", flush=True)


if __name__ == "__main__":
    main()
