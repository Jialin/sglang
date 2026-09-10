"""Model-free SWA metadata benchmark and CPU/CUDA profiler captures.

GPU timings are CUDA-event graph replay intervals, including graph-launch
overhead. CPU timings measure eager enqueue work, not GPU execution. Neither
measurement includes allocator free-list updates or end-to-end serving.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import triton
from sglang.kernels.ops.memory.allocator import get_and_clear_swa_pages

WARMUP = 10
SAMPLES = 100
FLUSH_BYTES = 256 * 1024 * 1024


def reference(representatives, mapping, page_size):
    swa_tokens = mapping[representatives]
    peers_mapped = swa_tokens > 0
    if page_size == 1:
        swa_pages = swa_tokens
        mapping_indices = representatives
    else:
        swa_pages = swa_tokens // page_size
        base = (representatives // page_size) * page_size
        offsets = torch.arange(
            page_size, dtype=representatives.dtype, device=representatives.device
        )
        mapping_indices = (base[:, None] + offsets[None, :]).reshape(-1)
    mapping.index_fill_(0, mapping_indices, 0)
    return swa_pages, peers_mapped


def make_inputs(page_size, num_pages):
    pages = torch.randperm(2 * num_pages, device="cuda")[:num_pages] + 1
    full_indices = (
        pages[:, None] * page_size + torch.arange(page_size, device="cuda")[None, :]
    ).reshape(-1)
    representatives = full_indices[::page_size]
    original = torch.arange((2 * num_pages + 2) * page_size, device="cuda")
    return representatives, original


def verify(representatives, original, page_size):
    expected_mapping = original.clone()
    expected = reference(representatives, expected_mapping, page_size)
    actual_mapping = original.clone()
    actual = get_and_clear_swa_pages(representatives, actual_mapping, page_size)
    assert len(actual) == 3 and actual[2] is None
    assert torch.equal(expected[0], actual[0])
    assert torch.equal(expected[1], actual[1])
    assert torch.equal(expected_mapping, actual_mapping)


def measure(fn, representatives, mapping, original, page_size, flush):
    for _ in range(WARMUP):
        mapping.copy_(original)
        fn(representatives, mapping, page_size)
    torch.cuda.synchronize()

    mapping.copy_(original)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_outputs = fn(representatives, mapping, page_size)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    gpu_us = []
    for _ in range(SAMPLES):
        mapping.copy_(original)
        flush.zero_()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        gpu_us.append(start.elapsed_time(end) * 1000)

    cpu_us = []
    for _ in range(SAMPLES):
        mapping.copy_(original)
        torch.cuda.synchronize()
        begin = time.perf_counter_ns()
        outputs = fn(representatives, mapping, page_size)
        cpu_us.append((time.perf_counter_ns() - begin) / 1000)
        torch.cuda.synchronize()

    assert len(graph_outputs) == len(outputs)
    return {
        "gpu_graph_replay_us_median": statistics.median(gpu_us),
        "cpu_enqueue_us_median": statistics.median(cpu_us),
        "gpu_graph_replay_us_samples": gpu_us,
        "cpu_enqueue_us_samples": cpu_us,
    }


def capture_trace(output_dir, label, fn):
    page_size = 128
    num_frees = 8
    torch.manual_seed(0)
    representatives, original = make_inputs(page_size, num_frees)
    mapping = original.clone()
    one_page_inputs = [representatives[i : i + 1] for i in range(num_frees)]

    for _ in range(WARMUP):
        mapping.copy_(original)
        for one_page in one_page_inputs:
            fn(one_page, mapping, page_size)
    torch.cuda.synchronize()

    # Initialize profiler machinery in a discarded capture before measuring.
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ):
        mapping.copy_(original)
        for one_page in one_page_inputs:
            fn(one_page, mapping, page_size)
        torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        mapping.copy_(original)
        torch.cuda.synchronize()
        outputs = []
        for one_page in one_page_inputs:
            with torch.profiler.record_function(f"{label}.swa_metadata"):
                outputs.append(fn(one_page, mapping, page_size))
        torch.cuda.synchronize()

    assert len(outputs) == num_frees
    expected_mapping = original.clone()
    reference(representatives, expected_mapping, page_size)
    assert torch.equal(mapping, expected_mapping)
    trace_name = f"{label}.trace.json"
    prof.export_chrome_trace(str(output_dir / trace_name))
    return trace_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), "CUDA or HIP device required"
    torch.manual_seed(0)
    flush = torch.empty(FLUSH_BYTES, dtype=torch.int8, device="cuda")
    results = []
    for page_size in (1, 16, 64, 128):
        for num_pages in (1, 8, 64, 1024):
            representatives, original = make_inputs(page_size, num_pages)
            verify(representatives, original, page_size)
            mapping = original.clone()
            row = {"page_size": page_size, "pages": num_pages}
            for label, fn in (
                ("baseline", reference),
                ("fused", get_and_clear_swa_pages),
            ):
                row[label] = measure(
                    fn, representatives, mapping, original, page_size, flush
                )
            results.append(row)
            print(
                json.dumps(
                    {
                        "page_size": page_size,
                        "pages": num_pages,
                        **{
                            label: {
                                name: value
                                for name, value in row[label].items()
                                if name.endswith("median")
                            }
                            for label in ("baseline", "fused")
                        },
                    }
                ),
                flush=True,
            )

    traces = {
        label: capture_trace(args.output_dir, label, fn)
        for label, fn in (
            ("baseline", reference),
            ("fused", get_and_clear_swa_pages),
        )
    }
    report = {
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "methodology": {
            "scope": "SWA page lookup and mapping clear only; no model or free-list updates",
            "warmup_iterations": WARMUP,
            "samples_per_metric": SAMPLES,
            "gpu_metric": "CUDA-event graph replay interval including graph-launch overhead",
            "gpu_cache_flush_bytes_per_sample": FLUSH_BYTES,
            "cpu_metric": "eager enqueue wall time excluding explicit completion waits",
            "input_dtype": "int64",
            "mapping": "identity peers with distinct randomly selected positive FULL pages",
            "debug_checks": False,
            "timing_profiler_enabled": False,
            "trace_page_size": 128,
            "trace_sequential_one_page_frees": 8,
            "trace_python_stacks": False,
        },
        "results": results,
        "traces": traces,
    }
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Verified 16 input shapes; wrote results.json and baseline/fused traces.")


if __name__ == "__main__":
    main()
