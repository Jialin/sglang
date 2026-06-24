# Multi-Turn Serving Perf Benchmark

Measures **TTFT** (time-to-first-token) and **TTIT** (time-per-output-token,
a.k.a. TPOT / inter-token latency) per turn of a growing multi-turn
conversation, to quantify how radix-cache prefix reuse affects prefill cost as
the conversation deepens.

The driver (`bench_multi_turn_serving.py`) hits `/generate` with raw
`input_ids`, so the server's reported `prompt_tokens` equals the request length
exactly. Each turn N sends turn 1's prompt + every prior turn's output + a short
continuation, so the radix cache hits the whole accumulated prefix and only the
~10 new continuation tokens need cache-miss prefill. Turn 1 is cold prefill;
turns 2..N show cache-hit prefill as the tree grows.

## Usage

Launch a server (any radix-cache backend), then drive it from another terminal:

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-32B --tp-size 2 \
    --radix-cache-backend rust_unified_tree \
    --host 127.0.0.1 --port 30000

python benchmark/multi_turn_serving/bench_multi_turn_serving.py \
    --server-url http://127.0.0.1:30000 \
    --input-tokens 1000 \
    --output-tokens-per-turn 1000 \
    --max-num-turns 20 \
    --num-trials 3
```

The first trial is a warmup (discarded); the remaining trials are aggregated
per turn position into one row each. To A/B two backends, run the same command
against each server (e.g. `--radix-cache-backend rust_unified_tree` vs the
default) — the synthetic prompt is byte-identical per `trial_idx`, so any TTFT
delta is server-side.

## Metrics

| Metric | Definition |
|--------|------------|
| **TTFT** | Wall time from request submit to the first SSE chunk with a non-empty `output_ids`. Reflects prefill cost (cache-miss tokens × prefill rate) + HTTP/SSE transport. |
| **TTIT** | `(turn_duration − TTFT) / max(1, output_tokens − 1)`. Steady-state per-output-token decode latency. |

All latencies are client-side (include HTTP/SSE transport — sub-ms on
localhost, and equal across A/B arms). For pure server-side timings, the
`/generate` response `meta_info` carries `request_received_ts` /
`response_sent_to_client_ts` / `request_finished_ts`.

## Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--server-url` | `http://127.0.0.1:30000` | server base URL (no `/v1` suffix) |
| `--input-tokens` | 1000 | turn-1 prompt length, exact |
| `--output-tokens-per-turn` | 1000 | `max_new_tokens` per turn (`ignore_eos=True`) |
| `--max-num-turns` | 20 | conversation depth; each turn reported as its own row |
| `--num-trials` | 3 | total conversations (1 warmup + the rest data) |
| `--timeout-s` | 600 | per-turn request timeout |
| `--output-json` | — | dump raw per-trial timings to this path |

## Notes

- `ignore_eos=True` forces exactly `--output-tokens-per-turn` tokens per turn,
  so TTIT averaging isn't skewed by early EOS stops.
- The synthetic prompt is deterministic per `trial_idx` (reproducible run to
  run) but unique across trials (defeats trivial cross-trial cache hits on the
  turn-1 prompt).
- Token IDs are sampled from `[100, 100000)`; this assumes the model vocab is
  at least ~100K (true for Qwen3 / Llama families). Lower `_SYNTH_TOKEN_HI` in
  the driver for smaller-vocab models.
