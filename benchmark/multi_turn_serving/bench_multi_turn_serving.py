"""Multi-turn serving perf driver — TTFT + TTIT against a live SGLang server.

Measures, per turn, how radix-cache reuse across a growing conversation
affects prefill (TTFT) and decode (TTIT):
  * TTFT — time-to-first-token (seconds from request submit to the first
    non-empty token in the SSE stream). Reflects prefill cost:
    cache-miss tokens x prefill rate.
  * TTIT — time-per-output-token after first (a.k.a. TPOT / inter-token
    latency): (turn_duration - TTFT) / max(1, output_tokens - 1).

The driver hits SGLang's `/generate` endpoint with raw `input_ids`
(bypassing the tokenizer + chat template), so `prompt_tokens` in the
response equals `len(input_ids)` exactly — no drift. Turn N's input_ids =
turn 1's prompt + every prior turn's output + a short per-turn
continuation, so the radix cache should hit on the whole accumulated
prefix and only the ~10 new continuation tokens need cache-miss prefill.
Turn 1 therefore measures cold prefill; turns 2..N measure cache-hit
prefill cost as the tree deepens.

Example — launch a server, then drive it:

    python -m sglang.launch_server \\
        --model-path Qwen/Qwen3-32B --tp-size 2 \\
        --radix-cache-backend rust_unified_tree \\
        --host 127.0.0.1 --port 30000

    python benchmark/multi_turn_serving/bench_multi_turn_serving.py \\
        --server-url http://127.0.0.1:30000 \\
        --input-tokens 1000 \\
        --output-tokens-per-turn 1000 \\
        --max-num-turns 20 \\
        --num-trials 3

The first trial is a warmup (discarded); the remaining trials are
aggregated per turn position into one row each. Pass `--output-json` to
dump raw per-trial timings.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import requests


@dataclass
class TurnTiming:
    turn_idx: int  # 0-based
    ttft_s: float  # time-to-first-token
    ttit_s: float  # time-per-output-token after first
    prompt_tokens: int  # accumulated input tokens at submission time
    output_tokens: int  # actual emitted
    duration_s: float  # full turn wall time
    gpu_prefill_ms: float = 0.0  # per-turn GPU prefill forward time (ms)


@dataclass
class TrialResult:
    trial_idx: int
    num_turns: int
    turns: list[TurnTiming] = field(default_factory=list)


# Token-ID range for the synthetic prompt. Avoid the bottom of the vocab
# (often special tokens: BOS/EOS/PAD/UNK). The range [100, 100000) gives
# ~100K distinct IDs to sample from — enough diversity to defeat trivial
# cross-trial cache hits. Assumes the model's vocab is at least this large
# (true for Qwen3 / Llama families); shrink `_SYNTH_TOKEN_HI` for smaller
# vocabularies.
_SYNTH_TOKEN_LO = 100
_SYNTH_TOKEN_HI = 100_000

# Continuation token IDs between turns come from a small reserved band, so
# they're visually distinct from the bulk prompt in traces.
_CONTINUE_TOKEN_BASE = 90_000


def _synthetic_prompt_ids(num_tokens: int, trial_idx: int = 0) -> list[int]:
    """Return a deterministic list of `num_tokens` random token IDs, seeded
    by `trial_idx`. Each trial gets a fresh sequence that is reproducible
    across runs but unique to this trial — defeats cross-trial cache hits on
    the initial prompt while keeping the cache key stable across A/A runs."""
    rng = random.Random(trial_idx)
    return [
        rng.randint(_SYNTH_TOKEN_LO, _SYNTH_TOKEN_HI - 1) for _ in range(num_tokens)
    ]


def _continue_token_ids(turn_idx: int) -> list[int]:
    """Short fixed-length token-ID sequence representing the continuation
    "prompt" between turns. ~10 tokens, deterministic per turn position."""
    return [_CONTINUE_TOKEN_BASE + turn_idx * 100 + i for i in range(10)]


def _stream_generate(
    server_url: str,
    input_ids: list[int],
    max_new_tokens: int,
    timeout_s: float,
) -> tuple[float, float, list[int], int, int, float]:
    """Send a streaming `/generate` request with raw token IDs; return
        (ttft_s, duration_s, output_ids, prompt_tokens, completion_tokens,
         gpu_prefill_ms)

    `output_ids` is the final accumulated list (the server emits the
    cumulative list each chunk; we keep the last). TTFT is the wall time
    from request submit to the first SSE chunk carrying a non-empty
    `output_ids`.
    """
    url = server_url.rstrip("/") + "/generate"
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            # Run out the full budget so TTIT is independent of whether the
            # model would otherwise stop early on EOS.
            "ignore_eos": True,
        },
        "stream": True,
    }
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

    start = time.perf_counter()
    first_token_time: float | None = None
    output_ids: list[int] = []
    prompt_tokens = 0
    completion_tokens = 0
    gpu_prefill_ms = 0.0

    with requests.post(
        url,
        json=payload,
        headers=headers,
        stream=True,
        timeout=timeout_s,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            cur_output_ids = event.get("output_ids")
            if cur_output_ids:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                output_ids = cur_output_ids
            meta = event.get("meta_info") or {}
            if meta.get("prompt_tokens") is not None:
                prompt_tokens = int(meta["prompt_tokens"])
            if meta.get("completion_tokens") is not None:
                completion_tokens = int(meta["completion_tokens"])
            if meta.get("gpu_prefill_ms") is not None:
                gpu_prefill_ms = float(meta["gpu_prefill_ms"])

    end = time.perf_counter()
    if first_token_time is None:
        first_token_time = end
    ttft_s = first_token_time - start
    duration_s = end - start
    # Fall back to len(output_ids) if the server didn't emit
    # completion_tokens in meta_info (defensive — SGLang does).
    completion_tokens = max(completion_tokens, len(output_ids))
    return (
        ttft_s,
        duration_s,
        output_ids,
        prompt_tokens,
        completion_tokens,
        gpu_prefill_ms,
    )


def run_trial(
    trial_idx: int,
    server_url: str,
    input_tokens: int,
    output_tokens_per_turn: int,
    num_turns: int,
    timeout_s: float,
) -> TrialResult:
    """Run a single multi-turn conversation against `/generate` with raw
    token IDs. Each turn N's input_ids = [turn 1's input] + [turn 1's
    output] + [continue tokens for turn 2] + ... + [continue tokens for
    turn N]. The accumulated input grows by exactly
    `output_tokens_per_turn + 10` tokens per turn, and turn N's prefix
    overlaps perfectly with turn N-1's payload — so the radix cache should
    hit on everything except the new ~10 continue tokens.
    """
    result = TrialResult(trial_idx=trial_idx, num_turns=num_turns)
    base_prompt_ids = _synthetic_prompt_ids(input_tokens, trial_idx)
    input_ids: list[int] = list(base_prompt_ids)

    for turn_idx in range(num_turns):
        if turn_idx > 0:
            input_ids = input_ids + _continue_token_ids(turn_idx)
        (
            ttft_s,
            duration_s,
            output_ids,
            prompt_tokens,
            completion_tokens,
            gpu_prefill_ms,
        ) = _stream_generate(
            server_url=server_url,
            input_ids=input_ids,
            max_new_tokens=output_tokens_per_turn,
            timeout_s=timeout_s,
        )
        post_first_tokens = max(1, completion_tokens - 1)
        ttit_s = (duration_s - ttft_s) / post_first_tokens
        result.turns.append(
            TurnTiming(
                turn_idx=turn_idx,
                ttft_s=ttft_s,
                ttit_s=ttit_s,
                prompt_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                duration_s=duration_s,
                gpu_prefill_ms=gpu_prefill_ms,
            )
        )
        # Append the assistant's output_ids so the next turn's prefix
        # naturally extends from this turn's full request + response.
        input_ids = input_ids + output_ids
    return result


def _stats_ms(xs_s: list[float]) -> dict[str, float]:
    """Convert a list of second-valued timings into ms-valued stats."""
    if not xs_s:
        return {"mean": 0.0, "p50": 0.0, "min": 0.0, "max": 0.0, "stdev": 0.0}
    xs_ms = [x * 1000 for x in xs_s]
    return {
        "mean": statistics.mean(xs_ms),
        "p50": statistics.median(xs_ms),
        "min": min(xs_ms),
        "max": max(xs_ms),
        "stdev": statistics.stdev(xs_ms) if len(xs_ms) > 1 else 0.0,
    }


def _aggregate_per_step(
    trials: list[TrialResult], report_steps_1idx: list[int]
) -> list[dict[str, Any]]:
    """Aggregate per-step across trials: one row per `report_steps_1idx`
    entry, each collapsing one turn position across all trials. All trials
    must share the same conversation depth (caller strips the warmup)."""
    rows: list[dict[str, Any]] = []
    for turn_1idx in sorted(set(report_steps_1idx)):
        step = turn_1idx - 1  # 0-indexed within the conversation
        ttfts = [t.turns[step].ttft_s for t in trials if step < len(t.turns)]
        ttits = [t.turns[step].ttit_s for t in trials if step < len(t.turns)]
        prompt_lens = [
            t.turns[step].prompt_tokens for t in trials if step < len(t.turns)
        ]
        output_lens = [
            t.turns[step].output_tokens for t in trials if step < len(t.turns)
        ]
        if not ttfts:
            continue
        ttft_stats = _stats_ms(ttfts)
        ttit_stats = _stats_ms(ttits)
        rows.append(
            {
                "turn": turn_1idx,
                "trials": len(ttfts),
                "prompt_tokens_avg": statistics.mean(prompt_lens),
                "output_tokens_avg": statistics.mean(output_lens),
                "ttft_ms_mean": ttft_stats["mean"],
                "ttft_ms_p50": ttft_stats["p50"],
                "ttft_ms_min": ttft_stats["min"],
                "ttft_ms_max": ttft_stats["max"],
                "ttft_ms_stdev": ttft_stats["stdev"],
                "ttit_ms_mean": ttit_stats["mean"],
                "ttit_ms_p50": ttit_stats["p50"],
                "ttit_ms_min": ttit_stats["min"],
                "ttit_ms_max": ttit_stats["max"],
                "ttit_ms_stdev": ttit_stats["stdev"],
            }
        )
    return rows


def _print_table(rows: list[dict[str, Any]]) -> None:
    """Render the per-step summary table — one row per turn position
    (1-indexed), aggregated across the data trials."""
    print()
    print(
        f"{'turn':>5}  {'trials':>6}  "
        f"{'prompt_tok':>10}  {'output_tok':>10}  "
        f"{'TTFT(ms) mean(±std)':>21}  {'TTFT p50':>9}  "
        f"{'TTIT(ms) mean(±std)':>21}  {'TTIT p50':>9}"
    )
    print("-" * 110)
    for r in rows:
        ttft_meanpm = f"{r['ttft_ms_mean']:.1f} (±{r['ttft_ms_stdev']:.1f})"
        ttit_meanpm = f"{r['ttit_ms_mean']:.2f} (±{r['ttit_ms_stdev']:.2f})"
        print(
            f"{r['turn']:>5}  {r['trials']:>6}  "
            f"{r['prompt_tokens_avg']:>10.0f}  {r['output_tokens_avg']:>10.0f}  "
            f"{ttft_meanpm:>21}  {r['ttft_ms_p50']:>9.1f}  "
            f"{ttit_meanpm:>21}  {r['ttit_ms_p50']:>9.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:30000",
        help="Base URL of the SGLang server (without /v1 suffix).",
    )
    parser.add_argument(
        "--input-tokens",
        type=int,
        default=1000,
        help="Exact turn-1 input prompt length in tokens (deterministically "
        "random integer IDs, seeded by trial_idx).",
    )
    parser.add_argument(
        "--output-tokens-per-turn",
        type=int,
        default=1000,
        help="Max tokens to generate per turn.",
    )
    parser.add_argument(
        "--max-num-turns",
        type=int,
        default=20,
        help="Conversation depth (in turns) each trial runs. Every turn "
        "from 1 to max-num-turns is reported as its own row.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=3,
        help="Total conversations to run, INCLUDING the warmup trial. The "
        "first trial is the warmup (timings discarded); the rest contribute "
        "to the per-step aggregates.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=600.0,
        help="Per-turn request timeout.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="If set, write raw per-trial timings to this JSON path.",
    )
    args = parser.parse_args()

    if args.num_trials < 2:
        parser.error(
            "--num-trials must be >= 2 (one warmup + at least one data "
            "trial). Got: %d" % args.num_trials
        )

    conversation_depth = args.max_num_turns
    report_steps = list(range(1, conversation_depth + 1))

    print(
        f"\n--- Running num_trials={args.num_trials} conversations, "
        f"{conversation_depth} turns each. First trial is warmup. ---"
    )
    print(
        f"--- Reporting every turn position 1..{conversation_depth} "
        f"(1-indexed). ---\n"
    )

    # trial_idx=0 is the warmup (discarded); 1..N-1 contribute to aggregates.
    all_trials: list[TrialResult] = []
    for trial_idx in range(args.num_trials):
        label = (
            "WARMUP" if trial_idx == 0 else f"trial {trial_idx}/{args.num_trials - 1}"
        )
        print(f"  [{label}] running {conversation_depth}-turn conversation...")
        result = run_trial(
            trial_idx=trial_idx,
            server_url=args.server_url,
            input_tokens=args.input_tokens,
            output_tokens_per_turn=args.output_tokens_per_turn,
            num_turns=conversation_depth,
            timeout_s=args.timeout_s,
        )
        all_trials.append(result)
        first_step = result.turns[0]
        last_step = result.turns[-1]
        print(
            f"    turn 1: TTFT={first_step.ttft_s * 1000:.1f}ms "
            f"(gpu={first_step.gpu_prefill_ms:.1f}ms), "
            f"prompt_tok={first_step.prompt_tokens}; "
            f"turn {len(result.turns)}: TTFT={last_step.ttft_s * 1000:.1f}ms "
            f"(gpu={last_step.gpu_prefill_ms:.1f}ms), "
            f"prompt_tok={last_step.prompt_tokens}"
        )

    data_trials = all_trials[1:]  # discard warmup
    rows = _aggregate_per_step(data_trials, report_steps)
    _print_table(rows)

    if args.output_json:
        payload = {
            "args": {
                "server_url": args.server_url,
                "input_tokens": args.input_tokens,
                "output_tokens_per_turn": args.output_tokens_per_turn,
                "max_num_turns": args.max_num_turns,
                "num_trials": args.num_trials,
            },
            # `trials` includes the warmup (trial_idx=0); filter by trial_idx
            # for data-only.
            "trials": [
                {
                    "trial_idx": tr.trial_idx,
                    "is_warmup": tr.trial_idx == 0,
                    "num_turns": tr.num_turns,
                    "turns": [
                        {
                            "turn_idx": t.turn_idx,
                            "ttft_s": t.ttft_s,
                            "ttit_s": t.ttit_s,
                            "prompt_tokens": t.prompt_tokens,
                            "output_tokens": t.output_tokens,
                            "duration_s": t.duration_s,
                            "gpu_prefill_ms": t.gpu_prefill_ms,
                        }
                        for t in tr.turns
                    ],
                }
                for tr in all_trials
            ],
        }
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nRaw timings written to {args.output_json}")


if __name__ == "__main__":
    main()
