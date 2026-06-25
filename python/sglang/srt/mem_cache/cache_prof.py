"""TEMPORARY profiling proxy for the Rust tree engine (NOT for commit).

Env-gated breakdown of per-op Rust cache PyO3 cost. Enable with
SGLANG_CACHE_PROF=1. Measurement starts only AFTER SGLANG_CACHE_PROF_WARMUP
match_prefix calls (default 100), so warmup is excluded. Dumps a per-op
breakdown to stderr every 50 measured match_prefix calls.
"""

import collections
import os
import sys
import time


def maybe_wrap(tree):
    if os.environ.get("SGLANG_CACHE_PROF") != "1":
        return tree
    print("[CACHEPROF] enabled (timing _tree PyO3 ops)", file=sys.stderr, flush=True)
    return _Proxy(tree)


class _Proxy:
    def __init__(self, inner):
        self._inner = inner
        self._data = collections.defaultdict(lambda: [0, 0.0])  # op -> [calls, total_s]
        self._mp = 0
        self._warmup = int(os.environ.get("SGLANG_CACHE_PROF_WARMUP", "100"))
        self._active = False

    def __getattr__(self, name):
        # Only invoked for names not on the proxy itself (i.e. inner's API).
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def wrapped(*a, **k):
            t = time.perf_counter()
            r = attr(*a, **k)
            self._tick(name, time.perf_counter() - t)
            return r

        return wrapped

    def _tick(self, name, dt):
        if name == "match_prefix":
            self._mp += 1
            if self._mp == self._warmup:
                self._active = True
                print(f"[CACHEPROF] warmup done at {self._mp} match_prefix; measuring now",
                      file=sys.stderr, flush=True)
            if self._active and self._mp % 50 == 0:
                self._dump()
        if self._active:
            d = self._data[name]
            d[0] += 1
            d[1] += dt

    def _dump(self):
        nreq = self._data.get("match_prefix", [0, 0.0])[0]
        tot = sum(t for _, t in self._data.values())
        print(f"[CACHEPROF] === post-warmup breakdown, {nreq} measured requests ===",
              file=sys.stderr, flush=True)
        for op, (c, t) in sorted(self._data.items(), key=lambda x: -x[1][1]):
            print(f"[CACHEPROF]   {op:28} calls={c:7} total={t*1000:9.1f}ms "
                  f"/req={t/max(1,nreq)*1000:7.3f}ms mean={t/max(1,c)*1e6:8.1f}us",
                  file=sys.stderr, flush=True)
        print(f"[CACHEPROF]   TOTAL cache-PyO3 /req = {tot/max(1,nreq)*1000:.3f}ms",
              file=sys.stderr, flush=True)


# --- TEMP allocator profiler (SGLANG_ALLOC_PROF=1) ----------------------------
# Monkeypatches the KV allocator INSTANCE's alloc/free methods in place (object
# identity preserved, so isinstance checks still work) to time + count tokens.
# Dumps a cumulative per-method breakdown every 200 measured free calls.
_ALLOC_METHODS = ("alloc", "free", "free_swa", "alloc_extend", "alloc_decode")


def maybe_wrap_allocator(alloc):
    if alloc is None or os.environ.get("SGLANG_ALLOC_PROF") != "1":
        return alloc
    data = collections.defaultdict(lambda: [0, 0.0, 0])  # name -> [calls, secs, tokens]
    st = {"free": 0, "warmup": int(os.environ.get("SGLANG_ALLOC_PROF_WARMUP", "200")),
          "on": False}

    def _ntok(name, a):
        try:
            if name in ("free", "free_swa"):
                return int(a[0].numel())
            if name == "alloc":
                return int(a[0])
            if name == "alloc_extend":
                return int(a[5])  # extend_num_tokens
        except Exception:
            pass
        return 0

    def mk(name, fn):
        def w(*a, **k):
            t = time.perf_counter()
            r = fn(*a, **k)
            dt = time.perf_counter() - t
            if st["on"]:
                d = data[name]; d[0] += 1; d[1] += dt; d[2] += _ntok(name, a)
            if name == "free":
                st["free"] += 1
                if st["free"] == st["warmup"]:
                    st["on"] = True
                    print("[ALLOCPROF] warmup done; measuring now", file=sys.stderr, flush=True)
                if st["on"] and st["free"] % 200 == 0:
                    _adump(data)
            return r
        return w

    print(f"[ALLOCPROF] enabled on {type(alloc).__name__}", file=sys.stderr, flush=True)
    for name in _ALLOC_METHODS:
        fn = getattr(alloc, name, None)
        if callable(fn):
            try:
                setattr(alloc, name, mk(name, fn))
            except Exception as e:  # noqa: BLE001
                print(f"[ALLOCPROF] skip {name}: {e}", file=sys.stderr, flush=True)
    return alloc


def _adump(data):
    n = data.get("free", [1, 0.0, 0])[0] or 1
    print(f"[ALLOCPROF] === {n} measured free calls ===", file=sys.stderr, flush=True)
    for nm, (c, s, tk) in sorted(data.items(), key=lambda x: -x[1][1]):
        print(f"[ALLOCPROF]   {nm:14} calls={c:7} total={s*1000:8.1f}ms "
              f"mean={s/max(1,c)*1e6:7.1f}us /freecall={s/n*1e6:7.1f}us tok={tk}",
              file=sys.stderr, flush=True)
