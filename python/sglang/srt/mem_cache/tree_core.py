# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The swappable tree-engine contract (`TreeCore`).

`TreeCore` is the radix tree + intrusive LRU + lock-ref accounting + eviction
engine. The Rust core (`RustPageRadixCacheWrapper` / `RustBigramRadixCacheWrapper`)
is one implementation; a pure-Python radix tree could be another. The orchestrator
(`RustUnifiedRadixCache`) and the tree-component handlers depend only on this
contract, so the engine can be swapped without touching either.

The result objects (insert / match / evict) and `ComponentType` are typed `Any`
here on purpose: a follow-up moves those to a language-neutral module (today they
come from the Rust extension `_mem_cache_core`). See the migration plan in the PR.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol


class TreeCore(Protocol):
    """The cross-layer contract any tree engine (Rust or Python) must satisfy."""

    # --- tree ops ---
    def match_prefix(self, token_ids: list[int], extra_key: Any) -> Any: ...

    def insert(
        self,
        token_ids: list[int],
        value: Any,
        extra_key: Any,
        prev_prefix_len: int,
        swa_evicted_seqlen: int,
        mamba_value: Any,
    ) -> Any: ...  # result carries `.deferred_actions`

    def evict(self, budgets: list[int]) -> Any: ...  # result carries `.freed`, `.evicted`

    def inc_lock_ref(self, node: Any) -> tuple[int, Optional[int]]: ...

    def dec_lock_ref(self, node: Any, swa_uuid_for_lock: Optional[int]) -> int: ...

    def reset(self) -> None: ...

    # --- pool bridge + size accounting ---
    def apply_swa_writes(self, node_indices: list[int], values: list[Any]) -> None: ...

    def component_evictable_size(self, component_type: Any) -> int: ...

    def component_protected_size(self, component_type: Any) -> int: ...

    def component_total_size(self, component_type: Any) -> int: ...

    def total_size(self) -> tuple[int, int]: ...
