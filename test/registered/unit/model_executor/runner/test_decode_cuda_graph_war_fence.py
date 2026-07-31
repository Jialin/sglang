import contextlib
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode, PPProxyTensors
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_utils import WarReadDonePolicy
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _SpecAlgorithm:
    def __init__(self, dflash: bool = False):
        self._dflash = dflash

    def is_dflash_family(self) -> bool:
        return self._dflash


def _attn_backend(*, breakable_metadata=False):
    return SimpleNamespace(
        use_captured_forward_metadata_for_breakable_cuda_graph=breakable_metadata,
    )


def _runner(*, dflash: bool = False, planted: bool = False):
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    runner.model_runner = SimpleNamespace(
        spec_algorithm=_SpecAlgorithm(dflash),
        device_timer=None,
        is_draft_worker=False,
        war_read_done_event=None,
        war_fastpath_read_done_event=None,
    )
    runner._war_read_done_node_planted = planted
    return runner


def test_war_read_done_policy():
    # Planted node: the graph re-arms it every replay.
    assert (
        _runner(planted=True)._war_read_done_policy(_attn_backend(), ForwardMode.DECODE)
        is WarReadDonePolicy.IN_GRAPH
    )
    # No node, snapshot backend: all shared reads finish before launch.
    assert (
        _runner()._war_read_done_policy(_attn_backend(), ForwardMode.DECODE)
        is WarReadDonePolicy.PRE_REPLAY
    )
    # Captured-metadata verify keeps reading throughout the graph, even planted.
    assert (
        _runner(planted=True)._war_read_done_policy(
            _attn_backend(breakable_metadata=True), ForwardMode.TARGET_VERIFY
        )
        is WarReadDonePolicy.POST_REPLAY
    )


def test_publish_war_read_done():
    runner = _runner()
    graph_event = object()
    runner.model_runner.war_read_done_event = graph_event
    runner._publish_war_read_done(in_graph=True)
    assert runner.model_runner.war_fastpath_read_done_event is graph_event

    recorded = []

    class Event:
        def record(self):
            recorded.append(self)

    runner.device_module = SimpleNamespace(Event=Event)
    runner._publish_war_read_done(in_graph=False)
    published = runner.model_runner.war_fastpath_read_done_event
    assert isinstance(published, Event) and recorded == [published]


def _execute_harness(runner, calls):
    key = ShapeKey(size=1)
    output = PPProxyTensors({"hidden_states": torch.ones(1, 1)})
    runner.ragged_verify_mode = False
    runner.bs = 1
    runner.load_batch = lambda *_: setattr(runner, "_replay_graph_key", key)

    class Backend:
        def replay_session(self):
            return contextlib.nullcontext()

        def replay(self, replay_key, _forward_batch):
            assert replay_key == key
            calls.append("replay")
            return output

    runner.backend = Backend()
    return SimpleNamespace(forward_mode=ForwardMode.DECODE, batch_size=1)


def test_execute_publishes_the_planted_graph_event():
    runner = _runner(planted=True)
    graph_event = object()
    runner.model_runner.war_read_done_event = graph_event
    runner.attn_backend = _attn_backend()
    runner.device_module = SimpleNamespace(
        Event=lambda: (_ for _ in ()).throw(
            AssertionError("execute must reuse the graph-recorded event")
        )
    )
    calls = []
    forward_batch = _execute_harness(runner, calls)

    result = runner.execute(forward_batch)

    assert result.tensors["hidden_states"].shape == (1, 1)
    assert runner.model_runner.war_fastpath_read_done_event is graph_event


def test_execute_records_pre_replay_for_snapshot_backends():
    runner = _runner()
    runner.attn_backend = _attn_backend()
    calls = []

    class Event:
        def record(self):
            calls.append("record")

    runner.device_module = SimpleNamespace(Event=Event)
    forward_batch = _execute_harness(runner, calls)

    runner.execute(forward_batch)

    # The eager record lands before the replay so the fence stays truthful.
    assert calls == ["record", "replay"]
    assert isinstance(runner.model_runner.war_fastpath_read_done_event, Event)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
