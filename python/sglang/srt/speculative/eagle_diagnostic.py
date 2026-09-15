"""Temporary diagnostic branch only: localize the first full-batch EAGLE fault."""

import torch

_active = False
_rounds = 0


def begin(batch):
    global _active, _rounds
    _active = (
        not batch.forward_mode.is_extend()
        and not batch.forward_mode.is_prebuilt()
        and len(batch.reqs) >= 32
        and _rounds < 3
    )
    if _active:
        _rounds += 1
        print(
            f"EAGLE-DIAG round={_rounds} mode={batch.forward_mode} bs={len(batch.reqs)}",
            flush=True,
        )
        sync("scheduler.before_seq_lens_cpu")


def sync(phase):
    if not _active:
        return
    print(f"EAGLE-DIAG before-sync {phase}", flush=True)
    torch.cuda.synchronize()
    print(f"EAGLE-DIAG after-sync {phase}", flush=True)


def kv_gather(backend, phase, req_indices, seq_lens, indptr, indices, compare):
    if not _active:
        return
    sync(phase)
    rows = req_indices.detach().cpu().long()
    lens = seq_lens.detach().cpu().long()
    ptr = indptr.detach().cpu().long()
    table = backend.req_to_token.detach().cpu()
    capacity = backend.token_to_kv_pool.get_key_buffer(0).shape[0]
    assert rows.numel() == lens.numel()
    assert torch.all((rows >= 0) & (rows < table.shape[0])), rows.tolist()
    assert torch.all((lens >= 0) & (lens <= table.shape[1])), lens.tolist()
    expected_ptr = torch.cat((torch.zeros(1, dtype=torch.long), lens.cumsum(0)))
    assert torch.equal(ptr, expected_ptr), (ptr.tolist(), expected_ptr.tolist())
    total = int(ptr[-1])
    assert total <= indices.numel(), (total, indices.numel())
    expected = torch.cat(
        [table[int(row), : int(length)] for row, length in zip(rows, lens)]
    )
    # Reserved request row zero legitimately maps graph padding to KV slot zero.
    # Every physical slot (padding included) must still fit the actual KV buffer.
    assert torch.all((expected >= 0) & (expected < capacity)), (
        int(expected.min()),
        int(expected.max()),
        capacity,
    )
    if compare:
        actual = indices[:total].detach().cpu()
        assert torch.equal(actual, expected), (
            "KV gather differs from live request table"
        )
    print(
        f"EAGLE-DIAG {phase} req_rows={rows.tolist()} seq_lens={lens.tolist()} "
        f"padding_rows={int((rows == 0).sum())} csr_total={total} "
        f"index_capacity={indices.numel()} kv_capacity={capacity} "
        f"slot_range=({int(expected.min())},{int(expected.max())}) exact_compare={compare}",
        flush=True,
    )


def selected_logits(runner, forward_batch, select_index, graph_bs):
    if not _active:
        return
    sync("draft_extend.before_selected_logits_graph")
    selected = select_index.detach().cpu().long()
    raw_bs = forward_batch.batch_size
    tokens = forward_batch.input_ids.numel()
    assert selected.numel() == raw_bs
    assert torch.all((selected >= 0) & (selected < tokens)), selected.tolist()
    assert tokens <= runner.buffers.input_ids.numel()
    assert graph_bs <= runner.buffers.next_token_logits_buffer.shape[0]
    assert raw_bs <= runner.buffers.select_index.numel()
    print(
        f"EAGLE-DIAG selected_logits raw_bs={raw_bs} graph_bs={graph_bs} "
        f"tokens={tokens} captured_width={runner.captured_req_width} "
        f"select_index={selected.tolist()} "
        f"logits_shape={tuple(runner.buffers.next_token_logits_buffer.shape)} "
        f"hidden_shape={tuple(forward_batch.spec_info.hidden_states.shape)}",
        flush=True,
    )
