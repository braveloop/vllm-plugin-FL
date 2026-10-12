# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Guard, registration ownership, output contract and native fallback tests."""
from types import SimpleNamespace

import pytest
import torch

if __name__ == "__main__":
    torch.cuda.init()

import flag_gems
from vllm.config import CUDAGraphMode
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
from vllm.v1.kv_cache_interface import KVQuantMode
from vllm_fl.dispatch.backends.vendor.iluvatar import paged_gqa_prefill as bridge

pytestmark = pytest.mark.skipif(flag_gems.vendor_name != "iluvatar", reason="target-card adapter tests")


def objects():
    impl = object.__new__(bridge.IluvatarPagedGQAPrefillImpl)
    for key, value in dict(num_heads=16, num_kv_heads=2, head_size=128,
                           attn_type=AttentionType.DECODER, _kv_quant_mode=KVQuantMode.NONE,
                           alibi_slopes=None, sinks=None, logits_soft_cap=0,
                           sliding_window=(-1, -1), chunk_lookback=-1,
                           use_td=False, use_alibi_sqrt=False,
                           kv_sharing_target_layer_name=None, scale=128 ** -.5).items():
        setattr(impl, key, value)
    query = torch.zeros((11, 20, 128), device="cuda", dtype=torch.bfloat16)[:, :16]
    output = torch.full((11, 16, 128), 137, device="cuda", dtype=torch.bfloat16)
    cache = torch.zeros((3, 2, 16, 2, 128), device="cuda", dtype=torch.bfloat16)
    metadata = SimpleNamespace(max_query_len=8, num_actual_tokens=8, causal=True,
                               use_cascade=False, mm_prefix_range=None,
                               mm_prefix_range_tensor=None,
                               query_start_loc=torch.tensor([0, 8], device="cuda", dtype=torch.int32),
                               seq_lens=torch.tensor([9], device="cuda", dtype=torch.int64),
                               block_table=torch.tensor([[2]], device="cuda", dtype=torch.int32))
    return impl, query, cache, output, metadata


def test_guards(monkeypatch):
    impl, q, cache, out, metadata = objects()
    monkeypatch.setattr(bridge, "current_platform", SimpleNamespace(vendor_name="iluvatar"))
    guard = lambda **kwargs: bridge.can_use_paged_gqa_prefill(impl, q, cache, out, metadata, **kwargs)
    assert guard()
    assert not guard(full_graph=True)
    assert not guard(output_scale=torch.ones(1, device="cuda"))
    for name, bad in (("num_heads", 32), ("head_size", 64), ("attn_type", AttentionType.ENCODER),
                      ("_kv_quant_mode", KVQuantMode.FP8_PER_TENSOR), ("alibi_slopes", object()),
                      ("sinks", object()), ("logits_soft_cap", 1), ("sliding_window", (31, 0)),
                      ("chunk_lookback", 0), ("use_td", True), ("use_alibi_sqrt", True),
                      ("kv_sharing_target_layer_name", "layer")):
        previous = getattr(impl, name)
        setattr(impl, name, bad)
        assert not guard(), name
        setattr(impl, name, previous)
    for name, bad in (("max_query_len", 1), ("causal", False), ("use_cascade", True),
                      ("mm_prefix_range", {}), ("mm_prefix_range_tensor", object())):
        previous = getattr(metadata, name)
        setattr(metadata, name, bad)
        assert not guard(), name
        setattr(metadata, name, previous)
    for vendor in ("metax", "unknown", None):
        monkeypatch.setattr(bridge, "current_platform", SimpleNamespace(vendor_name=vendor))
        assert not guard()
    monkeypatch.setattr(bridge, "current_platform", SimpleNamespace(vendor_name="iluvatar"))
    assert not bridge.can_use_paged_gqa_prefill(impl, q.float(), cache, out, metadata)
    assert not bridge.can_use_paged_gqa_prefill(impl, q, cache, out.transpose(1, 2), metadata)


def test_registration_preserves_foreign_override(monkeypatch):
    overrides, calls = {}, []
    monkeypatch.setattr(bridge, "_ATTN_OVERRIDES", overrides)
    monkeypatch.setattr(bridge, "current_platform", SimpleNamespace(vendor_name="iluvatar"))
    def register(kind, path):
        calls.append((kind, path))
        overrides[kind] = path
    monkeypatch.setattr(bridge, "register_backend", register)
    bridge.register_paged_gqa_prefill_backend()
    bridge.register_paged_gqa_prefill_backend()
    assert len(calls) == 1
    overrides[bridge.AttentionBackendEnum.TRITON_ATTN] = "third.party.Backend"
    bridge.register_paged_gqa_prefill_backend()
    assert len(calls) == 1 and overrides[bridge.AttentionBackendEnum.TRITON_ATTN] == "third.party.Backend"
    for vendor in ("metax", "unknown"):
        overrides.clear()
        monkeypatch.setattr(bridge, "current_platform", SimpleNamespace(vendor_name=vendor))
        bridge.register_paged_gqa_prefill_backend()
        assert not overrides
    assert bridge.IluvatarPagedGQAPrefillBackend.get_name() == "TRITON_ATTN"
    assert bridge.IluvatarPagedGQAPrefillBackend.get_impl_cls() is bridge.IluvatarPagedGQAPrefillImpl
    assert bridge.IluvatarPagedGQAPrefillImpl.do_kv_cache_update is TritonAttentionImpl.do_kv_cache_update


def test_forward_alias_fallback_and_error(monkeypatch):
    from flag_gems.runtime.backend._iluvatar.fused import paged_gqa_prefill as kernel

    impl, query, cache, output, metadata = objects()
    monkeypatch.setattr(bridge, "current_platform", SimpleNamespace(vendor_name="iluvatar"))
    monkeypatch.setattr(bridge, "is_forward_context_available", lambda: False)
    calls = []
    tiles = []
    def own(q, k, v, out, *args, query_tile=8):
        calls.append("own")
        tiles.append(query_tile)
        out.fill_(2)
        return out
    def fallback(self, *args, **kwargs):
        calls.append("native")
        return args[6]
    monkeypatch.setattr(kernel, "paged_gqa_prefill", own)
    monkeypatch.setattr(TritonAttentionImpl, "forward", fallback)
    returned = impl.forward(None, query, None, None, cache, metadata, output)
    assert returned is output and calls == ["own"]
    assert tiles == [8]
    assert torch.all(output[:8] == 2) and torch.all(output[8:] == 137)
    assert not query.any() and not cache.any()
    for maximum, tile in ((2, 2), (3, 8), (4, 8), (5, 8), (128, 8)):
        metadata.max_query_len = maximum
        impl.forward(None, query, None, None, cache, metadata, output)
        assert tiles[-1] == tile
    metadata.max_query_len = 1
    assert impl.forward(None, query, None, None, cache, metadata, output) is output
    assert calls[-1] == "native"
    metadata.max_query_len = 8
    monkeypatch.setattr(bridge, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(bridge, "get_forward_context", lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL))
    impl.forward(None, query, None, None, cache, metadata, output)
    assert calls[-1] == "native"
    monkeypatch.setattr(bridge, "is_forward_context_available", lambda: False)
    def error(*args, **kwargs):
        raise RuntimeError("own kernel failure must not silently fallback")
    monkeypatch.setattr(kernel, "paged_gqa_prefill", error)
    with pytest.raises(RuntimeError, match="must not silently"):
        impl.forward(None, query, None, None, cache, metadata, output)


def decode_objects():
    impl, query, cache, output, metadata = objects()
    metadata.max_query_len = 1
    metadata.query_start_loc = torch.arange(9, device="cuda", dtype=torch.int32)
    metadata.seq_lens = torch.full((8,), 9, device="cuda", dtype=torch.int32)
    metadata.block_table = torch.full((8, 1), 2, device="cuda", dtype=torch.int32)
    metadata.seq_threshold_3D = 64
    metadata.num_par_softmax_segments = 16
    metadata.softmax_segm_output = torch.empty((64, 16, 16, 128), device="cuda", dtype=torch.float32)
    metadata.softmax_segm_max = torch.empty((64, 16, 16), device="cuda", dtype=torch.float32)
    metadata.softmax_segm_expsum = torch.empty((64, 16, 16), device="cuda", dtype=torch.float32)
    return impl, query, cache, output, metadata


def test_decode_guards(monkeypatch):
    impl, q, cache, out, metadata = decode_objects()
    monkeypatch.setattr(bridge, "current_platform", SimpleNamespace(vendor_name="iluvatar"))
    monkeypatch.setattr(bridge, "is_batch_invariant", False)
    guard = lambda **kwargs: bridge.can_use_paged_gqa_decode(impl, q, cache, out, metadata, **kwargs)
    assert guard() and not bridge.can_use_paged_gqa_prefill(impl, q, cache, out, metadata)
    monkeypatch.setattr(bridge, "is_batch_invariant", True)
    assert not guard()
    monkeypatch.setattr(bridge, "is_batch_invariant", False)
    for name, bad in (("seq_threshold_3D", None), ("seq_threshold_3D", 7),
                      ("num_par_softmax_segments", 8), ("max_query_len", 2),
                      ("use_cascade", True), ("causal", False)):
        previous = getattr(metadata, name)
        setattr(metadata, name, bad)
        assert not guard(), name
        setattr(metadata, name, previous)
    for name in ("softmax_segm_output", "softmax_segm_max", "softmax_segm_expsum"):
        previous = getattr(metadata, name)
        for bad in (None, previous.to(torch.bfloat16), previous[:7], previous.transpose(1, 2), previous.cpu()):
            setattr(metadata, name, bad)
            assert not guard(), name
        setattr(metadata, name, previous)
    assert not guard(output_block_scale=torch.ones(1, device="cuda"))
    for name, bad in (("alibi_slopes", object()), ("sinks", object()), ("use_td", True),
                      ("sliding_window", (31, 0)), ("_kv_quant_mode", KVQuantMode.FP8_PER_TENSOR)):
        previous = getattr(impl, name)
        setattr(impl, name, bad)
        assert not guard(), name
        setattr(impl, name, previous)
    for vendor in ("metax", "unknown", None):
        monkeypatch.setattr(bridge, "current_platform", SimpleNamespace(vendor_name=vendor))
        assert not guard()


def test_decode_full_forward_contract(monkeypatch):
    from flag_gems.runtime.backend._iluvatar.fused import paged_gqa_decode as kernel

    impl, query, cache, output, metadata = decode_objects()
    monkeypatch.setattr(bridge, "current_platform", SimpleNamespace(vendor_name="iluvatar"))
    monkeypatch.setattr(bridge, "is_batch_invariant", False)
    monkeypatch.setattr(bridge, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(bridge, "get_forward_context", lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL))
    calls = []
    def own(q, k, v, out, starts, lengths, table, scale, partial, maximum, expsum):
        calls.append("own_decode")
        assert partial is metadata.softmax_segm_output
        assert maximum is metadata.softmax_segm_max and expsum is metadata.softmax_segm_expsum
        assert q.shape[0] == metadata.num_actual_tokens
        out.fill_(3)
        return out
    def fallback(self, *args, **kwargs):
        calls.append("native")
        return args[6]
    monkeypatch.setattr(kernel, "paged_gqa_decode", own)
    monkeypatch.setattr(TritonAttentionImpl, "forward", fallback)
    assert impl.forward(None, query, None, None, cache, metadata, output) is output
    assert calls == ["own_decode"]
    assert torch.all(output[:8] == 3) and torch.all(output[8:] == 137)
    assert not query.any() and not cache.any()
    metadata.softmax_segm_max = None
    assert impl.forward(None, query, None, None, cache, metadata, output) is output
    assert calls[-1] == "native"
    metadata.softmax_segm_max = torch.empty((64, 16, 16), device="cuda")
    def error(*args, **kwargs):
        raise RuntimeError("own Decode failure must not silently fallback")
    monkeypatch.setattr(kernel, "paged_gqa_decode", error)
    with pytest.raises(RuntimeError, match="must not silently"):
        impl.forward(None, query, None, None, cache, metadata, output)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "--confcutdir=" + __file__.rsplit("/", 1)[0], "-q"]))
