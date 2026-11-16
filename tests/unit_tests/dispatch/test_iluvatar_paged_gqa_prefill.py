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


def test_q32_density_and_mixed_workspace(monkeypatch):
    for maximum, tokens, sequences, expected in (
        (2, 4096, 1, 2), (31, 2048, 1, 16), (32, 31, 1, 16),
        (32, 32, 1, 32), (33, 64, 2, 32), (33, 63, 2, 16),
        (4096, 2030, 64, 16), (32, 32, 0, 8),
    ):
        assert bridge.select_query_tile(maximum, tokens, sequences) == expected
    _, query, _, _, metadata = decode_objects()
    metadata.iluvatar_mixed_has_q1 = True
    monkeypatch.setattr(bridge, "is_batch_invariant", False)
    for tile in (16, 32):
        assert bridge._can_split_mixed_q1(query, metadata, tile, CUDAGraphMode.NONE)
        assert not bridge._can_split_mixed_q1(query, metadata, tile, CUDAGraphMode.FULL)
    assert not bridge._can_split_mixed_q1(query, metadata, 8, CUDAGraphMode.NONE)
    metadata.softmax_segm_output = None
    assert not bridge._can_split_mixed_q1(query, metadata, 32, CUDAGraphMode.NONE)




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


def slab_config():
    model = SimpleNamespace(
        dtype=torch.bfloat16, quantization=None, is_hybrid=False, use_mla=False,
        is_encoder_decoder=False, is_mm_prefix_lm=False, attention_chunk_size=None,
        enable_sleep_mode=False, get_sliding_window=lambda: None,
        get_num_attention_heads=lambda p: 16, get_num_kv_heads=lambda p: 2,
        get_head_size=lambda: 128,
    )
    parallel = SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1,
                               data_parallel_size=1, decode_context_parallel_size=1,
                               prefill_context_parallel_size=1)
    cache = SimpleNamespace(cache_dtype="auto", kv_sharing_fast_prefill=False,
                            kv_offloading_size=None)
    return SimpleNamespace(model_config=model, cache_config=cache, parallel_config=parallel,
                            speculative_config=None, kv_transfer_config=None)


def test_kv_slab_config_and_actual_spec_guards(monkeypatch):
    from dataclasses import replace
    from vllm.v1.attention.backends import triton_attn
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    config = slab_config()
    monkeypatch.setattr(bridge, "current_platform", SimpleNamespace(vendor_name="iluvatar"))
    monkeypatch.setattr(bridge, "get_kv_cache_layout", lambda: "NHD")
    monkeypatch.setattr(triton_attn, "get_kv_cache_layout", lambda: "NHD")
    monkeypatch.setattr(bridge, "get_current_vllm_config_or_none", lambda: config)
    backend = bridge.IluvatarPagedGQAPrefillBackend
    assert bridge._can_use_kv_slabs(config)
    assert backend.get_kv_cache_stride_order() == (1, 0, 2, 3, 4)
    assert backend.get_kv_cache_stride_order(True) == (0, 2, 1, 3, 4, 5)
    assert not backend.indexes_kv_by_block_stride()
    for obj, key, value in (
        (config, "kv_transfer_config", object()), (config, "speculative_config", object()),
        (config.cache_config, "kv_offloading_size", 1),
        (config.cache_config, "kv_sharing_fast_prefill", True),
        (config.model_config, "dtype", torch.float16),
        (config.model_config, "is_hybrid", True),
        (config.model_config, "enable_sleep_mode", True),
        (config.parallel_config, "decode_context_parallel_size", 2),
    ):
        old = getattr(obj, key)
        setattr(obj, key, value)
        assert not bridge._can_use_kv_slabs(config), key
        setattr(obj, key, old)
    spec = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=128,
                             dtype=torch.bfloat16, indexes_kv_by_block_stride=False)
    cache_config = SimpleNamespace(kv_cache_groups=[object()], needs_kv_cache_zeroing=False)
    def order(actual=spec, packing=None):
        return backend.get_kv_cache_stride_order_for_spec(actual, packing=packing,
            kernel_block_size=16, kv_cache_config=cache_config, vllm_config=config)
    original = bridge.TritonAttentionBackend.get_kv_cache_stride_order()
    assert order() == (1, 0, 2, 3, 4)
    assert order(packing=(0, spec.page_size_bytes)) == original
    for changes in ({'page_size_padded': spec.page_size_bytes + 128},
                    {'indexes_kv_by_block_stride': True}, {'block_size': 32},
                    {'dtype': torch.float16}, {'non_causal': True},
                    {'sliding_window': 128}, {'num_kv_heads': 4}):
        assert order(replace(spec, **changes)) == original
    cache_config.needs_kv_cache_zeroing = True
    assert order() == original
    cache_config.needs_kv_cache_zeroing = False
    cache_config.kv_cache_groups.append(object())
    assert order() == original
    monkeypatch.setattr(bridge, "current_platform", SimpleNamespace(vendor_name="metax"))
    assert not bridge._can_use_kv_slabs(config)
    assert backend.get_kv_cache_stride_order() == original


def test_kv_slab_allocation_writer_graph_and_native_fallback():
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.worker.gpu.attn_utils import _reshape_attention_kv_cache
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import triton_reshape_and_cache_flash
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    spec = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=128, dtype=torch.bfloat16)
    shape = (8, 2, 16, 2, 128)
    raw = torch.zeros(8 * spec.page_size_bytes, dtype=torch.uint8, device="cuda")
    cache = _reshape_attention_kv_cache(raw, spec, shape, (1, 0, 2, 3, 4), 8, None)
    control = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
    assert cache.data_ptr() == raw.data_ptr() and cache.numel() == control.numel()
    k, v = cache.unbind(1)
    assert k.stride(0) == 16 * k.stride(1) and v.stride() == k.stride()
    keys = torch.randn((9, 2, 128), device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    slots = torch.tensor([0, 15, 16, 31, 32, 63, 64, 127, -1], device="cuda", dtype=torch.int64)
    scale = torch.ones((), device="cuda", dtype=torch.float32)
    def write(target):
        kc, vc = target.unbind(1)
        triton_reshape_and_cache_flash(keys, values, kc, vc, slots, "auto", scale, scale)
    write(control)
    write(cache)
    assert torch.equal(control, cache)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        write(cache)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        write(cache)
    keys.add_(.25)
    values.mul_(.5)
    write(control)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(control, cache)
    query = torch.randn((2, 16, 128), device="cuda", dtype=torch.bfloat16)
    starts = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([128], device="cuda", dtype=torch.int32)
    table = torch.arange(8, device="cuda", dtype=torch.int32).view(1, 8)
    outputs = []
    for target in (control, cache):
        kc, vc = target.unbind(1)
        out = torch.empty_like(query)
        unified_attention(q=query, k=kc, v=vc, out=out, cu_seqlens_q=starts,
            max_seqlen_q=2, seqused_k=lengths, max_seqlen_k=128,
            softmax_scale=128 ** -.5, causal=True, window_size=(-1, -1),
            block_table=table, softcap=0, q_descale=None, k_descale=None, v_descale=None)
        outputs.append(out)
    torch.testing.assert_close(outputs[0], outputs[1], atol=.02, rtol=.02)
    assert torch.equal(control, cache)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "--confcutdir=" + __file__.rsplit("/", 1)[0], "-q"]))
