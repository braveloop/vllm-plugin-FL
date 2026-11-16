# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Restricted own Prefill implementation, preserving native Triton fallbacks."""

import logging

import torch
from vllm.config import CUDAGraphMode, get_current_vllm_config_or_none
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum, _ATTN_OVERRIDES, register_backend,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend, TritonAttentionImpl, TritonAttentionMetadataBuilder,
)
from vllm.v1.attention.ops.triton_unified_attention import is_batch_invariant
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode

logger = logging.getLogger(__name__)
BACKEND_PATH = __name__ + ".IluvatarPagedGQAPrefillBackend"
_reported_external_override = False
_reported_kv_slabs_layout = False


def _can_use_kv_slabs(config):
    """Keep block-major storage for features that copy/pack/zero whole pages."""
    if config is None or getattr(current_platform, "vendor_name", None) != "iluvatar":
        return False
    model, cache, parallel = config.model_config, config.cache_config, config.parallel_config
    if model is None or get_kv_cache_layout() != "NHD":
        return False
    if (model.dtype != torch.bfloat16 or cache.cache_dtype not in ("auto", "bfloat16")
            or model.quantization is not None or model.is_hybrid or model.use_mla
            or model.is_encoder_decoder or model.is_mm_prefix_lm
            or model.get_sliding_window() is not None or model.attention_chunk_size is not None
            or model.enable_sleep_mode or cache.kv_sharing_fast_prefill
            or config.kv_transfer_config is not None or cache.kv_offloading_size is not None
            or config.speculative_config is not None):
        return False
    if any(getattr(parallel, name) != 1 for name in (
            "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
            "decode_context_parallel_size", "prefill_context_parallel_size")):
        return False
    return (model.get_num_attention_heads(parallel), model.get_num_kv_heads(parallel),
            model.get_head_size()) == (16, 2, 128)


def select_query_tile(max_query_len, tokens, sequences):
    if max_query_len == 2:
        return 2
    if max_query_len >= 32 and sequences > 0 and tokens >= 32 * sequences:
        return 32
    if max_query_len >= 16 and sequences > 0 and tokens >= 16 * sequences:
        return 16
    return 8


def _can_use_paged_gqa(impl, query, kv_cache, output, metadata,
                       output_scale=None, output_block_scale=None):
    if metadata is None:
        return False
    if getattr(current_platform, "vendor_name", None) != "iluvatar":
        return False
    if (impl.num_heads, impl.num_kv_heads, impl.head_size) != (16, 2, 128):
        return False
    if (impl.attn_type != AttentionType.DECODER or impl._kv_quant_mode != KVQuantMode.NONE
            or impl.alibi_slopes is not None or impl.sinks is not None
            or impl.logits_soft_cap != 0 or impl.sliding_window != (-1, -1)
            or impl.chunk_lookback != -1 or impl.use_td or impl.use_alibi_sqrt
            or impl.kv_sharing_target_layer_name is not None):
        return False
    if (metadata.causal is not True or metadata.use_cascade
            or metadata.mm_prefix_range_tensor is not None
            or metadata.mm_prefix_range is not None
            or output_scale is not None or output_block_scale is not None):
        return False
    count = metadata.num_actual_tokens
    if type(count) is not int or count <= 0:
        return False
    for tensor in (query, output):
        if (tensor.ndim != 3 or tensor.shape[1:] != (16, 128)
                or tensor.shape[0] < count or tensor.dtype != torch.bfloat16
                or tensor.device.type != "cuda" or tensor.stride(2) != 1
                or tensor.stride(1) != 128 or tensor.stride(0) < 2048):
            return False
    if output.device != query.device:
        return False
    if (kv_cache.ndim != 5 or kv_cache.shape[1] != 2
            or kv_cache.shape[3:] != (2, 128) or kv_cache.shape[0] <= 0
            or kv_cache.shape[2] <= 0 or kv_cache.shape[2] % 16
            or kv_cache.dtype != torch.bfloat16 or kv_cache.device != query.device
            or kv_cache.stride(4) != 1 or any(s <= 0 for s in kv_cache.stride())):
        return False
    starts, lengths, table = metadata.query_start_loc, metadata.seq_lens, metadata.block_table
    if (lengths.ndim != 1 or starts.ndim != 1 or table.ndim != 2
            or lengths.numel() <= 0 or starts.numel() != lengths.numel() + 1
            or table.shape[0] < lengths.numel() or table.shape[1] <= 0
            or starts.stride(0) != 1 or lengths.stride(0) != 1
            or table.stride(1) != 1 or table.stride(0) <= 0):
        return False
    return all(t.device == query.device and t.dtype in (torch.int32, torch.int64)
               for t in (starts, lengths, table))


def can_use_paged_gqa_prefill(impl, query, kv_cache, output, metadata,
                            output_scale=None, output_block_scale=None,
                            full_graph=False):
    return (metadata is not None and metadata.max_query_len > 1 and not full_graph
            and _can_use_paged_gqa(impl, query, kv_cache, output, metadata,
                                  output_scale, output_block_scale))


def _has_decode_workspace(query, metadata):
    sequences = metadata.seq_lens.numel()
    threshold = getattr(metadata, "seq_threshold_3D", None)
    if (type(threshold) is not int or sequences > threshold
            or getattr(metadata, "num_par_softmax_segments", None) != 16):
        return False
    for name, tail, strides in (
        ("softmax_segm_output", (16, 16, 128), (32768, 2048, 128, 1)),
        ("softmax_segm_max", (16, 16), (256, 16, 1)),
        ("softmax_segm_expsum", (16, 16), (256, 16, 1)),
    ):
        tensor = getattr(metadata, name, None)
        if (not isinstance(tensor, torch.Tensor) or tensor.ndim != len(tail) + 1
                or tensor.shape[0] < sequences or tensor.shape[1:] != tail
                or tensor.dtype != torch.float32 or tensor.device != query.device
                or tensor.stride() != strides):
            return False
    return True


def can_use_paged_gqa_decode(impl, query, kv_cache, output, metadata,
                           output_scale=None, output_block_scale=None):
    if metadata is None or metadata.max_query_len != 1 or is_batch_invariant:
        return False
    return (_can_use_paged_gqa(impl, query, kv_cache, output, metadata,
                             output_scale, output_block_scale)
            and _has_decode_workspace(query, metadata))


def _can_split_mixed_q1(query, metadata, query_tile, cudagraph_mode):
    """Called only after the unchanged Prefill feature/layout compatibility gate."""
    return (query_tile in (16, 32) and cudagraph_mode == CUDAGraphMode.NONE
            and not is_batch_invariant
            and getattr(metadata, "iluvatar_mixed_has_q1", False) is True
            and _has_decode_workspace(query, metadata))


class IluvatarPagedGQAMetadataBuilder(TritonAttentionMetadataBuilder):
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        metadata.iluvatar_mixed_has_q1 = False
        sequences = metadata.seq_lens.numel()
        if (self.vllm_config.speculative_config is None
                and select_query_tile(metadata.max_query_len,
                                      metadata.num_actual_tokens, sequences) in (16, 32)):
            starts_cpu = getattr(common_attn_metadata, "query_start_loc_cpu", None)
            if (isinstance(starts_cpu, torch.Tensor) and starts_cpu.device.type == "cpu"
                    and starts_cpu.ndim == 1 and starts_cpu.numel() >= sequences + 1
                    and starts_cpu.dtype in (torch.int32, torch.int64)):
                metadata.iluvatar_mixed_has_q1 = bool(
                    (starts_cpu[:sequences + 1].diff() == 1).any())
        return metadata


class IluvatarPagedGQAPrefillImpl(TritonAttentionImpl):
    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        if (attn_metadata is not None and attn_metadata.max_query_len == 1
                and can_use_paged_gqa_decode(self, query, kv_cache, output, attn_metadata,
                                            output_scale, output_block_scale)):
            from flag_gems.runtime.backend._iluvatar.fused.paged_gqa_decode import paged_gqa_decode

            key_cache, value_cache = kv_cache.unbind(1)
            count = attn_metadata.num_actual_tokens
            paged_gqa_decode(query[:count], key_cache, value_cache, output[:count],
                             attn_metadata.query_start_loc, attn_metadata.seq_lens,
                             attn_metadata.block_table, self.scale,
                             attn_metadata.softmax_segm_output,
                             attn_metadata.softmax_segm_max,
                             attn_metadata.softmax_segm_expsum)
            return output
        # Unsupported Decode/capture features keep their original implementation.
        if attn_metadata is None or attn_metadata.max_query_len <= 1:
            return super().forward(layer, query, key, value, kv_cache, attn_metadata,
                                   output, output_scale, output_block_scale)
        cudagraph_mode = (get_forward_context().cudagraph_runtime_mode
                          if is_forward_context_available() else None)
        full_graph = cudagraph_mode == CUDAGraphMode.FULL
        if not can_use_paged_gqa_prefill(self, query, kv_cache, output, attn_metadata,
                                       output_scale, output_block_scale, full_graph):
            return super().forward(layer, query, key, value, kv_cache, attn_metadata,
                                   output, output_scale, output_block_scale)
        from flag_gems.runtime.backend._iluvatar.fused.paged_gqa_prefill import paged_gqa_prefill

        key_cache, value_cache = kv_cache.unbind(1)
        count = attn_metadata.num_actual_tokens
        query_tile = select_query_tile(attn_metadata.max_query_len, count,
                                       attn_metadata.seq_lens.numel())
        if _can_split_mixed_q1(query, attn_metadata, query_tile, cudagraph_mode):
            from flag_gems.runtime.backend._iluvatar.fused.paged_gqa_decode import paged_gqa_decode

            paged_gqa_decode(query[:count], key_cache, value_cache, output[:count],
                             attn_metadata.query_start_loc, attn_metadata.seq_lens,
                             attn_metadata.block_table, self.scale,
                             attn_metadata.softmax_segm_output,
                             attn_metadata.softmax_segm_max,
                             attn_metadata.softmax_segm_expsum)
            paged_gqa_prefill(query[:count], key_cache, value_cache, output[:count],
                             attn_metadata.query_start_loc, attn_metadata.seq_lens,
                             attn_metadata.block_table, self.scale,
                             query_tile=query_tile, skip_q1=True)
            return output
        paged_gqa_prefill(query[:count], key_cache, value_cache, output[:count],
                         attn_metadata.query_start_loc, attn_metadata.seq_lens,
                         attn_metadata.block_table, self.scale,
                         query_tile=query_tile)
        return output


class IluvatarPagedGQAPrefillBackend(TritonAttentionBackend):
    @staticmethod
    def get_kv_cache_stride_order(include_num_layers_dimension=False):
        if _can_use_kv_slabs(get_current_vllm_config_or_none()):
            # Logical [block, K/V, token, head, dim] remains unchanged.
            return ((0, 2, 1, 3, 4, 5) if include_num_layers_dimension
                    else (1, 0, 2, 3, 4))
        return TritonAttentionBackend.get_kv_cache_stride_order(include_num_layers_dimension)

    @classmethod
    def indexes_kv_by_block_stride(cls):
        if _can_use_kv_slabs(get_current_vllm_config_or_none()):
            return False
        return TritonAttentionBackend.indexes_kv_by_block_stride()

    @classmethod
    def get_kv_cache_stride_order_for_spec(
        cls, spec, *, packing, kernel_block_size, kv_cache_config, vllm_config,
    ):
        global _reported_kv_slabs_layout
        # The actual spec can still require padding/packing after backend selection.
        eligible = (
            _can_use_kv_slabs(vllm_config)
            and type(spec) is FullAttentionSpec
            and spec.dtype == torch.bfloat16 and spec.kv_quant_mode == KVQuantMode.NONE
            and spec.num_kv_heads == 2 and spec.head_size == 128
            and spec.head_size_v in (None, 128)
            and spec.sliding_window is None and spec.attention_chunk_size is None
            and not spec.non_causal and not spec.indexes_kv_by_block_stride
            and packing is None and spec.page_size_padded is None
            and spec.storage_block_size == spec.block_size == kernel_block_size
            and kernel_block_size in (16, 32, 64)
            and len(kv_cache_config.kv_cache_groups) == 1
            and not kv_cache_config.needs_kv_cache_zeroing
        )
        if eligible:
            if not _reported_kv_slabs_layout:
                logger.info("Using separate BF16 K/V slabs for iluvatar Decode; page=%d, Prefill/native fallbacks retained", kernel_block_size)
                _reported_kv_slabs_layout = True
            return (1, 0, 2, 3, 4)
        return TritonAttentionBackend.get_kv_cache_stride_order()

    @staticmethod
    def get_builder_cls():
        return IluvatarPagedGQAMetadataBuilder

    @staticmethod
    def get_impl_cls():
        return IluvatarPagedGQAPrefillImpl


def register_paged_gqa_prefill_backend():
    global _reported_external_override
    if getattr(current_platform, "vendor_name", None) != "iluvatar":
        return
    existing = _ATTN_OVERRIDES.get(AttentionBackendEnum.TRITON_ATTN)
    if existing == BACKEND_PATH:
        return
    if existing is not None:
        if not _reported_external_override:
            logger.info("Keeping existing TRITON_ATTN override %s; own Prefill not registered", existing)
            _reported_external_override = True
        return
    register_backend(AttentionBackendEnum.TRITON_ATTN, BACKEND_PATH)
    logger.info("Registered own iluvatar paged-GQA Prefill/Decode with native Triton fallbacks")
