"""Bounded extent-first placement; cached-free eviction choice may change."""
from dataclasses import dataclass, fields
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry


@dataclass(frozen=True, kw_only=True)
class ExtentFirstFullAttentionSpec(FullAttentionSpec):
    max_blocks: int

    def __post_init__(self):
        super().__post_init__()
        if self.block_size != 16 or self.max_blocks <= 0:
            raise ValueError('four-page reservation requires page16 and positive capacity')

    @classmethod
    def merge(cls, specs):
        if not specs or any(type(s) is not cls for s in specs):
            raise ValueError('only identical custom spec types may merge')
        if len({s.max_blocks for s in specs}) != 1:
            raise ValueError('max_blocks must agree across layers')
        merged = FullAttentionSpec.merge(specs)
        return cls(**{f.name: getattr(merged, f.name) for f in fields(FullAttentionSpec)},
                   max_blocks=specs[0].max_blocks)


def target_blocks(num_tokens, max_blocks, prefix_or_held_blocks=0):
    """Round total capacity, never the number of new blocks."""
    required = max((num_tokens + 15) // 16, prefix_or_held_blocks)
    if num_tokens < 0 or required > max_blocks:
        raise ValueError('request capacity exceeds the spec/table bound')
    return min(4 * ((required + 3) // 4), max_blocks)


class ExtentFirstFullAttentionManager(FullAttentionManager):
    def __init__(self, kv_cache_spec, *args, extent_observer=None, **kwargs):
        if type(kv_cache_spec) is not ExtentFirstFullAttentionSpec:
            raise ValueError('custom manager requires its own spec')
        if kwargs.get('dcp_world_size', 1) != 1 or kwargs.get('pcp_world_size', 1) != 1:
            raise ValueError('context parallel configurations keep the original spec')
        super().__init__(kv_cache_spec, *args, **kwargs)
        self.extent_observer = extent_observer
        print(
            'BI_EXTENT_MANAGER_BOUND '
            f'manager={type(self).__module__}.{type(self).__qualname__} '
            f'spec={type(self.kv_cache_spec).__module__}.{type(self.kv_cache_spec).__qualname__} '
            f'block_size={self.block_size} pool_blocks={self.block_pool.num_gpu_blocks} '
            f'max_blocks={self.kv_cache_spec.max_blocks} anchors=8 extent=4',
            flush=True)

    def _capacity_tokens(self, request_id, num_tokens, new_computed_blocks=()):
        held_or_prefix = len(self.req_to_blocks.get(request_id, ())) + len(new_computed_blocks)
        return target_blocks(num_tokens, self.kv_cache_spec.max_blocks, held_or_prefix) * 16

    def get_num_blocks_to_allocate(self, request_id, num_tokens, new_computed_blocks,
                                   total_computed_tokens, num_tokens_main_model,
                                   apply_admission_cap=False):
        capacity = self._capacity_tokens(request_id, num_tokens, new_computed_blocks)
        return super().get_num_blocks_to_allocate(
            request_id, capacity, new_computed_blocks, total_computed_tokens,
            num_tokens_main_model, apply_admission_cap)

    def _new_quad(self):
        pool = self.block_pool
        queue = pool.free_block_queue
        count_before = pool.get_num_free_blocks()
        node = queue.fake_free_list_head.next_free_block
        anchors = []
        for _ in range(8):
            if node is queue.fake_free_list_tail:
                break
            anchors.append(node)
            node = node.next_free_block
        selected = None
        checked = 0
        seen = set()
        if pool.null_block.block_id == 0:
            for anchor in anchors:
                checked += 1
                extent_start = 1 + 4 * ((anchor.block_id - 1) // 4)
                if extent_start in seen:
                    continue
                seen.add(extent_start)
                if extent_start < 1 or extent_start + 3 >= pool.num_gpu_blocks:
                    continue
                blocks = pool.blocks[extent_start:extent_start + 4]
                if all(not block.is_null and block.ref_cnt == 0
                       and block.prev_free_block is not None
                       and block.next_free_block is not None
                       and block.prev_free_block.next_free_block is block
                       and block.next_free_block.prev_free_block is block
                       for block in blocks):
                    selected = blocks
                    break
        observed = selected if selected is not None else anchors[:4]
        event = None
        if self.extent_observer is not None:
            event = dict(anchors=[b.block_id for b in anchors],anchors_checked=checked,
                selected_extent=selected is not None,free_before=count_before,
                original=[dict(id=b.block_id,ref=b.ref_cnt,hash=None if b.block_hash is None else bytes(b.block_hash).hex(),
                    prev=b.prev_free_block.block_id,next=b.next_free_block.block_id) for b in observed])
        if selected is not None:
            for block in selected:
                queue.remove(block)
            queue.prepend_n(selected)
        count_after_move = pool.get_num_free_blocks()
        assert count_after_move == count_before
        # No await or external callback between queue adjustment and this call.
        allocated = pool.get_new_blocks(4)
        if event is not None:
            event.update(free_after_move=count_after_move,allocated=[b.block_id for b in allocated],
                         refs_after=[b.ref_cnt for b in allocated],
                         hashes_after=[None if b.block_hash is None else bytes(b.block_hash).hex() for b in allocated])
        return allocated,event

    def allocate_new_blocks(self, request_id, num_tokens, num_tokens_main_model):
        table = self.req_to_blocks[request_id]
        capacity = self._capacity_tokens(request_id,num_tokens)
        needed = capacity // self.block_size - len(table)
        if needed <= 0:
            return []
        if needed > self.block_pool.get_num_free_blocks():
            raise ValueError('insufficient free blocks; caller admission must wait/preempt')
        new_blocks = []
        while len(new_blocks) < needed:
            logical = len(table)
            remaining = needed-len(new_blocks)
            if logical % 4 == 0 and remaining >= 4:
                allocated,event = self._new_quad()
            else:
                count = min(remaining,4-logical%4) if logical%4 else remaining
                allocated = self.block_pool.get_new_blocks(count)
                event = None
            table.extend(allocated)
            new_blocks.extend(allocated)
            self.new_block_ids.extend(block.block_id for block in allocated)
            if event is not None:
                event.update(request_id=request_id,logical_start=logical)
                self.extent_observer(event)
        return new_blocks

    def allocate_external_computed_blocks(self, request_id, num_local_computed_tokens,
                                          num_external_computed_tokens):
        # Connectors are excluded by the future activation guard. Preserve the
        # base method and its bookkeeping defensively, without remapping them.
        before = len(self.req_to_blocks.get(request_id, ()))
        super().allocate_external_computed_blocks(
            request_id, num_local_computed_tokens, num_external_computed_tokens)
        self.new_block_ids.extend(b.block_id for b in self.req_to_blocks[request_id][before:])


def register_extent_first_full_attention():
    """Add only the new type, after the normal built-in registration path."""
    KVCacheSpecRegistry.register(ExtentFirstFullAttentionSpec, ExtentFirstFullAttentionManager,
                                uniform_type_base_spec=FullAttentionSpec)


def maybe_use_extent_first_specs(specs, config, vendor):
    """All-or-nothing activation for the retained42-layer page16 BF16 GQA ABI."""
    if vendor != 'iluvatar':
        return specs
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.v1.kv_cache_interface import KVQuantMode
    from vllm_fl.dispatch.backends.vendor.iluvatar.paged_gqa_prefill import _can_use_kv_slabs

    cache = config.cache_config
    parallel = config.parallel_config
    model = config.model_config
    if (vendor != 'iluvatar' or len(specs) != 42 or model is None
            or getattr(config, 'kv_transfer_config', None) is not None
            or getattr(config, 'speculative_config', None) is not None
            or getattr(model, 'is_encoder_decoder', False)
            or getattr(model, 'use_mla', False)
            or cache.block_size != 16 or cache.cache_dtype != 'auto'
            or any(getattr(parallel, field, 1) != 1 for field in (
                'tensor_parallel_size', 'pipeline_parallel_size',
                'decode_context_parallel_size', 'prefill_context_parallel_size'))):
        return specs
    with set_current_vllm_config(config):
        if not _can_use_kv_slabs(config):
            return specs
    max_len = model.max_model_len
    if not isinstance(max_len, int) or max_len <= 0:
        return specs
    for spec in specs.values():
        if not (type(spec) is FullAttentionSpec and spec.block_size == 16
                and spec.dtype == torch.bfloat16 and spec.kv_quant_mode == KVQuantMode.NONE
                and spec.num_kv_heads == 2 and spec.head_size == 128
                and spec.head_size_v in (None, 128)
                and spec.sliding_window is None and spec.attention_chunk_size is None
                and not spec.non_causal and not spec.indexes_kv_by_block_stride
                and spec.page_size_padded is None):
            return specs
    converted = {
        name: ExtentFirstFullAttentionSpec(
            **{field.name: getattr(spec, field.name) for field in fields(FullAttentionSpec)},
            max_blocks=(max_len + 15) // 16)
        for name, spec in specs.items()
    }
    return converted
