"""Runtime-proved top-p prefix to original-order contiguous filtering."""

import importlib
import weakref

import torch
import triton
import triton.language as tl

_registrar_seen = None
_registrar_ok = False


def eligible(logits, k, p):
    if k is not None or not isinstance(p, torch.Tensor) or logits.ndim != 2:
        return False
    m, n = logits.shape
    if not (0 < m <= 65535 and 2048 <= n <= 262144 and m * n < 2**31):
        return False
    if (logits.dtype != torch.float32 or not logits.is_contiguous()
            or logits.device.type != "cuda" or p.dtype != torch.float32
            or p.device != logits.device or p.shape != (m,) or not p.is_contiguous()):
        return False
    return _stable_sort_registered()


def _stable_sort_registered():
    global _registrar_seen, _registrar_ok
    try:
        import flag_gems
        registrar = getattr(flag_gems, "current_work_registrar", None)
        if registrar is None or flag_gems.vendor_name != "metax":
            _registrar_seen, _registrar_ok = None, False
            return False
        if _registrar_seen is not None and _registrar_seen() is registrar:
            return _registrar_ok
        expected = importlib.import_module("flag_gems.ops.sort").sort
        ok = (registrar.reg_key == "CUDA"
              and registrar.get_current_device() == "cuda"
              and "sort" in registrar.get_all_keys()
              and "sort" in registrar.get_all_ops()
              and any(item[0] == "sort" and item[1] is expected
                      for item in registrar.config)
              and torch._C._dispatch_has_kernel_for_dispatch_key("aten::sort", "CUDA"))
        get_kernel = getattr(torch.library, "get_kernel", None)
        if ok and callable(get_kernel):
            actual = get_kernel("aten::sort", "CUDA")
            ok = actual is expected or getattr(actual, "fn", None) is expected
        _registrar_seen, _registrar_ok = weakref.ref(registrar), bool(ok)
        return _registrar_ok
    except (ImportError, AttributeError, RuntimeError, TypeError):
        _registrar_seen, _registrar_ok = None, False
        return False


@triton.jit
def _float_order_key(value):
    raw = value.to(tl.uint32, bitcast=True)
    signed = value.to(tl.int32, bitcast=True)
    return raw ^ (0x80000000 | (signed >> 31).to(tl.uint32, bitcast=True))


@triton.jit
def _mask_tiles(CDF, Threshold, Counts, Last, N: tl.constexpr,
                G: tl.constexpr, BLOCK: tl.constexpr):
    tile, row = tl.program_id(0), tl.program_id(1)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    offset = row.to(tl.int64) * N + col
    cdf = tl.load(CDF + offset, col < N, other=0.0)
    threshold = tl.load(Threshold + row)
    remove = (col < N - 1) & (cdf <= threshold)
    count = tl.sum(remove.to(tl.int32), 0)
    last = tl.max(tl.where(remove, col, -1), 0)
    position = row.to(tl.int64) * G + tile
    tl.store(Counts + position, count)
    tl.store(Last + position, last)


@triton.jit
def _row_boundary(Values, Indices, Counts, Last, Descriptor,
                  N: tl.constexpr, G: tl.constexpr, BLOCK_G: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.arange(0, BLOCK_G)
    offset = row.to(tl.int64) * G + tile
    count = tl.sum(tl.load(Counts + offset, tile < G, other=0), 0)
    last = tl.max(tl.load(Last + offset, tile < G, other=-1), 0)
    prefix = count == last + 1
    key = tl.full((), 0, tl.uint32)
    token_index = tl.full((), 0, tl.int64)
    if prefix & (count > 0):
        position = row.to(tl.int64) * N + last
        value = tl.load(Values + position)
        key = _float_order_key(value)
        token_index = tl.load(Indices + position)
    base = row.to(tl.int64) * 4
    tl.store(Descriptor + base, prefix.to(tl.int64))
    tl.store(Descriptor + base + 1, count.to(tl.int64))
    tl.store(Descriptor + base + 2, key.to(tl.int64))
    tl.store(Descriptor + base + 3, token_index)


@triton.jit
def _write_filter(Logits, Values, Indices, CDF, Threshold, Descriptor,
                  N: tl.constexpr, BLOCK: tl.constexpr):
    tile, row = tl.program_id(0), tl.program_id(1)
    col = tile * BLOCK + tl.arange(0, BLOCK)
    valid = col < N
    row_offset = row.to(tl.int64) * N
    base = row.to(tl.int64) * 4
    prefix = tl.load(Descriptor + base) != 0
    if prefix:
        count = tl.load(Descriptor + base + 1)
        value = tl.load(Logits + row_offset + col, valid, other=0.0)
        key = _float_order_key(value)
        cut_key = tl.load(Descriptor + base + 2).to(tl.uint32)
        cut_index = tl.load(Descriptor + base + 3)
        remove = (count > 0) & ((key < cut_key) | ((key == cut_key) & (col <= cut_index)))
        result = tl.where(remove, -float("inf"), value)
        tl.store(Logits + row_offset + col, result, valid)
    else:
        # Device-only row fallback; no partial direct write precedes it.
        value = tl.load(Values + row_offset + col, valid, other=0.0)
        index = tl.load(Indices + row_offset + col, valid, other=0)
        cdf = tl.load(CDF + row_offset + col, valid, other=0.0)
        threshold = tl.load(Threshold + row)
        remove = (col < N - 1) & (cdf <= threshold)
        result = tl.where(remove, -float("inf"), value)
        tl.store(Logits + row_offset + index, result, valid)


def filter_from_sorted(logits, values, indices, cdf, threshold):
    # Output may alias logits, but never the independent sorted/read-only state.
    output_storage = logits.untyped_storage().data_ptr()
    if any(t.untyped_storage().data_ptr() == output_storage
           for t in (values, indices, cdf, threshold)):
        return None
    if (any(t.shape != logits.shape or not t.is_contiguous()
            or t.device != logits.device for t in (values, indices, cdf))
            or values.dtype != torch.float32 or cdf.dtype != torch.float32
            or indices.dtype != torch.int64 or threshold.dtype != torch.float32
            or threshold.shape != (logits.shape[0], 1)
            or not threshold.is_contiguous() or threshold.device != logits.device):
        return None
    m, n = logits.shape
    g = triton.cdiv(n, 2048)
    counts = torch.empty((m, g), dtype=torch.int32, device=logits.device)
    last = torch.empty_like(counts)
    descriptor = torch.empty((m, 4), dtype=torch.int64, device=logits.device)
    _mask_tiles[(g, m)](cdf, threshold, counts, last, n, g, 2048, num_warps=4)
    _row_boundary[(m,)](values, indices, counts, last, descriptor,
                        n, g, triton.next_power_of_2(g), num_warps=4)
    _write_filter[(g, m)](logits, values, indices, cdf, threshold, descriptor,
                          n, 2048, num_warps=4)
    return logits


def apply(logits, p):
    # Preserve the original scalar/tensor math and operation order verbatim.
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
    threshold = 1 - p.unsqueeze(dim=1)
    return filter_from_sorted(logits, logits_sort, logits_idx, probs_sum, threshold)
