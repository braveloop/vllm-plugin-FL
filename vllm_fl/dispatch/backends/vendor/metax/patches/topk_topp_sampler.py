# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# vLLM's Triton topk_topp kernel fails to compile on MetaX
# (PassManager::run failed in ttgir stage). Patch apply_top_k_top_p
# to always use the PyTorch fallback.
# TODO: remove once FlagGems provides a MetaX-compatible topk_topp kernel.

import torch
import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_sampler

from . import topp_prefix_filter

_USE_TOPP_PREFIX = True


def _apply_top_k_top_p_no_triton(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    if p is None and k is None:
        return logits
    if _USE_TOPP_PREFIX and topp_prefix_filter.eligible(logits, k, p):
        result = topp_prefix_filter.apply(logits, p)
        if result is not None:
            return result
    return topk_topp_sampler.apply_top_k_top_p_pytorch(logits, k, p)


# Replace the dispatch function with one that skips Triton
topk_topp_sampler.apply_top_k_top_p = _apply_top_k_top_p_no_triton
