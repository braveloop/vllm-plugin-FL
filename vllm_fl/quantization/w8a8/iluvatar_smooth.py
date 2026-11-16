# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Unregistered SmoothQuant candidate helpers for actual activation calibration.

All scaling metadata belongs to the quantized model ABI. This file alone does
not alter production dispatch, model configuration, or service arguments.
"""
import torch

ABI = 'raw_i8_NK_f32_N1_inverse_s_f32_K_dynamic_a8_bf16_v1'


def quantize_smooth_weight(weight, activation_absmax):
    """Fixed alpha=0.5, using only the independent training calibration split."""
    if weight.ndim != 2 or weight.dtype != torch.bfloat16:
        raise ValueError('expected original BF16 weight[N,K]')
    if activation_absmax.shape != (weight.shape[1],):
        raise ValueError('activation absmax must contain exactly K channels')
    x_max = activation_absmax.to(device=weight.device, dtype=torch.float32)
    wf = weight.detach().float()
    w_max = wf.abs().amax(dim=0)
    if not (torch.isfinite(wf).all() and torch.isfinite(x_max).all()
            and (x_max >= 0).all()):
        raise ValueError('non-finite weights or invalid calibration statistics')
    # Dead channels use one. Clamping still preserves paired algebraic scaling.
    nonzero = (x_max > 0) & (w_max > 0)
    smooth = (x_max.clamp_min(1e-12) / w_max.clamp_min(1e-12)).sqrt()
    smooth = torch.where(nonzero, smooth.clamp(1e-5, 1e5), 1.0)
    inverse_s = smooth.reciprocal().contiguous()
    wf.mul_(smooth.unsqueeze(0))
    maximum = wf.abs().amax(dim=1, keepdim=True)
    # Independent direct division rounded once to FP32, matching tl.div_rn.
    # Tensor/scalar CPU division may instead multiply by a rounded reciprocal.
    scale = (maximum.double() / 127.0).float()
    inverse = torch.where(maximum > 0, (127.0 / maximum.double().clamp_min(1e-30)).float(), 0.0)
    weight_q = (wf * inverse).round().clamp(-128, 127).to(torch.int8)
    return weight_q, scale.contiguous(), inverse_s


def apply_smooth(x, weight_q, weight_scale, inverse_s):
    from flag_gems.runtime.backend._iluvatar.ops.smooth_w8a8 import smooth_w8a8_out
    if x.dtype != torch.bfloat16 or x.device.type != 'cuda':
        raise ValueError('BI candidate requires CUDA BF16 activations')
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    out = torch.empty((flat.shape[0], weight_q.shape[0]), device=x.device, dtype=x.dtype)
    smooth_w8a8_out(flat, weight_q, weight_scale, inverse_s, out=out)
    return out.reshape(*x.shape[:-1], weight_q.shape[0])
