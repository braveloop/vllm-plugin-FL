# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Independent checkpoint/prototype adapter. Not registered in any policy."""
import json
from pathlib import Path
import torch
from safetensors import safe_open
from .reference import dynamic_per_token_quant_int8


def load_gate_up(model_path, layer):
    root = Path(model_path)
    mapping = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    names = [f'model.layers.{layer}.mlp.{part}_proj.weight' for part in ('gate', 'up')]
    pieces = []
    for name in names:
        with safe_open(root / mapping[name], framework='pt', device='cpu') as handle:
            pieces.append(handle.get_tensor(name))
    weight = torch.cat(pieces, dim=0)
    assert weight.dtype == torch.bfloat16 and weight.shape == (12288, 2048)
    assert weight.stride() == (2048, 1)
    return weight, names


def quantize_weight(weight):
    # Per-token reference applied to rows is symmetric per-output-channel W8.
    # No group, physical transpose, repacking, MSE calibration, or model update.
    return dynamic_per_token_quant_int8(weight)


def apply_out(x, weight, weight_scale, *, out, **config):
    from flag_gems.runtime.backend._iluvatar.ops.w8a8_dynamic_prototype import dynamic_w8a8_out
    return dynamic_w8a8_out(x, weight, weight_scale, out=out, **config)
