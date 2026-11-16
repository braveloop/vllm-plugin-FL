# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""BI SmoothQuant W8A8 candidate integration for calibrated QKV/gate-up.

The recipe, module allowlist and native kernel ABI identify the actual model.
No CLI quantization switch, model-directory mutation, or BF16 mirror is used.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
from pathlib import Path
import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.utils.torch_utils import direct_register_custom_op

from .iluvatar_smooth import ABI, apply_smooth

logger = init_logger(__name__)
NAME = "bi_smooth_native_w8a8_v2"
VERSION = 2
RECIPE_PATH = Path(__file__).with_name("smooth_recipe.pt")
ROLE_SHAPES = {"qkv_proj": (2560, 2048), "gate_up_proj": (12288, 2048)}


def _linear_impl(x: torch.Tensor, weight: torch.Tensor,
                 scale: torch.Tensor, inverse_s: torch.Tensor) -> torch.Tensor:
    return apply_smooth(x, weight, scale, inverse_s)


def _linear_fake(x: torch.Tensor, weight: torch.Tensor,
                 scale: torch.Tensor, inverse_s: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


direct_register_custom_op(
    op_name="bi_smooth_native_w8a8_v2", op_func=_linear_impl,
    mutates_args=[], fake_impl=_linear_fake,
)


class BISmoothW8A8Method(UnquantizedLinearMethod):
    """Keep the ordinary shard loader, then fuse load-time weight conversion."""
    def __init__(self, module_name, record):
        self.module_name = module_name
        self.record = record

    def process_weights_after_loading(self, layer):
        from flag_gems.runtime.backend._iluvatar.ops.smooth_w8a8 import quantize_weight_out
        weight = layer.weight
        role = self.module_name.rsplit(".", 1)[-1]
        if (weight.dtype != torch.bfloat16 or weight.device.type != "cuda"
                or tuple(weight.shape) != ROLE_SHAPES[role]
                or weight.stride() != (weight.shape[1], 1)
                or getattr(layer, "bias", None) is not None):
            raise ValueError("calibrated module no longer matches the BF16 weight contract")
        # Validate the original checkpoint bytes once at load time, never per token.
        digest = hashlib.sha256(weight.detach().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
        if digest != self.record["weight_sha256"]:
            raise ValueError("recipe/checkpoint weight identity mismatch")
        smooth = self.record["smooth"].to(device=weight.device)
        inverse = self.record["inverse_s"].to(device=weight.device)
        wq = torch.empty_like(weight, dtype=torch.int8)
        scale = torch.empty((weight.shape[0], 1), dtype=torch.float32, device=weight.device)
        quantize_weight_out(weight, smooth, out=wq, scale=scale)
        layer.weight = torch.nn.Parameter(wq, requires_grad=False)
        layer.register_parameter("weight_scale", torch.nn.Parameter(scale, requires_grad=False))
        layer.register_parameter("input_inverse_s", torch.nn.Parameter(inverse, requires_grad=False))

    def apply(self, layer, x, bias=None):
        if bias is not None:
            raise ValueError("biased Linear is outside the calibrated candidate")
        return torch.ops.vllm.bi_smooth_native_w8a8_v2(
            x, layer.weight, layer.weight_scale, layer.input_inverse_s)


@register_quantization_config(NAME)
class BISmoothW8A8Config(QuantizationConfig):
    def __init__(self):
        super().__init__()
        self.version = VERSION
        self.abi = ABI
        self.output_dtype = torch.bfloat16
        self.kv_quantization = False
        self.recipe_sha256 = hashlib.sha256(RECIPE_PATH.read_bytes()).hexdigest()
        from flag_gems.runtime.backend._iluvatar.ops import smooth_w8a8
        self.kernel_sha256 = hashlib.sha256(Path(smooth_w8a8.__file__).read_bytes()).hexdigest()
        self.vendor_version = importlib.metadata.version("ixformer")
        from ixformer.inference.functions.w8a8 import w8a8 as vendor_implementation
        self.vendor_wrapper_sha256 = hashlib.sha256(
            Path(inspect.getfile(vendor_implementation)).read_bytes()).hexdigest()
        recipe = torch.load(RECIPE_PATH, map_location="cpu", weights_only=True)
        if recipe.get("abi") != ABI or recipe.get("alpha") != .5:
            raise ValueError("invalid SmoothQuant recipe ABI/alpha")
        self.records = {name: record for name, record in recipe["modules"].items() if record["enabled"]}
        if not self.records:
            raise ValueError("no modules passed independent activation screening")
        for name, record in self.records.items():
            role = name.rsplit(".", 1)[-1]
            if role not in ROLE_SHAPES:
                raise ValueError("module outside calibrated roles")
            for field in ("smooth", "inverse_s"):
                value = record[field]
                if (value.dtype != torch.float32 or value.shape != (ROLE_SHAPES[role][1],)
                        or not torch.isfinite(value).all() or not (value > 0).all()):
                    raise ValueError("invalid positive finite scaling metadata")
        self.roles = tuple(sorted(self.records))

    @classmethod
    def get_name(cls):
        return NAME

    def get_supported_act_dtypes(self):
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 0

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config):
        if config.get("version") != VERSION or config.get("abi") != ABI:
            raise ValueError("unsupported BI W8A8 ABI/version")
        result = cls()
        if config.get("recipe_sha256") != result.recipe_sha256 or config.get("roles") != list(result.roles):
            raise ValueError("recipe changed since model configuration was constructed")
        return result

    def description(self):
        return dict(quant_method=NAME, version=VERSION, abi=ABI,
                    roles=list(self.roles), recipe_sha256=self.recipe_sha256,
                    kernel_sha256=self.kernel_sha256,
                    alpha=.5, weight_scheme="symmetric_channel",
                    activation_scheme="inverse_s_dynamic_symmetric_token",
                    output_dtype="bfloat16", kv_quantization=False,
                    runtime_m=True, fused_weight_load=True,
                    compute_backend="ixformer.inference.functions.w8a8.w8a8/TN",
                    compute_shapes={"K":2048,"N":[2560,12288]},
                    vendor_version=self.vendor_version,
                    vendor_wrapper_sha256=self.vendor_wrapper_sha256,
                    fallback="original_same_abi_triton_w8a8")

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            if prefix in self.records:
                return BISmoothW8A8Method(prefix, self.records[prefix])
            return UnquantizedLinearMethod()
        # Embedding/head retain their original unquantized implementations.
        return None


def is_bf16_kv_compatible(config):
    quant = getattr(config, "quant_config", None)
    model = config.model_config
    return (type(quant) is BISmoothW8A8Config
            and quant.version == VERSION and quant.abi == ABI
            and quant.output_dtype == torch.bfloat16 and quant.kv_quantization is False
            and model.quantization == NAME
            and model.quantization_config == quant.description())


def maybe_enable(vllm_config, vendor_name):
    """Only verified dense single-rank BF16 checkpoint configurations opt in."""
    model = vllm_config.model_config
    cache = vllm_config.cache_config
    parallel = vllm_config.parallel_config
    if vendor_name != "iluvatar" or model is None or cache is None or not RECIPE_PATH.exists():
        return False
    if (model.quantization is not None or vllm_config.quant_config is not None
            or model.dtype != torch.bfloat16 or cache.cache_dtype not in ("auto", "bfloat16")
            or model.is_hybrid or model.use_mla or model.is_encoder_decoder
            or model.is_mm_prefix_lm or model.is_multimodal_model
            or model.get_sliding_window() is not None or model.attention_chunk_size is not None
            or model.enable_sleep_mode or vllm_config.lora_config is not None
            or vllm_config.speculative_config is not None
            or vllm_config.kv_transfer_config is not None
            or cache.kv_offloading_size is not None):
        return False
    if any(getattr(parallel, name) != 1 for name in (
            "tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size",
            "decode_context_parallel_size", "prefill_context_parallel_size")):
        return False
    offload = getattr(vllm_config, "offload_config", None)
    if offload is not None and (offload.uva.cpu_offload_gb != 0
                                or offload.prefetch.offload_group_size != 0):
        return False
    hf = model.hf_text_config
    if (getattr(hf, "model_type", None) != "llama"
            or (hf.num_hidden_layers, hf.hidden_size, hf.intermediate_size) != (42, 2048, 6144)
            or getattr(hf, "attention_bias", False) or getattr(hf, "mlp_bias", False)
            or getattr(hf, "tie_word_embeddings", True)
            or getattr(hf, "vocab_size", None) != 130560
            or getattr(hf, "quantization_config", None)
            or getattr(model.model_arch_config, "quantization_config", None)
            or getattr(model, "quantization_config", None)):
        return False
    quant = BISmoothW8A8Config()
    model.quantization = NAME
    # This declared dataclass field is included in ModelConfig.compute_hash;
    # therefore ABI/version/role allowlist also identify compiled graphs.
    model.quantization_config = quant.description()
    vllm_config.quant_config = quant
    logger.info("Experimental BI native W8A8 enabled: %s", quant.description())
    return True


