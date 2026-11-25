# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM W8A8 GEMM for FlashHead transformer blocks."""

import torch
import torch.nn as nn


class FP8Linear(nn.Module):
    def __init__(self, linear):
        super().__init__()
        from vllm import _custom_ops as ops
        from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
        from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
            Fp8LinearOp,
            maybe_create_device_identity,
        )

        self.linear = linear
        self._ops = ops
        maybe_create_device_identity()
        self._fp8_linear_op = Fp8LinearOp(
            act_quant_static=False,
            act_quant_group_shape=GroupShape.PER_TOKEN,
            pad_output=False,
        )
        self.register_buffer("_fp8_weight", None, persistent=False)
        self.register_buffer("_fp8_weight_scale", None, persistent=False)
        if linear.weight.is_cuda:
            self._quantize_weight(linear.weight.device)

    def _quantize_weight(self, device):
        weight = self.linear.weight
        if (
            self._fp8_weight is not None
            and self._fp8_weight.device == device
        ):
            return
        weight = weight.detach().to(device=device, dtype=torch.bfloat16).contiguous()
        quantized, scale = self._ops.scaled_fp8_quant(weight, scale=None)
        # CUTLASS expects a transposed [K, N] view with stride(0) == 1.
        self._fp8_weight = quantized.t()
        self._fp8_weight_scale = scale

    def forward(self, x):
        # Keep the original BF16 path for small GEMMs, as in the old pipeline.
        if not x.is_cuda or x.numel() // x.shape[-1] < 256:
            return self.linear(x)
        dtype = x.dtype
        if dtype not in (torch.float16, torch.bfloat16):
            x = x.to(torch.bfloat16)
        self._quantize_weight(x.device)
        bias = self.linear.bias
        if bias is not None:
            bias = bias.to(device=x.device, dtype=x.dtype)
        output = self._fp8_linear_op.apply(
            input=x,
            weight=self._fp8_weight,
            weight_scale=self._fp8_weight_scale,
            out_dtype=x.dtype,
            input_scale=None,
            bias=bias,
        )
        return output.to(dtype)


def enable_fp8_gemm(model):
    """Replace Linear layers inside transformer blocks, keeping BF16 weights."""
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and "blocks." in name:
            parent, child = name.rsplit(".", 1)
            setattr(model.get_submodule(parent), child, FP8Linear(module))
    return model
