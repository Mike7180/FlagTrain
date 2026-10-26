# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Ascend fast GELU.

Same math as ``flag_train.megatron.fast_gelu``; the kernels cap the grid at
Ascend's ``coreDim`` limit and stride over several blocks per program.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_train.runtime import device as runtime_device
from flag_train.runtime import torch_device_fn
from flag_train.utils import libentry, tl_extra_shim
from flag_train.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

BLOCK = 1024
# Ascend rejects kernel launches with coreDim > 65535, so cap the grid and let
# each program stride over several blocks: the core benchmark shapes need
# 131072 and 352256 programs otherwise.
_MAX_GRID = 65535
_FAST_GELU_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _check_input(tensor, name, dtypes):
    if tensor.dtype not in dtypes:
        wanted = " / ".join(str(dtype) for dtype in dtypes)
        raise ValueError(f"{name} needs a {wanted} tensor, got {tensor.dtype}")
    if tensor.device.type == "cpu":
        raise ValueError(f"{name} needs a device tensor, not one on the CPU")
    if tensor.device.type != runtime_device.name:
        raise ValueError(
            f"{name} needs a {runtime_device.name} tensor, got {tensor.device.type}"
        )


@libentry()
@triton.jit
def fast_gelu_kernel(x_ptr, out_ptr, n, n_blocks, BLOCK: tl.constexpr):
    num_programs = tl.num_programs(0)
    for pid in range(tle.program_id(0), n_blocks, num_programs):
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        out = (
            0.5
            * x
            * (1.0 + tl_extra_shim.tanh(0.7978845608 * x * (1.0 + 0.044715 * x * x)))
        )
        tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit
def fast_gelu_backward_kernel(
    grad_out_ptr, x_ptr, grad_x_ptr, n, n_blocks, BLOCK: tl.constexpr
):
    num_programs = tl.num_programs(0)
    for pid in range(tle.program_id(0), n_blocks, num_programs):
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        a = 0.044715
        c = 0.7978845608
        t = tl_extra_shim.tanh(c * x * (1 + a * x * x))
        grad_x = grad_out * (
            0.5 * (1 + t) + 0.5 * x * (1 - t * t) * c * (1 + 3 * a * x * x)
        )
        tl.store(
            grad_x_ptr + offsets, grad_x.to(grad_x_ptr.dtype.element_ty), mask=mask
        )


class FastGELU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        _check_input(x, "fast_gelu", _FAST_GELU_DTYPES)
        ctx.save_for_backward(x)
        if x.numel() == 0:
            return torch.empty_like(x)
        x_flat = x.contiguous().view(-1)
        out = torch.empty_like(x_flat)
        n_blocks = triton.cdiv(x.numel(), BLOCK)
        grid = (min(n_blocks, _MAX_GRID),)
        with torch_device_fn.device(x.device):
            fast_gelu_kernel[grid](x_flat, out, x.numel(), n_blocks, BLOCK=BLOCK)
        return out.view_as(x)

    @staticmethod
    def backward(ctx, grad_out):
        logger.debug("TRAIN_ASCEND FAST_GELU_BACKWARD")
        (x,) = ctx.saved_tensors
        _check_input(x, "fast_gelu", _FAST_GELU_DTYPES)
        _check_input(grad_out, "fast_gelu grad", _FAST_GELU_DTYPES)
        if grad_out.shape != x.shape or grad_out.dtype != x.dtype:
            raise ValueError(
                f"fast_gelu grad needs the same shape and dtype as the input, got "
                f"{tuple(grad_out.shape)} {grad_out.dtype} for "
                f"{tuple(x.shape)} {x.dtype}"
            )
        if x.numel() == 0:
            return torch.empty_like(x)

        x_flat = x.contiguous().view(-1)
        grad_out_flat = grad_out.contiguous().view(-1)
        n = x_flat.numel()

        grad_x = torch.empty_like(x_flat)
        n_blocks = triton.cdiv(n, BLOCK)
        grid = (min(n_blocks, _MAX_GRID),)
        with torch_device_fn.device(x.device):
            fast_gelu_backward_kernel[grid](
                grad_out_flat, x_flat, grad_x, n, n_blocks, BLOCK=BLOCK
            )
        return grad_x.view(x.shape)


def fast_gelu(x):
    logger.debug("TRAIN_ASCEND FAST_GELU")
    return FastGELU.apply(x)
