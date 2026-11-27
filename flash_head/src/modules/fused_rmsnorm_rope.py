"""Fused q/k RMSNorm + 3D RoPE for SelfAttention: one Triton kernel instead of RMSNorm followed by rope_apply.

Each program handles one token: it reads the token's hidden vector once, computes RMSNorm in fp32 (rounded to
the input dtype before the weight multiply, as RMSNorm.forward does), rotates every (a, b) pair with the
precomputed cos/sin table in fp32, and writes the result once. The rotation runs in fp32 instead of float64,
so the output is not bit-identical to RMSNorm + rope_apply; see the PR description for the measured difference.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_rope_kernel(X, W, COS, SIN, OUT, stride_x, stride_o, eps,
                         DIM: tl.constexpr, HALF_HEAD: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)                     # token index
    p = tl.arange(0, BLOCK)                    # pair index: pair p = elements 2p, 2p+1
    m = p < DIM // 2
    xa = tl.load(X + row * stride_x + 2 * p, mask=m, other=0.).to(tl.float32)
    xb = tl.load(X + row * stride_x + 2 * p + 1, mask=m, other=0.).to(tl.float32)
    rstd = tl.rsqrt((tl.sum(xa * xa) + tl.sum(xb * xb)) / DIM + eps)
    wa = tl.load(W + 2 * p, mask=m, other=0.)
    wb = tl.load(W + 2 * p + 1, mask=m, other=0.)
    xa = ((xa * rstd).to(tl.bfloat16) * wa).to(tl.float32)
    xb = ((xb * rstd).to(tl.bfloat16) * wb).to(tl.float32)
    cos = tl.load(COS + row * HALF_HEAD + p % HALF_HEAD, mask=m, other=0.)   # all heads share one row of the table
    sin = tl.load(SIN + row * HALF_HEAD + p % HALF_HEAD, mask=m, other=0.)
    tl.store(OUT + row * stride_o + 2 * p, (xa * cos - xb * sin).to(tl.bfloat16), mask=m)
    tl.store(OUT + row * stride_o + 2 * p + 1, (xa * sin + xb * cos).to(tl.bfloat16), mask=m)


@torch.library.custom_op("flash_head::rmsnorm_rope", mutates_args=())
def rmsnorm_rope(x: torch.Tensor, w: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, eps: float) -> torch.Tensor:
    """x: [L, dim] bf16, w: [dim] bf16, cos/sin: [L, head_dim // 2] fp32 -> [L, dim] bf16."""
    L, D = x.shape
    assert x.dtype == w.dtype == torch.bfloat16 and cos.dtype == sin.dtype == torch.float32
    assert x.stride(1) == 1 and w.is_contiguous() and cos.is_contiguous() and sin.is_contiguous()
    assert cos.shape[0] == L and D % (2 * cos.shape[1]) == 0
    out = torch.empty((L, D), device=x.device, dtype=torch.bfloat16)
    _rmsnorm_rope_kernel[(L,)](x, w, cos, sin, out, x.stride(0), out.stride(0), eps,
                               DIM=D, HALF_HEAD=cos.shape[1], BLOCK=triton.next_power_of_2(D // 2), num_warps=4)
    return out


@rmsnorm_rope.register_fake
def _(x, w, cos, sin, eps):
    return torch.empty_like(x, dtype=torch.bfloat16)
