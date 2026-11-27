"""Checks for the RoPE changes in flash_head/src/modules/flash_head_model.py. Run from the repo root:

    python tests/test_rope.py

1. rope_apply (real arithmetic, optional precomputed cos/sin table) vs the original complex implementation:
   must be bit-identical, eager and under torch.compile.
2. Fused RMSNorm + RoPE Triton kernel vs RMSNorm + rope_apply: reports how many output elements differ
   (fp32 rotation instead of float64) and the time per call.
No checkpoints needed; random inputs at FlashHead Pro shapes (9 x 32 x 32 tokens, 12 heads x 128). Needs a CUDA GPU.
"""
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
from flash_head.src.modules.flash_head_model import (RMSNorm, precompute_freqs_cis_3d, rope_apply,
                                                      rope_cos_sin)
from flash_head.src.modules.fused_rmsnorm_rope import rmsnorm_rope


@torch.amp.autocast("cuda", enabled=False)
def rope_apply_original(x, freqs, grid_sizes):
    # Original single-GPU implementation (complex multiply), kept here as the reference.
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    (f, h, w) = grid_sizes
    seq_len = f * h * w
    x_i = torch.view_as_complex(x[0, :s].to(torch.float64).reshape(s, n, -1, 2))
    freqs_i = torch.cat([
        freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(seq_len, 1, -1)
    x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
    x_i = torch.cat([x_i, x[0, seq_len:]])
    return x_i.unsqueeze(0).to(x.dtype)


def time_ms(fn, *args, warmup=20, iters=200):
    for _ in range(warmup):
        fn(*args)
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return sorted(times)[iters // 2]


def main():
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    torch.manual_seed(0)
    F, H, W, N, D = 9, 32, 32, 12, 128
    L, DIM = F * H * W, N * D
    grid = (F, H, W)
    freqs = precompute_freqs_cis_3d(D).cuda()
    rope = rope_cos_sin(freqs, grid)
    print(f"torch {torch.__version__}, {torch.cuda.get_device_name()}")

    print("1. rope_apply vs original (complex):")
    ok = True
    x = torch.randn(1, L, N, D, device="cuda", dtype=torch.bfloat16)
    ref = rope_apply_original(x, freqs, grid)
    for name, fn in [("eager", lambda t: rope_apply(t, freqs, grid)),
                     ("eager, precomputed table", lambda t: rope_apply(t, freqs, grid, rope=rope)),
                     ("torch.compile", torch.compile(lambda t: rope_apply(t, freqs, grid))),
                     ("torch.compile, precomputed table", torch.compile(lambda t: rope_apply(t, freqs, grid, rope=rope)))]:
        same = torch.equal(fn(x), ref)
        ok &= same
        print(f"   {name:34s} {'bit-identical' if same else 'DIFFERENT'}")

    print("2. fused RMSNorm + RoPE kernel vs RMSNorm + rope_apply (q/k path, one call):")
    xq = torch.randn(1, L, DIM, device="cuda", dtype=torch.bfloat16)
    norm = RMSNorm(DIM, eps=1e-6).cuda().to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(1 + 0.1 * torch.randn(DIM, device="cuda"))
    cos32, sin32 = (t.view(L, -1).float() for t in rope)
    original = torch.compile(lambda t: rope_apply_original(norm(t).view(1, L, N, D), freqs, grid).flatten(2))
    real64 = torch.compile(lambda t: rope_apply(norm(t).view(1, L, N, D), freqs, grid, rope=rope).flatten(2))
    fused = lambda t: rmsnorm_rope(t[0], norm.weight, cos32, sin32, norm.eps)[None]
    with torch.no_grad():
        ref = rope_apply_original(norm(xq).view(1, L, N, D), freqs, grid).flatten(2)
        out = fused(xq)
        diff = (out.float() - ref.float()).abs()
        print(f"   elements that differ from eager RMSNorm + rope_apply: {100 * (diff > 0).float().mean().item():.4f}%, "
              f"max |diff| {diff.max().item():.4f}")
        t0, t1, t2 = time_ms(original, xq), time_ms(real64, xq), time_ms(fused, xq)
    print(f"   original, compiled            {t0:.3f} ms")
    print(f"   real arithmetic, compiled     {t1:.3f} ms  ({t0 / t1:.1f}x)")
    print(f"   fused Triton kernel           {t2:.3f} ms  ({t0 / t2:.1f}x)")
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
