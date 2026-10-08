#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

# Add nanolab to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../nanolab")))
from mixers import MinGRU, _parallel_scan_log, ssd_chunk_parallel

def dump_mingru():
    print("Dumping minGRU golden...")
    B, T, H = 2, 32, 128
    torch.manual_seed(1337)
    
    # Inputs to the scan
    Z_in = torch.randn(B, T, H, requires_grad=True)
    H_in = torch.randn(B, T, H, requires_grad=True)
    
    # minGRU PyTorch math (from mixers.py)
    def _log_g(x):
        return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))
    
    log_z = -F.softplus(-Z_in)
    log_coeff = -F.softplus(Z_in)
    log_h_tilde = _log_g(H_in)
    log_zh = log_z + log_h_tilde
    
    # Forward
    h_out = _parallel_scan_log(log_coeff, log_zh)
    
    # Backward
    grad_h_out = torch.randn(B, T, H)
    h_out.backward(grad_h_out)
    
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../golden/mingru"))
    os.makedirs(out_dir, exist_ok=True)
    
    np.save(os.path.join(out_dir, "Z_in.npy"), Z_in.detach().numpy())
    np.save(os.path.join(out_dir, "H_in.npy"), H_in.detach().numpy())
    np.save(os.path.join(out_dir, "h_out.npy"), h_out.detach().numpy())
    np.save(os.path.join(out_dir, "grad_h_out.npy"), grad_h_out.numpy())
    np.save(os.path.join(out_dir, "grad_Z_in.npy"), Z_in.grad.numpy())
    np.save(os.path.join(out_dir, "grad_H_in.npy"), H_in.grad.numpy())

def dump_mingru_vr():
    """Value-residual blend: h_pre = vr_lambda[0]*v0_up + vr_lambda[1]*h_raw (mixers.py)."""
    print("Dumping minGRU VR blend golden...")
    B, T, H = 2, 32, 128
    torch.manual_seed(1337)

    h_raw = torch.randn(B, T, H, requires_grad=True)
    v0_up = torch.randn(B, T, H, requires_grad=True)
    vr_lambda = torch.tensor([0.5, 0.5], requires_grad=True)

    h_pre = vr_lambda[0] * v0_up + vr_lambda[1] * h_raw

    grad_h_pre = torch.randn(B, T, H)
    h_pre.backward(grad_h_pre)

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../golden/mingru_vr"))
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "h_raw.npy"), h_raw.detach().numpy())
    np.save(os.path.join(out_dir, "v0_up.npy"), v0_up.detach().numpy())
    np.save(os.path.join(out_dir, "vr_lambda.npy"), vr_lambda.detach().numpy())
    np.save(os.path.join(out_dir, "h_pre.npy"), h_pre.detach().numpy())
    np.save(os.path.join(out_dir, "grad_h_pre.npy"), grad_h_pre.numpy())
    np.save(os.path.join(out_dir, "grad_h_raw.npy"), h_raw.grad.numpy())
    np.save(os.path.join(out_dir, "grad_v0_up.npy"), v0_up.grad.numpy())
    np.save(os.path.join(out_dir, "grad_vr_lambda.npy"), vr_lambda.grad.numpy())

def dump_mingru_vr_layer():
    """Full MinGRU+VR layer (v_proj, v0_up, blend, scan) matching nanolab mixers.py."""
    print("Dumping minGRU+VR full-layer golden...")
    B, T = 2, 16
    torch.manual_seed(4242)

    class Cfg:
        d_model = 64
        n_kv_head = 2
        head_dim = 16
        value_residual = True

    cfg = Cfg()
    layer = MinGRU(cfg)
    x = torch.randn(B, T, cfg.d_model, requires_grad=True)
    v0 = torch.randn(B, T, cfg.n_kv_head, cfg.head_dim)

    out, raw_v = layer(x, v0=v0)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)

    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../golden/mingru_vr_layer"))
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "x.npy"), x.detach().numpy())
    np.save(os.path.join(out_dir, "v0.npy"), v0.numpy())
    np.save(os.path.join(out_dir, "out.npy"), out.detach().numpy())
    np.save(os.path.join(out_dir, "raw_v.npy"), raw_v.detach().numpy())
    np.save(os.path.join(out_dir, "grad_out.npy"), grad_out.numpy())
    np.save(os.path.join(out_dir, "grad_x.npy"), x.grad.numpy())
    np.save(os.path.join(out_dir, "to_z_weight.npy"), layer.to_z.weight.detach().numpy())
    np.save(os.path.join(out_dir, "to_h_weight.npy"), layer.to_h.weight.detach().numpy())
    np.save(os.path.join(out_dir, "out_weight.npy"), layer.out.weight.detach().numpy())
    np.save(os.path.join(out_dir, "v_proj_weight.npy"), layer.v_proj.weight.detach().numpy())
    np.save(os.path.join(out_dir, "v0_up_weight.npy"), layer.v0_up.weight.detach().numpy())
    np.save(os.path.join(out_dir, "vr_lambda.npy"), layer.vr_lambda.detach().numpy())
    for name, p in [
        ("grad_to_z", layer.to_z.weight),
        ("grad_to_h", layer.to_h.weight),
        ("grad_out_w", layer.out.weight),
        ("grad_v_proj", layer.v_proj.weight),
        ("grad_v0_up", layer.v0_up.weight),
    ]:
        if p.grad is not None:
            np.save(os.path.join(out_dir, f"{name}.npy"), p.grad.numpy())
    if layer.vr_lambda.grad is not None:
        np.save(os.path.join(out_dir, "grad_vr_lambda.npy"), layer.vr_lambda.grad.numpy())

def dump_mamba2():
    print("Dumping Mamba2 golden...")
    Bsz, L, H, P, N = 2, 32, 4, 16, 16
    chunk = 16
    torch.manual_seed(1337)
    
    # Inputs to the scan
    x_scaled = torch.randn(Bsz, L, H, P, requires_grad=True)
    B_h = torch.randn(Bsz, L, N, requires_grad=True)
    C_h = torch.randn(Bsz, L, N, requires_grad=True)
    
    # log_dA must be <= 0 for numerical stability in the scan
    log_dA = (torch.rand(Bsz, L, H) * -1.0).detach().requires_grad_()
    
    # Forward
    y = ssd_chunk_parallel(x_scaled, B_h, C_h, log_dA, chunk)
    
    # Backward
    grad_y = torch.randn(Bsz, L, H, P)
    y.backward(grad_y)
    
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../golden/mamba2"))
    os.makedirs(out_dir, exist_ok=True)
    
    np.save(os.path.join(out_dir, "x_scaled.npy"), x_scaled.detach().numpy())
    np.save(os.path.join(out_dir, "B_h.npy"), B_h.detach().numpy())
    np.save(os.path.join(out_dir, "C_h.npy"), C_h.detach().numpy())
    np.save(os.path.join(out_dir, "log_dA.npy"), log_dA.detach().numpy())
    np.save(os.path.join(out_dir, "y.npy"), y.detach().numpy())
    np.save(os.path.join(out_dir, "grad_y.npy"), grad_y.numpy())
    np.save(os.path.join(out_dir, "grad_x_scaled.npy"), x_scaled.grad.numpy())
    np.save(os.path.join(out_dir, "grad_B_h.npy"), B_h.grad.numpy())
    np.save(os.path.join(out_dir, "grad_C_h.npy"), C_h.grad.numpy())
    np.save(os.path.join(out_dir, "grad_log_dA.npy"), log_dA.grad.numpy())

if __name__ == "__main__":
    dump_mingru()
    dump_mingru_vr()
    dump_mingru_vr_layer()
    dump_mamba2()
    print("Done!")
