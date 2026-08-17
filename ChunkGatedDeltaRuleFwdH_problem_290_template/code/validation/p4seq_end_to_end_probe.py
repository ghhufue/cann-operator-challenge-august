#!/usr/bin/env python3
"""End-to-end probe: MM1 = P4 (contiguous A), MM2 = seq (transposed A)."""
import json
from pathlib import Path
import numpy as np
import torch

CASE_DIR = Path(r"D:\code\tmp_golden_probe\official_example")
CHUNK_SIZE = 64
DTYPE_MAP = {"bfloat16": (torch.bfloat16, "<u2"), "float32": (torch.float32, "<f4"), "int64": (torch.int64, "<i8")}

def load_bin(name, shape, dtype):
    tdt, ndt = DTYPE_MAP[dtype]
    arr = np.fromfile(CASE_DIR / name, dtype=np.dtype(ndt))
    ten = torch.from_numpy(arr.reshape(shape))
    if dtype == "bfloat16":
        ten = ten.view(torch.bfloat16)
    return ten

def cast_to_float16(t):
    return t.to(torch.float32).to(torch.float16)

def mm_p4(a, b):
    M, K = a.shape
    N = b.shape[1]
    af = a.to(torch.float32)
    bf = b.to(torch.float32)
    accs = [torch.zeros((M, N), dtype=torch.float32) for _ in range(4)]
    for k in range(K):
        accs[k % 4] += af[:, k:k+1] * bf[k:k+1, :]
    out = (accs[0] + accs[1]) + (accs[2] + accs[3])
    return out.to(torch.bfloat16)

def mm_seq(a, b):
    M, K = a.shape
    N = b.shape[1]
    af = a.to(torch.float32)
    bf = b.to(torch.float32)
    acc = torch.zeros((M, N), dtype=torch.float32)
    for k in range(K):
        acc += af[:, k:k+1] * bf[k:k+1, :]
    return acc.to(torch.bfloat16)

def run_reference(k, w, u, g, initial_state, cu_seqlens, chunk_indices):
    k_by_head = k.transpose(1, 2).contiguous()
    batch, key_heads, total_tokens, key_dim = k_by_head.shape
    value_heads, value_dim = u.shape[1], u.shape[3]
    sequence_count = cu_seqlens.numel() - 1
    chunk_count = chunk_indices.shape[0]
    chunk_offsets = [int(chunk_indices[n, 0]) for n in range(sequence_count)]
    head_ratio = value_heads // key_heads
    h_output = torch.zeros((batch, value_heads, chunk_count, key_dim, value_dim), dtype=k.dtype)
    v_output = torch.zeros((batch, value_heads, total_tokens, value_dim), dtype=u.dtype)
    final_state = torch.zeros((sequence_count, value_heads, key_dim, value_dim), dtype=u.dtype)
    for sequence in range(sequence_count):
        bos = int(cu_seqlens[sequence]); eos = int(cu_seqlens[sequence + 1])
        sequence_chunks = (eos - bos + CHUNK_SIZE - 1) // CHUNK_SIZE
        chunk_base = chunk_offsets[sequence]
        for head in range(value_heads):
            if initial_state is None:
                state = torch.zeros((key_dim, value_dim), dtype=k.dtype)
            else:
                state = initial_state[sequence, head].to(torch.float32).to(k.dtype)
            for chunk in range(sequence_chunks):
                token_start = bos + chunk * CHUNK_SIZE
                token_end = min(token_start + CHUNK_SIZE, eos)
                actual_len = token_end - token_start
                global_chunk = chunk_base + chunk
                h_output[0, head, global_chunk] = state
                k_sel = k_by_head[0, head // head_ratio, token_start:token_end, :]
                w_sel = w[0, head, token_start:token_end, :]
                u_sel = u[0, head, token_start:token_end, :]
                g_sel = g[0, head, token_start:token_end]
                ws = mm_p4(w_sel, state)
                ws_fp16 = torch.nan_to_num(ws.to(torch.float16), nan=0.0, posinf=torch.inf, neginf=-torch.inf)
                v_new = cast_to_float16(u_sel).float() - ws_fp16.float()
                v_new_fp16 = cast_to_float16(v_new)
                g_last = g_sel[actual_len - 1 : actual_len]
                gate = cast_to_float16((g_last - g_sel).exp().float())
                v_decay = v_new_fp16.float() * gate[..., None]
                v_decay_bf16 = cast_to_float16(v_decay).to(u.dtype)
                old_state_f16 = cast_to_float16(state)
                state_gate = cast_to_float16(g_last).exp().float()
                h_decay = old_state_f16.float() * state_gate[..., None]
                h_decay_fp16 = cast_to_float16(h_decay)
                update = mm_seq(k_sel.transpose(-1, -2).contiguous(), v_decay_bf16)
                update_fp16 = torch.nan_to_num(update.to(torch.float16), nan=0.0, posinf=torch.inf, neginf=-torch.inf)
                state = (h_decay_fp16.to(torch.bfloat16) + update_fp16.to(torch.bfloat16)).to(k.dtype)
                v_output[0, head, token_start:token_end, :] = v_new_fp16.to(u.dtype)
            final_state[sequence, head] = state.to(u.dtype)
    return h_output, v_output, final_state

def load_case():
    manifest = json.loads((CASE_DIR / "manifest.json").read_text())
    tensors = {}
    for name, meta in manifest["tensors"].items():
        shape = tuple(meta["shape"])
        if name.startswith("input") or name.startswith("golden"):
            tensors[name] = load_bin(meta["file"], shape, meta["dtype"])
    return tensors

def main():
    t = load_case()
    h, v, fs = run_reference(
        t["input_k"], t["input_w"], t["input_u"], t["input_g"],
        t["input_initial_state"], t["input_cu_seqlens"], t["input_chunk_indices"])
    for name, out, gold in (("h", h, t["golden_h"]), ("v", v, t["golden_v"]),
                            ("final_state", fs, t["golden_final_state"])):
        mism = (out != gold).sum().item()
        print(f"{name}: mismatches={mism}/{out.numel()}")

if __name__ == "__main__":
    main()
