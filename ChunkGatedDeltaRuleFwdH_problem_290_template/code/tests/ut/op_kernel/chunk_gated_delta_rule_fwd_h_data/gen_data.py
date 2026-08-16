#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from ml_dtypes import bfloat16

import json
import torch
import torch.nn.functional as F
import math
import random

import os

def cdiv_torch(a, b):
    return (a + b - 1) // b
    
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int
) -> torch.LongTensor:
    indices = torch.cat([torch.arange(n) for n in cdiv_torch(prepare_lens(cu_seqlens), chunk_size).tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)

def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]
    
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor,
    chunk_size: int
) -> torch.LongTensor:
    return torch.cat([cu_seqlens.new_tensor([0]), cdiv_torch(prepare_lens(cu_seqlens), chunk_size)]).cumsum(-1) 

def cast_to_float16(x):
    is_special = torch.isinf(x) | torch.isnan(x)
    x_f16 = x.to(torch.float16)
    clamped = torch.clamp(x_f16, -65504.0, 65504.0)
    return torch.where(is_special, x_f16, clamped)
    
    
def run_cpu(k, w, u, g, initial_state, cu_seqlens, chunk_indices, chunk_size):    
    k = k.transpose(1,2).contiguous() # 转置后shape：[B,HK, T,K]
    B, HK, T, K = k.shape[0], k.shape[1], k.shape[2], k.shape[3]
    HV, V = u.shape[1], u.shape[3]
    BT = chunk_size  # 固定为64
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, (T + BT - 1) # 等长序列
    else:
        N, NT, chunk_offsets = len(cu_seqlens) - 1, len(chunk_indices), prepare_chunk_offsets(cu_seqlens, BT) # 变长序列
    if initial_state is not None:
        initial_state = initial_state.reshape([N, HV, K, V]).contiguous().to(torch.float32)  # 保持 float32 用于初始化
    else:
        initial_state = None

    S = torch.zeros((B, HV, NT, K, V), device=k.device, dtype=k.dtype)
    v_new_output = torch.zeros((B, HV, T, V), device=k.device, dtype=u.dtype)
    final_state = torch.zeros((N, HV, K, V), device=k.device, dtype=u.dtype)

    head_ratio = HV // HK
    for n in range(N):
        if cu_seqlens is None:
            bos = 0
            eos = T
            T_inner = T
            NT_inner = NT
            boh = 0
        else:
            bos = cu_seqlens[n]
            eos = cu_seqlens[n + 1]
            T_inner = eos - bos
            NT_inner = (T_inner + BT - 1) // BT
            boh = chunk_offsets[n]

        for h in range(HV):
            for i in range(NT_inner):
                actual_len = min(bos + (i + 1) * BT, eos) - (bos + i * BT)

                if cu_seqlens is None:
                    k_sel = k[n, h // head_ratio, bos + i * BT : bos + i * BT + actual_len, :]
                    w_sel = w[n, h, bos + i * BT : bos + i * BT + actual_len, :]
                    u_sel = u[n, h, bos + i * BT : bos + i * BT + actual_len, :]
                    g_sel = g[n, h, bos + i * BT : bos + i * BT + actual_len]

                    if initial_state is not None and i == 0:
                        S[n, h, boh+i] = initial_state[n, h]
                else:
                    k_sel = k[0, h // head_ratio, bos + i * BT : bos + i * BT + actual_len, :]
                    w_sel = w[0, h, bos + i * BT : bos + i * BT + actual_len, :]
                    u_sel = u[0, h, bos + i * BT : bos + i * BT + actual_len, :]
                    g_sel = g[0, h, bos + i * BT : bos + i * BT + actual_len]
                    
                    if initial_state is not None and i == 0:
                        S[0, h, boh+i] = initial_state[n, h]

                # if h==1 and i==1:
                #     breakpoint()
                ws = w_sel @ S[0, h, boh+i]
                ws_fp16 = ws.to(torch.float16)
                ws_fp16 = torch.nan_to_num(ws_fp16, nan=0.0, posinf=torch.inf, neginf=-torch.inf)
                v_new = cast_to_float16(u_sel).float() - ws_fp16.float()
                v_new_fp16 = cast_to_float16(v_new)      
                g_last = g_sel[actual_len-1:actual_len]
                v_new_decay = v_new_fp16.float() * cast_to_float16((g_last - g_sel).exp().float())[..., None]
                v_new_decay_fp16 = cast_to_float16(v_new_decay)
                v_new_decay = v_new_decay_fp16.to(u.dtype)
                h_decay = cast_to_float16(S[0, h, boh+i]).float() * cast_to_float16(g_last).exp().float()[..., None]
                h_decay_fp16 = cast_to_float16(h_decay)

                k_vnew = (k_sel.transpose(-1, -2) @ v_new_decay)
                k_vnew_fp16 = k_vnew.to(torch.float16)
                k_vnew_fp16 = torch.nan_to_num(k_vnew_fp16, nan=0.0, posinf=torch.inf, neginf=-torch.inf)
                next_h = h_decay_fp16.to(torch.bfloat16) + k_vnew_fp16.to(torch.bfloat16)

                if cu_seqlens is None:
                    if i != NT_inner-1:
                        S[n, h, boh+i+1] = next_h.to(S.dtype)
                    else:
                        final_state[n, h] = next_h.to(S.dtype)
                    v_new_output[n, h, bos + i * BT: bos + i * BT + actual_len, :] = v_new_fp16.to(u.dtype)
                else:
                    if i != NT_inner-1:
                        S[0, h, boh+i+1] = next_h.to(S.dtype)
                    else:
                        final_state[n, h] = next_h.to(S.dtype)
                    v_new_output[0, h, bos + i * BT: bos + i * BT + actual_len, :] = v_new_fp16.to(u.dtype) 

    return S, v_new_output, final_state


def impl(k, w, u, g, initial_state, cu_seqlens, chunk_indices, chunk_size):
    k = np.asarray(k, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    u = np.asarray(u, dtype=np.float32)
    initial_state = np.asarray(initial_state, dtype=np.float32)
    
    k = torch.from_numpy(k)
    w = torch.from_numpy(w)
    u = torch.from_numpy(u)
    g = torch.from_numpy(g)
    initial_state = torch.from_numpy(initial_state)
    cu_seqlens = torch.from_numpy(cu_seqlens)
    chunk_indices = torch.from_numpy(chunk_indices)
    
    cpu_output_0, cpu_output_1, cpu_output_2 = run_cpu(k, w, u, g, initial_state, cu_seqlens, chunk_indices, chunk_size)
    
    return cpu_output_0.detach().cpu().numpy().astype(bfloat16), cpu_output_1.detach().cpu().numpy().astype(bfloat16), cpu_output_2.detach().cpu().numpy().astype(bfloat16)


if __name__ == "__main__":
    # 清理bin文件
    for f in glob.glob("*.bin"):
        os.remove(f)
    
    # 从 JSON 第一个 case 获取参数
    d_type = "bfloat16"
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": bfloat16,
        "float64": np.float64,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
        "bool": np.bool_,
        "fp8_e4m3fn": np.uint8,
        "fp8_e5m2": np.uint8,
    }
    np_type = d_type_dict[d_type]
    
    # 生成输入数据
    input_k = np.ones((1, 10016, 2, 128)).astype(d_type_dict["bfloat16"])
    input_w = np.ones((1, 8, 10016, 128)).astype(d_type_dict["bfloat16"])
    input_u = np.ones((1, 8, 10016, 128)).astype(d_type_dict["bfloat16"])
    input_g = np.ones((1, 8, 10016)).astype(d_type_dict["float32"])
    input_initial_state = np.ones((2, 8, 128, 128)).astype(d_type_dict["bfloat16"])
    input_cu_seqlens = np.ones((3)).astype(d_type_dict["int64"])
    input_chunk_indices = np.ones((158, 2)).astype(d_type_dict["int64"])
    attr_chunk_size = 64
    
    # 计算 golden 数据
    golden = impl(input_k, input_w, input_u, input_g, input_initial_state, input_cu_seqlens, input_chunk_indices, attr_chunk_size)
    
    # 保存数据到文件
    input_k.astype(d_type_dict["bfloat16"]).tofile(f"{d_type}_input_chunk_gated_delta_rule_fwd_h_k.bin")
    input_w.astype(d_type_dict["bfloat16"]).tofile(f"{d_type}_input_chunk_gated_delta_rule_fwd_h_w.bin")
    input_u.astype(d_type_dict["bfloat16"]).tofile(f"{d_type}_input_chunk_gated_delta_rule_fwd_h_u.bin")
    input_g.astype(d_type_dict["float32"]).tofile(f"{d_type}_input_chunk_gated_delta_rule_fwd_h_g.bin")
    input_initial_state.astype(d_type_dict["bfloat16"]).tofile(f"{d_type}_input_chunk_gated_delta_rule_fwd_h_initial_state.bin")
    input_cu_seqlens.astype(d_type_dict["int64"]).tofile(f"{d_type}_input_chunk_gated_delta_rule_fwd_h_cu_seqlens.bin")
    input_chunk_indices.astype(d_type_dict["int64"]).tofile(f"{d_type}_input_chunk_gated_delta_rule_fwd_h_chunk_indices.bin")
    if golden is not None:
        if isinstance(golden, (list, tuple)):
            _out_dtypes = ["bfloat16", "bfloat16", "bfloat16"]
            for _gi, _g in enumerate(golden):
                _dt = _out_dtypes[_gi] if _gi < len(_out_dtypes) else _out_dtypes[-1]
                _g.astype(d_type_dict[_dt]).tofile(f"{_dt}_golden_chunk_gated_delta_rule_fwd_h_{_gi}.bin")
        else:
            golden.astype(d_type_dict["bfloat16"]).tofile("bfloat16_golden_chunk_gated_delta_rule_fwd_h_0.bin")
    
    print(f"生成完成: dtype={d_type}")
