#!/usr/bin/env python3
"""Compare one-chunk intermediate values dumped into the NPU workspace.

The kernel writes the following fields for every task at chunk 0 into a
per-task slot at the tail of the workspace:

  0: ws bf16           [actual_len, tile]
  1: v_new fp16        [actual_len, tile]
  2: gate fp16         [actual_len]
  3: v_decay bf16      [actual_len, tile]
  4: h_decay bf16      [key_dim, tile]
  5: mm2 bf16          [key_dim, tile]
  6: next_h bf16       [key_dim, tile]

This script recomputes the same values for one task with the Python reference
formulas and compares raw 16-bit storage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ml_dtypes import bfloat16

CHUNK_SIZE = 64
WORKSPACE_ALIGNMENT = 512


def align_up(value: int, alignment: int = WORKSPACE_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def debug_layout(chunk_size: int, key_dim: int) -> tuple[list[int], int]:
    tile = min(64, key_dim)
    fields = [
        ("ws", chunk_size * tile),
        ("v_new", chunk_size * tile),
        ("gate", chunk_size),
        ("v_decay", chunk_size * tile),
        ("h_decay", key_dim * tile),
        ("mm2", key_dim * tile),
        ("next_h", key_dim * tile),
    ]
    offsets: list[int] = []
    pos = 0
    for _name, elements in fields:
        pos = align_up(pos)
        offsets.append(pos)
        pos += elements * 2
    return offsets, align_up(pos)


def load_manifest_tensor(manifest: dict, key: str, case_dir: Path) -> torch.Tensor:
    meta = manifest["tensors"][key]
    shape = tuple(int(dim) for dim in meta["shape"])
    path = case_dir / meta["file"]
    dtype = meta["dtype"]
    if dtype == "bfloat16":
        bits = np.fromfile(path, dtype=np.uint16)
        values = bits.view(bfloat16).reshape(shape)
        return torch.from_numpy(np.asarray(values, dtype=np.float32)).to(torch.bfloat16)
    if dtype == "float32":
        values = np.fromfile(path, dtype=np.float32).reshape(shape)
        return torch.from_numpy(values.copy())
    if dtype == "int64":
        values = np.fromfile(path, dtype=np.int64).reshape(shape)
        return torch.from_numpy(values.copy())
    raise ValueError(f"Unsupported dtype: {dtype}")


def cast_to_float16(value: torch.Tensor) -> torch.Tensor:
    is_special = torch.isinf(value) | torch.isnan(value)
    value_f16 = value.to(torch.float16)
    clamped = torch.clamp(value_f16, -65504.0, 65504.0)
    return torch.where(is_special, value_f16, clamped)


def torch_bits(value: torch.Tensor, dtype: str) -> np.ndarray:
    if dtype == "bfloat16":
        arr = np.asarray(value.to(torch.float32).numpy(), dtype=np.float32).astype(bfloat16)
        return arr.view(np.uint16).reshape(value.shape)
    if dtype == "float16":
        arr = np.asarray(value.to(torch.float32).numpy(), dtype=np.float32).astype(np.float16)
        return arr.view(np.uint16).reshape(value.shape)
    raise ValueError(dtype)


def read_bits(data: bytes, offset: int, elements: int, shape: tuple[int, ...]) -> np.ndarray:
    bits = np.frombuffer(data, dtype=np.uint16, count=elements, offset=offset)
    return bits.reshape(shape)


def compare_field(name: str, actual_bits: np.ndarray, golden_bits: np.ndarray) -> bool:
    mismatches = int(np.count_nonzero(actual_bits != golden_bits))
    total = actual_bits.size
    passed = mismatches == 0
    print(
        f"{name}: {'PASSED' if passed else 'FAILED'}; "
        f"failed={mismatches}/{total} ({mismatches / total:.6%})"
    )
    if not passed:
        flat = np.flatnonzero((actual_bits != golden_bits).reshape(-1))[:8]
        for idx in flat:
            index = np.unravel_index(int(idx), actual_bits.shape)
            print(
                f"  index={list(index)}, "
                f"actual_bits=0x{int(actual_bits[index]):04x}, "
                f"golden_bits=0x{int(golden_bits[index]):04x}"
            )
    return passed


def run_case(case_dir: Path, task_id: int) -> int:
    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    value_heads = int(manifest["tensors"]["input_w"]["shape"][1])
    key_heads = int(manifest["tensors"]["input_k"]["shape"][2])
    key_dim = int(manifest["tensors"]["input_k"]["shape"][3])
    value_dim = int(manifest["tensors"]["input_u"]["shape"][3])
    tile = min(64, value_dim)
    v_tile_count = (value_dim + tile - 1) // tile

    cu_seqlens = load_manifest_tensor(manifest, "input_cu_seqlens", case_dir)
    sequence_count = int(cu_seqlens.numel() - 1)
    task_count = sequence_count * value_heads * v_tile_count
    if task_id < 0 or task_id >= task_count:
        print(
            f"task_id {task_id} out of range; valid range is [0, {task_count - 1}]"
        )
        return 2

    v_tile = task_id % v_tile_count
    sequence_and_head = task_id // v_tile_count
    head = sequence_and_head % value_heads
    sequence = sequence_and_head // value_heads
    v_start = v_tile * tile
    bos = int(cu_seqlens[sequence].item())
    eos = int(cu_seqlens[sequence + 1].item())
    actual_len = min(CHUNK_SIZE, eos - bos)

    k = load_manifest_tensor(manifest, "input_k", case_dir)
    w = load_manifest_tensor(manifest, "input_w", case_dir)
    u = load_manifest_tensor(manifest, "input_u", case_dir)
    g = load_manifest_tensor(manifest, "input_g", case_dir)

    head_ratio = value_heads // key_heads
    key_head = head // head_ratio
    k_by_head = k.transpose(1, 2).contiguous()

    if "input_initial_state" in manifest["tensors"]:
        initial_state = load_manifest_tensor(manifest, "input_initial_state", case_dir)
        state = initial_state[sequence, head].to(torch.float32).to(k.dtype)
    else:
        state = torch.zeros((key_dim, value_dim), dtype=k.dtype)

    w_sel = w[0, head, bos:bos + actual_len, :]
    u_sel = u[0, head, bos:bos + actual_len, :]
    g_sel = g[0, head, bos:bos + actual_len]
    k_sel = k_by_head[0, key_head, bos:bos + actual_len, :]

    ws = w_sel @ state
    ws_fp16 = torch.nan_to_num(
        ws.to(torch.float16), nan=0.0, posinf=torch.inf, neginf=-torch.inf
    )
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

    update = k_sel.transpose(-1, -2) @ v_decay_bf16
    update_fp16 = torch.nan_to_num(
        update.to(torch.float16), nan=0.0, posinf=torch.inf, neginf=-torch.inf
    )
    next_state = (
        h_decay_fp16.to(torch.bfloat16) + update_fp16.to(torch.bfloat16)
    ).to(k.dtype)

    fields = [
        ("ws", 0, "bfloat16", actual_len * tile, (actual_len, tile),
         torch_bits(ws[:, v_start:v_start + tile], "bfloat16")),
        ("v_decay", 3, "bfloat16", actual_len * tile, (actual_len, tile),
         torch_bits(v_decay_bf16[:, v_start:v_start + tile], "bfloat16")),
        ("h_decay", 4, "bfloat16", key_dim * tile, (key_dim, tile),
         torch_bits(h_decay_fp16.to(torch.bfloat16)[:, v_start:v_start + tile], "bfloat16")),
        ("mm2", 5, "bfloat16", key_dim * tile, (key_dim, tile),
         torch_bits(update[:, v_start:v_start + tile], "bfloat16")),
        ("next_h", 6, "bfloat16", key_dim * tile, (key_dim, tile),
         torch_bits(next_state[:, v_start:v_start + tile], "bfloat16")),
    ]

    workspace_path = case_dir / "actual_workspace.bin"
    if not workspace_path.exists():
        print("Missing actual_workspace.bin; run the ACLNN runner first.")
        return 2

    data = workspace_path.read_bytes()
    offsets, per_task_bytes = debug_layout(CHUNK_SIZE, key_dim)
    debug_bytes = per_task_bytes * task_count
    if len(data) < debug_bytes:
        print(
            f"Workspace dump is shorter than debug region: {len(data)} < {debug_bytes}"
        )
        return 2
    debug = data[-debug_bytes:]

    print(
        f"Task {task_id}: sequence={sequence}, value_head={head}, "
        f"v_tile={v_tile}, v_start={v_start}, actual_len={actual_len}, "
        f"tile={tile}, task_count={task_count}"
    )

    all_passed = True
    slot_start = task_id * per_task_bytes
    for field, offset_index, _dtype, elements, shape, golden_bits in fields:
        actual_bits = read_bits(debug, slot_start + offsets[offset_index], elements, shape)
        all_passed &= compare_field(field, actual_bits, golden_bits)

    return 0 if all_passed else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path, help="generated Golden case directory")
    parser.add_argument(
        "--task-id",
        type=int,
        default=0,
        help="logical task id: task_id = (sequence * HV + value_head) * v_tiles + v_tile",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run_case(args.case_dir.resolve(), args.task_id)
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(f"Intermediate comparison failed: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
