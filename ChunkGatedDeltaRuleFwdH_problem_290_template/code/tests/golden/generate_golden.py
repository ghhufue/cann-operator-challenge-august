#!/usr/bin/env python3
"""Generate deterministic small golden cases for ChunkGatedDeltaRuleFwdH.

BF16 tensors are written as their raw 16-bit storage. Each case also gets a
manifest.json containing shapes, dtypes, file sizes, and SHA-256 digests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


CHUNK_SIZE = 64


@dataclass(frozen=True)
class CaseSpec:
    name: str
    seq_lens: tuple[int, ...]
    hv: int
    hk: int
    dim: int
    seed: int
    description: str
    with_initial_state: bool = True


CASES: tuple[CaseSpec, ...] = (
    CaseSpec(
        name="smoke_single_partial",
        seq_lens=(5,),
        hv=2,
        hk=1,
        dim=32,
        seed=11,
        description="Single short sequence; exercises a partial first/last chunk.",
    ),
    CaseSpec(
        name="boundary_63_64_65",
        seq_lens=(63, 64, 65),
        hv=4,
        hk=2,
        dim=32,
        seed=23,
        description="Packed sequences immediately below, at, and above one chunk.",
    ),
    CaseSpec(
        name="boundary_127_128_129",
        seq_lens=(127, 128, 129),
        hv=2,
        hk=1,
        dim=32,
        seed=37,
        description="Exercises state propagation around the two-chunk boundary.",
    ),
    CaseSpec(
        name="grouped_heads_packed",
        seq_lens=(1, 70, 128, 129),
        hv=8,
        hk=2,
        dim=32,
        seed=53,
        description="Four packed sequences with four value heads per key head.",
    ),
    CaseSpec(
        name="two_chunk_single",
        seq_lens=(65,),
        hv=1,
        hk=1,
        dim=32,
        seed=67,
        description="Minimal two-chunk recurrence used to isolate state propagation.",
    ),
    CaseSpec(
        name="two_v_tiles",
        seq_lens=(65,),
        hv=1,
        hk=1,
        dim=128,
        seed=71,
        description="Two value tiles and two chunks for task/workspace validation.",
    ),
    CaseSpec(
        name="four_v_tiles",
        seq_lens=(65,),
        hv=1,
        hk=1,
        dim=256,
        seed=79,
        description="Four value tiles for concurrent task and workspace validation.",
    ),
    CaseSpec(
        name="without_initial_state",
        seq_lens=(65,),
        hv=2,
        hk=1,
        dim=32,
        seed=83,
        description="Two-chunk recurrence starting from the implicit zero state.",
        with_initial_state=False,
    ),
    CaseSpec(
        name="judge_min_length",
        seq_lens=(1024,),
        hv=2,
        hk=1,
        dim=128,
        seed=97,
        description="Minimum official sequence length with grouped value heads.",
    ),

    CaseSpec(
        name="official_example",
        seq_lens=(61, 3),
        hv=16,
        hk=2,
        dim=128,
        seed=101,
        description="Packed config from problem.txt: HV=16, HK=2, D=128, two short sequences.",
    ),
    CaseSpec(
        name="single_token",
        seq_lens=(1,),
        hv=2,
        hk=1,
        dim=32,
        seed=103,
        description="Single-token sequence; partial chunk of length one.",
    ),
    CaseSpec(
        name="long_1024_single",
        seq_lens=(1024,),
        hv=1,
        hk=1,
        dim=128,
        seed=107,
        description="Sixteen full chunks, single head/tile, long recurrence.",
    ),
    CaseSpec(
        name="long_1024_heads",
        seq_lens=(1024,),
        hv=16,
        hk=2,
        dim=128,
        seed=109,
        description="Official-style head count at T=1024: HV=16, HK=2, D=128.",
    ),
    CaseSpec(
        name="long_1024_wide",
        seq_lens=(1024,),
        hv=2,
        hk=1,
        dim=256,
        seed=113,
        description="Four value tiles with a long recurrence.",
    ),
    CaseSpec(
        name="long_2048_heads",
        seq_lens=(2048,),
        hv=8,
        hk=2,
        dim=128,
        seed=127,
        description="Thirty-two chunks with grouped heads.",
    ),
    CaseSpec(
        name="long_4096_single",
        seq_lens=(4096,),
        hv=1,
        hk=1,
        dim=64,
        seed=131,
        description="Sixty-four chunks; many tasks over the whole device.",
    ),
    CaseSpec(
        name="long_4096_heads",
        seq_lens=(4096,),
        hv=16,
        hk=4,
        dim=128,
        seed=137,
        description="Sixty-four chunks with many heads; stresses multi-core task spread.",
    ),
    CaseSpec(
        name="varlen_many",
        seq_lens=(5, 64, 129, 1023),
        hv=4,
        hk=2,
        dim=64,
        seed=139,
        description="Four packed sequences from tiny to long (token_batch limit is 4).",
    ),
    CaseSpec(
        name="varlen_eval_style",
        seq_lens=(61, 3, 512, 1024),
        hv=16,
        hk=2,
        dim=128,
        seed=149,
        description="Official example packed with two long sequences.",
    ),
    CaseSpec(
        name="no_state_long",
        seq_lens=(1024,),
        hv=2,
        hk=1,
        dim=64,
        seed=151,
        description="Long recurrence from implicit zero state.",
        with_initial_state=False,
    ),
    CaseSpec(
        name="boundary_1023_1024_1025",
        seq_lens=(1023, 1024, 1025),
        hv=2,
        hk=1,
        dim=32,
        seed=157,
        description="Packed sequences around the sixteen-chunk boundary.",
    ),
    CaseSpec(
        name="heads_equal_hk_hv",
        seq_lens=(1, 2, 3, 64),
        hv=8,
        hk=8,
        dim=32,
        seed=163,
        description="HK equals HV: no head grouping, one key head per value head.",
    ),
    CaseSpec(
        name="wide_state_256",
        seq_lens=(65,),
        hv=4,
        hk=1,
        dim=256,
        seed=167,
        description="Four value tiles with grouped value heads.",
    ),
    CaseSpec(
        name="judge_2048_32h",
        seq_lens=(2048,),
        hv=32,
        hk=4,
        dim=128,
        seed=173,
        description="Thirty-two value heads over thirty-two chunks.",
    ),
)


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def prepare_chunk_indices(
    cu_seqlens: torch.Tensor, chunk_size: int = CHUNK_SIZE
) -> torch.Tensor:
    """Return rows of (sequence index, chunk index within that sequence)."""
    rows: list[tuple[int, int]] = []
    for sequence in range(cu_seqlens.numel() - 1):
        seq_len = int(cu_seqlens[sequence + 1] - cu_seqlens[sequence])
        rows.extend((sequence, chunk) for chunk in range(ceil_div(seq_len, chunk_size)))
    if not rows:
        return torch.empty((0, 2), dtype=torch.int64)
    return torch.tensor(rows, dtype=torch.int64)


def prepare_chunk_offsets(
    cu_seqlens: torch.Tensor, chunk_size: int = CHUNK_SIZE
) -> torch.Tensor:
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    chunk_counts = torch.div(
        seq_lens + chunk_size - 1, chunk_size, rounding_mode="floor"
    )
    return torch.cat((torch.zeros(1, dtype=torch.int64), chunk_counts.cumsum(0)))


def cast_to_float16(value: torch.Tensor) -> torch.Tensor:
    """Match the saturation and special-value behavior in problem.txt."""
    is_special = torch.isinf(value) | torch.isnan(value)
    value_f16 = value.to(torch.float16)
    clamped = torch.clamp(value_f16, -65504.0, 65504.0)
    return torch.where(is_special, value_f16, clamped)


def validate_inputs(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    chunk_size: int,
) -> None:
    if chunk_size != CHUNK_SIZE:
        raise ValueError(f"chunk_size must be {CHUNK_SIZE}, got {chunk_size}")
    if k.ndim != 4 or w.ndim != 4 or u.ndim != 4 or g.ndim != 3:
        raise ValueError("k/w/u/g ranks must be 4/4/4/3")
    batch, total_tokens, key_heads, key_dim = k.shape
    value_heads, value_tokens, value_dim = u.shape[1:]
    if batch != 1:
        raise ValueError(f"packed varlen reference expects B=1, got {batch}")
    if w.shape != (batch, value_heads, total_tokens, key_dim):
        raise ValueError(f"unexpected w shape: {tuple(w.shape)}")
    if u.shape != (batch, value_heads, total_tokens, value_dim):
        raise ValueError(f"unexpected u shape: {tuple(u.shape)}")
    if g.shape != (batch, value_heads, total_tokens):
        raise ValueError(f"unexpected g shape: {tuple(g.shape)}")
    if value_tokens != total_tokens:
        raise ValueError("k and u must have the same T")
    if key_dim != value_dim:
        raise ValueError("the challenge requires K == V")
    if value_heads % key_heads != 0:
        raise ValueError("HV must be divisible by HK")
    if k.dtype != torch.bfloat16 or w.dtype != torch.bfloat16 or u.dtype != torch.bfloat16:
        raise TypeError("stage-1 golden cases use BF16 k/w/u")
    if g.dtype != torch.float32:
        raise TypeError("g must be float32")
    if cu_seqlens.dtype != torch.int64 or chunk_indices.dtype != torch.int64:
        raise TypeError("cu_seqlens and chunk_indices must be int64")
    if cu_seqlens.ndim != 1 or int(cu_seqlens[0]) != 0:
        raise ValueError("cu_seqlens must be one-dimensional and start at zero")
    if int(cu_seqlens[-1]) != total_tokens:
        raise ValueError("cu_seqlens[-1] must equal T")
    if not bool(torch.all(cu_seqlens[1:] > cu_seqlens[:-1])):
        raise ValueError("all packed sequence lengths must be positive")
    expected_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if not torch.equal(chunk_indices, expected_indices):
        raise ValueError("chunk_indices does not match cu_seqlens")
    sequence_count = cu_seqlens.numel() - 1
    if initial_state is not None and initial_state.shape != (
        sequence_count,
        value_heads,
        key_dim,
        value_dim,
    ):
        raise ValueError(f"unexpected initial_state shape: {tuple(initial_state.shape)}")


def run_reference(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference matching the explicit cast points in problem.txt section 3.1."""
    validate_inputs(k, w, u, g, initial_state, cu_seqlens, chunk_indices, chunk_size)

    k_by_head = k.transpose(1, 2).contiguous()  # [1, HK, T, K]
    batch, key_heads, total_tokens, key_dim = k_by_head.shape
    value_heads, value_dim = u.shape[1], u.shape[3]
    sequence_count = cu_seqlens.numel() - 1
    chunk_count = chunk_indices.shape[0]
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, chunk_size)
    head_ratio = value_heads // key_heads

    h_output = torch.zeros(
        (batch, value_heads, chunk_count, key_dim, value_dim), dtype=k.dtype
    )
    v_output = torch.zeros((batch, value_heads, total_tokens, value_dim), dtype=u.dtype)
    final_state = torch.zeros(
        (sequence_count, value_heads, key_dim, value_dim), dtype=u.dtype
    )

    for sequence in range(sequence_count):
        bos = int(cu_seqlens[sequence])
        eos = int(cu_seqlens[sequence + 1])
        sequence_chunks = ceil_div(eos - bos, chunk_size)
        chunk_base = int(chunk_offsets[sequence])

        for head in range(value_heads):
            if initial_state is None:
                state = torch.zeros((key_dim, value_dim), dtype=k.dtype)
            else:
                # The problem reference reshapes through FP32 and stores into BF16 S.
                state = initial_state[sequence, head].to(torch.float32).to(k.dtype)

            for chunk in range(sequence_chunks):
                token_start = bos + chunk * chunk_size
                token_end = min(token_start + chunk_size, eos)
                actual_len = token_end - token_start
                global_chunk = chunk_base + chunk

                h_output[0, head, global_chunk] = state

                k_sel = k_by_head[
                    0, head // head_ratio, token_start:token_end, :
                ]
                w_sel = w[0, head, token_start:token_end, :]
                u_sel = u[0, head, token_start:token_end, :]
                g_sel = g[0, head, token_start:token_end]

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
                    update.to(torch.float16),
                    nan=0.0,
                    posinf=torch.inf,
                    neginf=-torch.inf,
                )
                state = (
                    h_decay_fp16.to(torch.bfloat16)
                    + update_fp16.to(torch.bfloat16)
                ).to(k.dtype)

                v_output[0, head, token_start:token_end, :] = v_new_fp16.to(u.dtype)

            final_state[sequence, head] = state.to(u.dtype)

    return h_output, v_output, final_state


def build_gate(
    seq_lens: Iterable[int], value_heads: int, chunk_size: int = CHUNK_SIZE
) -> torch.Tensor:
    """Build negative, strictly decreasing gates independently in every chunk."""
    total_tokens = sum(seq_lens)
    gate = torch.empty((1, value_heads, total_tokens), dtype=torch.float32)
    token_base = 0
    for seq_len in seq_lens:
        for chunk_start in range(0, seq_len, chunk_size):
            actual_len = min(chunk_size, seq_len - chunk_start)
            positions = torch.arange(1, actual_len + 1, dtype=torch.float32)
            for head in range(value_heads):
                rate = 0.01 + 0.0025 * (head % 5)
                begin = token_base + chunk_start
                gate[0, head, begin : begin + actual_len] = -rate * positions
        token_base += seq_len
    return gate


def build_case(spec: CaseSpec) -> dict[str, torch.Tensor | None]:
    if spec.hv % spec.hk != 0:
        raise ValueError(f"invalid case {spec.name}: HV must be divisible by HK")
    if not 1 <= len(spec.seq_lens) <= 4:
        raise ValueError(f"invalid case {spec.name}: token_batch must be in [1, 4]")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(spec.seed)
    total_tokens = sum(spec.seq_lens)

    def uniform(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.rand(shape, generator=generator, dtype=torch.float32) * 2.0 - 1.0

    k = uniform((1, total_tokens, spec.hk, spec.dim)).to(torch.bfloat16)
    w = (uniform((1, spec.hv, total_tokens, spec.dim)) / spec.dim).to(
        torch.bfloat16
    )
    u = uniform((1, spec.hv, total_tokens, spec.dim)).to(torch.bfloat16)
    g = build_gate(spec.seq_lens, spec.hv)
    initial_state = None
    if spec.with_initial_state:
        initial_state = (
            torch.randn(
                (len(spec.seq_lens), spec.hv, spec.dim, spec.dim),
                generator=generator,
                dtype=torch.float32,
            )
            * 0.25
        ).to(torch.bfloat16)

    cu_values = [0]
    for seq_len in spec.seq_lens:
        cu_values.append(cu_values[-1] + seq_len)
    cu_seqlens = torch.tensor(cu_values, dtype=torch.int64)
    chunk_indices = prepare_chunk_indices(cu_seqlens)

    return {
        "k": k,
        "w": w,
        "u": u,
        "g": g,
        "initial_state": initial_state,
        "cu_seqlens": cu_seqlens,
        "chunk_indices": chunk_indices,
    }


def assert_chunk_gates_are_valid(
    g: torch.Tensor, cu_seqlens: torch.Tensor, chunk_size: int = CHUNK_SIZE
) -> None:
    for sequence in range(cu_seqlens.numel() - 1):
        bos = int(cu_seqlens[sequence])
        eos = int(cu_seqlens[sequence + 1])
        for token_start in range(bos, eos, chunk_size):
            token_end = min(token_start + chunk_size, eos)
            chunk_gate = g[:, :, token_start:token_end]
            if not bool(torch.all(chunk_gate < 0)):
                raise AssertionError("g must be negative")
            if chunk_gate.shape[-1] > 1 and not bool(
                torch.all(chunk_gate[..., 1:] < chunk_gate[..., :-1])
            ):
                raise AssertionError("g must be strictly decreasing within each chunk")


def validate_outputs(
    spec: CaseSpec,
    inputs: dict[str, torch.Tensor | None],
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    h_output, v_output, final_state = outputs
    cu_seqlens = inputs["cu_seqlens"]
    chunk_indices = inputs["chunk_indices"]
    initial_state = inputs["initial_state"]
    g = inputs["g"]
    assert isinstance(cu_seqlens, torch.Tensor)
    assert isinstance(chunk_indices, torch.Tensor)
    assert isinstance(g, torch.Tensor)

    total_tokens = sum(spec.seq_lens)
    chunk_count = sum(ceil_div(length, CHUNK_SIZE) for length in spec.seq_lens)
    expected_shapes = (
        (1, spec.hv, chunk_count, spec.dim, spec.dim),
        (1, spec.hv, total_tokens, spec.dim),
        (len(spec.seq_lens), spec.hv, spec.dim, spec.dim),
    )
    for tensor, expected_shape in zip(outputs, expected_shapes):
        if tuple(tensor.shape) != expected_shape:
            raise AssertionError(
                f"{spec.name}: expected {expected_shape}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.bfloat16:
            raise AssertionError(f"{spec.name}: output must be BF16")
        if not bool(torch.all(torch.isfinite(tensor))):
            raise AssertionError(f"{spec.name}: generated output contains NaN/Inf")

    assert_chunk_gates_are_valid(g, cu_seqlens)
    expected_indices = prepare_chunk_indices(cu_seqlens)
    if not torch.equal(chunk_indices, expected_indices):
        raise AssertionError(f"{spec.name}: invalid chunk_indices")

    chunk_offsets = prepare_chunk_offsets(cu_seqlens)
    for sequence in range(len(spec.seq_lens)):
        first_chunk = int(chunk_offsets[sequence])
        expected_initial = (
            torch.zeros((spec.hv, spec.dim, spec.dim), dtype=torch.bfloat16)
            if initial_state is None
            else initial_state[sequence]
        )
        if not torch.equal(h_output[0, :, first_chunk], expected_initial):
            raise AssertionError(
                f"{spec.name}: first h state does not match sequence initial_state"
            )

    # The output covers every packed token exactly once and must not remain all zero.
    if v_output.numel() and not bool(torch.any(v_output != 0)):
        raise AssertionError(f"{spec.name}: v output was not populated")
    if final_state.numel() and not bool(torch.any(final_state != 0)):
        raise AssertionError(f"{spec.name}: final_state output was not populated")


def tensor_to_numpy_storage(tensor: torch.Tensor) -> np.ndarray:
    contiguous = tensor.detach().cpu().contiguous()
    if contiguous.dtype == torch.bfloat16:
        return contiguous.view(torch.uint16).numpy()
    return contiguous.numpy()


def write_tensor(path: Path, tensor: torch.Tensor) -> dict[str, object]:
    storage = tensor_to_numpy_storage(tensor)
    payload = storage.tobytes(order="C")
    path.write_bytes(payload)
    return {
        "file": path.name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "elements": tensor.numel(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def generate_case(spec: CaseSpec, output_dir: Path | None) -> dict[str, object]:
    inputs = build_case(spec)
    outputs = run_reference(
        k=inputs["k"],
        w=inputs["w"],
        u=inputs["u"],
        g=inputs["g"],
        initial_state=inputs["initial_state"],
        cu_seqlens=inputs["cu_seqlens"],
        chunk_indices=inputs["chunk_indices"],
        chunk_size=CHUNK_SIZE,
    )
    validate_outputs(spec, inputs, outputs)

    # A second run verifies that the CPU reference is bitwise reproducible.
    repeated = run_reference(
        k=inputs["k"],
        w=inputs["w"],
        u=inputs["u"],
        g=inputs["g"],
        initial_state=inputs["initial_state"],
        cu_seqlens=inputs["cu_seqlens"],
        chunk_indices=inputs["chunk_indices"],
        chunk_size=CHUNK_SIZE,
    )
    if not all(torch.equal(first, second) for first, second in zip(outputs, repeated)):
        raise AssertionError(f"{spec.name}: reference is not bitwise deterministic")

    summary: dict[str, object] = {
        "case": asdict(spec),
        "chunk_size": CHUNK_SIZE,
        "total_tokens": sum(spec.seq_lens),
        "total_chunks": int(inputs["chunk_indices"].shape[0]),
    }
    if output_dir is None:
        return summary

    case_dir = output_dir / spec.name
    case_dir.mkdir(parents=True, exist_ok=True)
    tensor_manifest: dict[str, dict[str, object]] = {}
    for name, tensor in inputs.items():
        if tensor is not None:
            tensor_manifest[f"input_{name}"] = write_tensor(
                case_dir / f"input_{name}.bin", tensor
            )
    for name, tensor in zip(("h", "v", "final_state"), outputs):
        tensor_manifest[f"golden_{name}"] = write_tensor(
            case_dir / f"golden_{name}.bin", tensor
        )

    summary["tensors"] = tensor_manifest
    manifest_path = case_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary




STRESS_SEQ_LEN_POOL = (1, 2, 5, 63, 64, 65, 127, 128, 129, 200, 255, 256, 511, 512, 1023, 1024, 2048, 4096)
STRESS_HV_POOL = (1, 2, 4, 8, 16, 32)
STRESS_HK_POOL = (1, 2, 4, 8)
STRESS_DIM_POOL = (16, 32, 64, 128, 256)
MAX_STRESS_H_ELEMENTS = 24_000_000  # cap the h output size (~48 MB BF16) per case


def build_stress_specs(count: int, seed: int) -> list[CaseSpec]:
    """Deterministically sample a diverse grid of legal shapes."""
    rng = random.Random(seed)
    specs: list[CaseSpec] = []
    attempt = 0
    while len(specs) < count and attempt < 2000:
        attempt += 1
        dim = rng.choice(STRESS_DIM_POOL)
        hv = rng.choice(STRESS_HV_POOL)
        hk = rng.choice([h for h in STRESS_HK_POOL if hv % h == 0])
        layout = rng.choice(("single", "packed"))
        if layout == "single":
            seq_lens = (rng.choice(STRESS_SEQ_LEN_POOL),)
        else:
            nseq = rng.randint(2, 4)  # build_case caps token_batch at 4
            seq_lens = tuple(rng.choice(STRESS_SEQ_LEN_POOL) for _ in range(nseq))
        chunks = sum((length + CHUNK_SIZE - 1) // CHUNK_SIZE for length in seq_lens)
        h_elements = hv * chunks * dim * dim
        if h_elements > MAX_STRESS_H_ELEMENTS:
            # Scale heads down until the case fits; keeps the layout informative.
            while hv > 1 and h_elements > MAX_STRESS_H_ELEMENTS:
                hv //= 2
                hk = min(hk, hv)
                while hk > 1 and hv % hk != 0:
                    hk //= 2
                h_elements = hv * chunks * dim * dim
        with_state = rng.random() < 0.7
        seed_i = (seed * 1_000_003 + len(specs) * 7919 + attempt * 13) & 0x7FFFFFFF
        specs.append(
            CaseSpec(
                name=f"stress_{len(specs):03d}",
                seq_lens=seq_lens,
                hv=hv,
                hk=hk,
                dim=dim,
                seed=seed_i,
                description=f"Stress sample: layout={layout}, T={sum(seq_lens)}, HV/HK={hv}/{hk}, D={dim}.",
                with_initial_state=with_state,
            )
        )
    return specs


def select_cases(names: list[str] | None) -> list[CaseSpec]:
    if not names or "all" in names:
        return list(CASES)
    by_name = {case.name: case for case in CASES}
    return [by_name[name] for name in names]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        choices=["all", *(case.name for case in CASES)],
        help="case to generate; repeat the option to select multiple cases",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "generated",
        help="root directory for generated case folders",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="run all reference and invariant checks without writing files",
    )
    parser.add_argument(
        "--list-cases", action="store_true", help="list available cases and exit"
    )
    parser.add_argument(
        "--stress",
        type=int,
        default=0,
        metavar="N",
        help="additionally generate N deterministic random stress cases",
    )
    parser.add_argument(
        "--stress-seed",
        type=int,
        default=20260817,
        help="seed for the deterministic stress-case sampler",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list_cases:
        for case in CASES:
            print(
                f"{case.name}: seq_lens={list(case.seq_lens)}, "
                f"HV/HK={case.hv}/{case.hk}, K=V={case.dim} -- {case.description}"
            )
        return 0

    if args.stress > 0 and not args.case:
        # A bare --stress run only materializes the sampled cases so the
        # server does not regenerate every named case on each iteration.
        selected = []
    else:
        selected = list(select_cases(args.case))
    if args.stress > 0:
        selected.extend(build_stress_specs(args.stress, args.stress_seed))
    output_dir = None if args.validate_only else args.output_dir.resolve()
    for spec in selected:
        summary = generate_case(spec, output_dir)
        print(
            f"PASS {spec.name}: T={summary['total_tokens']}, "
            f"NT={summary['total_chunks']}, HV/HK={spec.hv}/{spec.hk}, K=V={spec.dim}"
        )
    if output_dir is None:
        print(f"Validated {len(selected)} case(s); no files written.")
    else:
        if args.stress > 0:
            specs_path = output_dir / "stress_cases.json"
            specs_path.write_text(
                json.dumps(
                    [asdict(spec) for spec in selected[-args.stress:]],
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"Wrote stress specs to {specs_path}")
        if not selected:
            print("No cases selected (use --case NAME or --stress N).")
            return 1
        print(f"Generated {len(selected)} case(s) under {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
