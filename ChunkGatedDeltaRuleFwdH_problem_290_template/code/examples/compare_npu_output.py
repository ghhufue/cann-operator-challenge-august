#!/usr/bin/env python3
"""Compare real-NPU BF16 outputs with a generated Golden case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16


ABS_TOLERANCE = 1e-6
REL_TOLERANCE = 1e-5
OUTPUTS = (
    ("h", "golden_h", "actual_h.bin"),
    ("v", "golden_v", "actual_v.bin"),
    ("final_state", "golden_final_state", "actual_final_state.bin"),
)


def load_bf16(path: Path, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    expected_elements = int(np.prod(shape, dtype=np.int64))
    bits = np.fromfile(path, dtype=np.uint16)
    if bits.size != expected_elements:
        raise ValueError(
            f"{path}: expected {expected_elements} BF16 elements "
            f"({expected_elements * 2} bytes), got {bits.size} elements "
            f"({bits.nbytes} bytes)"
        )
    values = bits.view(bfloat16).reshape(shape)
    return bits.reshape(shape), values


def format_index(index: tuple[int, ...]) -> str:
    return "[" + ", ".join(str(item) for item in index) + "]"


def compare_output(
    name: str,
    golden_path: Path,
    actual_path: Path,
    shape: tuple[int, ...],
    max_mismatches: int,
    abs_tolerance: float = ABS_TOLERANCE,
    rel_tolerance: float = REL_TOLERANCE,
) -> bool:
    golden_bits, golden_bf16 = load_bf16(golden_path, shape)
    actual_bits, actual_bf16 = load_bf16(actual_path, shape)
    golden = np.asarray(golden_bf16, dtype=np.float32)
    actual = np.asarray(actual_bf16, dtype=np.float32)

    bit_equal = actual_bits == golden_bits
    special_equal = (
        (np.isnan(actual) & np.isnan(golden))
        | ((actual == golden) & np.isinf(actual))
    )
    absolute_error = np.abs(actual - golden)
    relative_error = np.full_like(absolute_error, np.inf, dtype=np.float32)
    nonzero_golden = np.abs(golden) > 0
    np.divide(
        absolute_error,
        np.abs(golden),
        out=relative_error,
        where=nonzero_golden,
    )
    passed = (
        bit_equal
        | special_equal
        | (absolute_error <= abs_tolerance)
        | (relative_error <= rel_tolerance)
    )

    total = passed.size
    failed = int(total - np.count_nonzero(passed))
    bit_mismatch = int(total - np.count_nonzero(bit_equal))
    finite_absolute = absolute_error[np.isfinite(absolute_error)]
    finite_relative = relative_error[np.isfinite(relative_error)]
    max_absolute = float(finite_absolute.max(initial=0.0))
    max_relative = float(finite_relative.max(initial=0.0))

    print(
        f"{name}: {'PASSED' if failed == 0 else 'FAILED'}; "
        f"shape={list(shape)}, failed={failed}/{total} ({failed / total:.6%}), "
        f"bit_mismatch={bit_mismatch}/{total} ({bit_mismatch / total:.6%}), "
        f"max_abs={max_absolute:.9g}, max_rel={max_relative:.9g}"
    )

    if failed != 0:
        flat_failures = np.flatnonzero(~passed.reshape(-1))[:max_mismatches]
        for flat_index in flat_failures:
            index = np.unravel_index(int(flat_index), shape)
            print(
                f"  index={format_index(index)}, "
                f"actual={float(actual[index]):.9g} "
                f"(0x{int(actual_bits[index]):04x}), "
                f"golden={float(golden[index]):.9g} "
                f"(0x{int(golden_bits[index]):04x}), "
                f"abs={float(absolute_error[index]):.9g}, "
                f"rel={float(relative_error[index]):.9g}"
            )
    return failed == 0


def compare_case(
    case_dir: Path,
    max_mismatches: int,
    abs_tolerance: float = ABS_TOLERANCE,
    rel_tolerance: float = REL_TOLERANCE,
) -> bool:
    manifest_path = case_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tensors = manifest["tensors"]
    print(f"Comparing case: {manifest['case']['name']}")

    all_passed = True
    for output_name, golden_key, actual_name in OUTPUTS:
        golden = tensors[golden_key]
        if golden["dtype"] != "bfloat16":
            raise ValueError(f"Unsupported Golden dtype for {golden_key}: {golden['dtype']}")
        shape = tuple(int(dim) for dim in golden["shape"])
        passed = compare_output(
            output_name,
            case_dir / golden["file"],
            case_dir / actual_name,
            shape,
            max_mismatches,
            abs_tolerance,
            rel_tolerance,
        )
        all_passed = all_passed and passed
    return all_passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path, help="generated Golden case directory")
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=8,
        help="maximum number of failing elements printed per output",
    )
    parser.add_argument(
        "--abs-tol",
        type=float,
        default=ABS_TOLERANCE,
        help="absolute tolerance (default: %(default)g)",
    )
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=REL_TOLERANCE,
        help="relative tolerance (default: %(default)g)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        passed = compare_case(
            args.case_dir.resolve(),
            args.max_mismatches,
            args.abs_tol,
            args.rel_tol,
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(f"Comparison failed: {error}")
        return 2
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
