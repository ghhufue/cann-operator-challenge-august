# Real-NPU numerical validation

This directory runs `ChunkGatedDeltaRuleFwdH` through ACLNN on a real NPU and
compares all three BF16 outputs with the reference implementation in
`tests/golden/generate_golden.py`.

Build the current operator and run every validation case:

```bash
cd code
bash build.sh -j8
bash build.sh -e
```

Start with the minimal two-chunk recurrence when debugging correctness:

```bash
VALIDATION_CASES=two_chunk_single bash build.sh -e
```

Select several cases with a comma-separated list:

```bash
VALIDATION_CASES=two_chunk_single,two_v_tiles,four_v_tiles bash build.sh -e
```

The runner uses logical device `0` by default. Override it when the process can
see more than one logical device:

```bash
NPU_DEVICE_ID=1 VALIDATION_CASES=two_chunk_single bash build.sh -e
```

Generated inputs, Golden files, and NPU outputs are written under
`examples/npu_validation_data/` and are ignored by Git. A case fails if ACLNN
execution fails or if any element of `h`, `v`, or `final_state` violates both
the absolute tolerance `1e-6` and relative tolerance `1e-5`.
