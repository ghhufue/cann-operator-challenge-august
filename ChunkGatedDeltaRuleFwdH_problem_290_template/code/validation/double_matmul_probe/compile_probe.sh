#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${CODE_DIR}/build"
PROBE_BUILD_DIR="${SCRIPT_DIR}/build"
CANN_ROOT="${ASCEND_HOME_PATH:-/home/workspace/hvm/Ascend/cann-9.1.0}"

if [[ ! -f "${CANN_ROOT}/set_env.sh" ]]; then
    echo "[ERROR] CANN set_env.sh not found under ${CANN_ROOT}" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${CANN_ROOT}/set_env.sh" >/dev/null 2>&1

TILING_LIB="${BUILD_DIR}/op_host/libcust_opmaster_rt2.0.so"
CONFIG_INI="${BUILD_DIR}/autogen/aic-ascend910b-ops-info.ini"
OPS_INFO_JSON="${BUILD_DIR}/op_kernel/ascendc_kernels/tbe/op_info_cfg/ai_core/ascend910b/aic-ascend910b-ops-info.json"

if [[ ! -f "${TILING_LIB}" || ! -f "${CONFIG_INI}" || ! -f "${OPS_INFO_JSON}" ]]; then
    echo "[INFO] Preparing generated Host metadata used by the compile probe..."
    cmake -S "${CODE_DIR}" -B "${BUILD_DIR}" -DASCEND_COMPUTE_UNIT=ascend910b
    cmake --build "${BUILD_DIR}" --target \
        chunk_gated_delta_rule_fwd_h_custom_ascendc_cust_optiling \
        ascendc_kernels_ops_info_gen_ascend910b -- -j2
fi

mkdir -p "${PROBE_BUILD_DIR}/binary" "${PROBE_BUILD_DIR}/dynamic"

COMPILE_DRIVER="${CANN_ROOT}/compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/util/ascendc_compile_kernel.py"
if [[ ! -f "${COMPILE_DRIVER}" ]]; then
    echo "[ERROR] AscendC compile driver not found: ${COMPILE_DRIVER}" >&2
    exit 1
fi

echo "[INFO] Compiling fixed-shape two-Matmul MIX probe for Ascend910B..."
pushd "${PROBE_BUILD_DIR}" >/dev/null
python3 "${COMPILE_DRIVER}" \
    --op-type=ChunkGatedDeltaRuleFwdH \
    --src-file="${SCRIPT_DIR}/chunk_gated_delta_rule_fwd_h.cpp" \
    --compute-unit=ascend910b \
    --compile-options="" \
    --debug-config="" \
    --config-ini="${CONFIG_INI}" \
    --tiling-lib="${TILING_LIB}" \
    --output-path="${PROBE_BUILD_DIR}/binary" \
    --dynamic-dir="${PROBE_BUILD_DIR}/dynamic" \
    --enable-binary=TRUE \
    --json-file="${OPS_INFO_JSON}" \
    --target-name=double_matmul_probe \
    --auto-gen-path="${BUILD_DIR}/autogen" \
    --build-tool=make
popd >/dev/null

PROBE_OBJECT="$(find "${PROBE_BUILD_DIR}/binary/ascend910b" -name '*.o' -print -quit 2>/dev/null || true)"
if [[ -z "${PROBE_OBJECT}" ]]; then
    echo "[ERROR] Compile driver finished without producing a probe object." >&2
    exit 1
fi

echo "[PASS] Two-Matmul MIX probe compiled successfully."
echo "[INFO] Object: ${PROBE_OBJECT}"
