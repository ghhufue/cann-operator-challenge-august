#!/bin/bash
# End-to-end real-NPU numerical validation for ChunkGatedDeltaRuleFwdH.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
OP_BUILD_DIR="${PROJECT_DIR}/build"
GOLDEN_GENERATOR="${PROJECT_DIR}/tests/golden/generate_golden.py"
DATA_DIR="${SCRIPT_DIR}/npu_validation_data"
COMPARE_SCRIPT="${SCRIPT_DIR}/compare_npu_output.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE_ID="${NPU_DEVICE_ID:-0}"

echo "========================================"
echo "ChunkGatedDeltaRuleFwdH real-NPU validation"
echo "========================================"

if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    export ASCEND_HOME_PATH=/usr/local/Ascend/cann
fi
if [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    # Load the runtime/driver search paths supplied by the active CANN kit.
    # Do not use CANN devlib here: it may contain link-time stubs that cannot
    # initialize a real device.
    source "${ASCEND_HOME_PATH}/set_env.sh" >/dev/null
fi

MACHINE_ARCH="$(uname -m)"
if [ -d "${ASCEND_HOME_PATH}/${MACHINE_ARCH}-linux" ]; then
    CANN_ARCH_DIR="${ASCEND_HOME_PATH}/${MACHINE_ARCH}-linux"
elif [ "${MACHINE_ARCH}" = "x86_64" ] && [ -d "${ASCEND_HOME_PATH}/x86_64-linux" ]; then
    CANN_ARCH_DIR="${ASCEND_HOME_PATH}/x86_64-linux"
elif [ -d "${ASCEND_HOME_PATH}/aarch64-linux" ]; then
    CANN_ARCH_DIR="${ASCEND_HOME_PATH}/aarch64-linux"
else
    CANN_ARCH_DIR="${ASCEND_HOME_PATH}"
fi

CURRENT_VENDOR_DIR="${OP_BUILD_DIR}/packages/vendors/chunk_gated_delta_rule_fwd_h_custom"
if [ ! -d "${CURRENT_VENDOR_DIR}" ]; then
    echo "[ERROR] Current operator build was not found: ${CURRENT_VENDOR_DIR}"
    echo "[ERROR] Run 'bash build.sh -j8' before real-NPU validation."
    exit 1
fi

export ASCEND_CUSTOM_OPP_PATH="${CURRENT_VENDOR_DIR}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"

DRIVER_LIBRARY_PATH=""
for driver_dir in \
    /usr/local/Ascend/driver/lib64/common \
    /usr/local/Ascend/driver/lib64/driver \
    /usr/local/Ascend/driver/lib64; do
    if [ -d "${driver_dir}" ]; then
        DRIVER_LIBRARY_PATH="${DRIVER_LIBRARY_PATH:+${DRIVER_LIBRARY_PATH}:}${driver_dir}"
    fi
done
export LD_LIBRARY_PATH="${OP_BUILD_DIR}/op_host:${CURRENT_VENDOR_DIR}/op_api/lib${DRIVER_LIBRARY_PATH:+:${DRIVER_LIBRARY_PATH}}:${ASCEND_HOME_PATH}/lib64:${CANN_ARCH_DIR}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if [ -n "${VALIDATION_CASES:-}" ]; then
    IFS=',' read -r -a CASE_NAMES <<< "${VALIDATION_CASES}"
else
    mapfile -t CASE_NAMES < <("${PYTHON_BIN}" "${GOLDEN_GENERATOR}" --list-cases | cut -d: -f1)
fi

if [ "${#CASE_NAMES[@]}" -eq 0 ]; then
    echo "[ERROR] No validation cases were selected."
    exit 1
fi

GENERATOR_ARGS=()
for case_name in "${CASE_NAMES[@]}"; do
    GENERATOR_ARGS+=(--case "${case_name}")
done

echo "[INFO] Logical NPU device: ${DEVICE_ID}"
echo "[INFO] Current OPP: ${CURRENT_VENDOR_DIR}"
echo "[INFO] Cases: ${CASE_NAMES[*]}"
echo "[INFO] Generating legal inputs and Golden outputs..."
"${PYTHON_BIN}" "${GOLDEN_GENERATOR}" \
    --output-dir "${DATA_DIR}" "${GENERATOR_ARGS[@]}"

echo "[INFO] Building ACLNN validation runner..."
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --parallel "$(nproc)"
RUNNER="${BUILD_DIR}/bin/test_aclnn_chunk_gated_delta_rule_fwd_h"

MISSING_LIBRARIES="$(ldd "${RUNNER}" | awk '/not found/ {print $1}' | sort -u | tr '\n' ' ')"
if [ -n "${MISSING_LIBRARIES}" ]; then
    echo "[ERROR] Validation runner has unresolved runtime libraries: ${MISSING_LIBRARIES}"
    echo "[ERROR] Check the CANN set_env.sh and real Ascend driver library mounts."
    exit 1
fi

PASSED_CASES=()
FAILED_CASES=()
for case_name in "${CASE_NAMES[@]}"; do
    case_dir="${DATA_DIR}/${case_name}"
    echo "----------------------------------------"
    echo "[INFO] Running ${case_name} on NPU..."
    if ! "${RUNNER}" "${case_dir}" "${DEVICE_ID}"; then
        echo "[FAILED] ${case_name}: ACLNN execution failed"
        FAILED_CASES+=("${case_name}:execution")
        continue
    fi
    if "${PYTHON_BIN}" "${COMPARE_SCRIPT}" "${case_dir}"; then
        echo "[PASSED] ${case_name}"
        PASSED_CASES+=("${case_name}")
    else
        echo "[FAILED] ${case_name}: numerical comparison failed"
        FAILED_CASES+=("${case_name}:precision")
    fi
done

echo "========================================"
echo "Real-NPU validation summary"
echo "========================================"
echo "Passed: ${#PASSED_CASES[@]} ${PASSED_CASES[*]:-}"
echo "Failed: ${#FAILED_CASES[@]} ${FAILED_CASES[*]:-}"

if [ "${#FAILED_CASES[@]}" -ne 0 ]; then
    exit 1
fi
