# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import itertools
import os
import json
import logging

import torch
import torch_npu

# ******入参调用
from test_chunk_gated_delta_rule_paramset import ENABLED_PARAMS
from test_chunk_gated_delta_rule_paramset_rdv import ENABLED_PARAMS_RDV

# ******CPU侧算子逻辑实现获取golden与npu算子直调结果
import chunk_gated_delta_rule_operator_single
import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

CUSTOM_CASE = os.environ.get("CUSTOM_CASE", "")

if CUSTOM_CASE:
    _case = json.loads(CUSTOM_CASE)
    _case["data_type"] = _DTYPE_MAP.get(
        _case.get("data_type", "bfloat16"), torch.bfloat16
    )
    _case["state_data_type"] = _DTYPE_MAP.get(
        _case.get("state_data_type", "bfloat16"), torch.bfloat16
    )
    _case.setdefault("_name", "custom")
    _case.setdefault("chunk_size", 64)
    _case.setdefault("has_g", True)
    _case.setdefault("is_contiguous", True)
    _case.setdefault("pt_path", "")
    param_combinations = [_case]
    logger.info(f"CUSTOM_CASE mode: {_case}")
else:
    TEST_MODE = os.environ.get("TEST_MODE", "single")

    if TEST_MODE not in ["single", "rdv"]:
        raise ValueError(f"Invalid TEST_MODE: {TEST_MODE}, must be 'single' or 'rdv'")

    if TEST_MODE == "rdv":
        PARAM_SET = ENABLED_PARAMS_RDV
    else:
        PARAM_SET = ENABLED_PARAMS

    logger.info(f"TEST_MODE: {TEST_MODE}")

    param_combinations = []

    for _, params in enumerate(PARAM_SET):
        param_names = [
            "_name",
            "B",
            "seqlen",
            "nk",
            "nv",
            "dk",
            "dv",
            "chunk_size",
            "data_type",
            "state_data_type",
            "has_g",
            "is_contiguous",
            "pt_path",
        ]

        param_values = [
            params["_name"] if "_name" in params else [""],
            params["B"],
            params["seqlen"],
            params["nk"],
            params["nv"],
            params["dk"],
            params["dv"],
            params["chunk_size"],
            params["data_type"],
            params["state_data_type"],
            params["has_g"],
            params.get("is_contiguous", [True]),
            params.get("pt_path", [""]),
        ]

        for combo in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combo))
            param_combinations.append(param_dict)

logger.info(f"Total test cases: {len(param_combinations)}")


@pytest.mark.ci
@pytest.mark.parametrize("param_combinations", param_combinations)
def test_chunk_gated_delta_rule(param_combinations):
    # 初始化参数和tensor
    B = param_combinations["B"]
    seqlen = param_combinations["seqlen"]
    nk = param_combinations["nk"]
    nv = param_combinations["nv"]
    dk = param_combinations["dk"]
    dv = param_combinations["dv"]
    chunk_size = param_combinations["chunk_size"]
    data_type = param_combinations["data_type"]
    state_data_type = param_combinations["state_data_type"]
    has_g = param_combinations["has_g"]
    is_contiguous = param_combinations["is_contiguous"]
    pt_path = param_combinations.get("pt_path", "")

    test_data = (
        B,
        seqlen,
        nk,
        nv,
        dk,
        dv,
        chunk_size,
        data_type,
        state_data_type,
        has_g,
        is_contiguous,
    )

    torch_npu.npu.set_device(0)

    # 获取cpu结果(真值)和算子结果（测试值)
    chunk_gated_delta_rule_operator_single.run_precision_test(
        test_data, pt_path=pt_path
    )
