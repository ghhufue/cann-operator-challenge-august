#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "chunk_gated_delta_rule_fwd_h_tiling_data.h"

namespace ChunkGatedDeltaRuleFwdHUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "ChunkGatedDeltaRuleFwdH";

struct ChunkGatedDeltaRuleFwdHTestParam {
    std::string caseName;
    std::initializer_list<int64_t> kShape;
    ge::DataType kDtype;
    ge::Format kFormat;
    std::initializer_list<int64_t> wShape;
    ge::DataType wDtype;
    ge::Format wFormat;
    std::initializer_list<int64_t> uShape;
    ge::DataType uDtype;
    ge::Format uFormat;
    std::initializer_list<int64_t> gShape;
    ge::DataType gDtype;
    ge::Format gFormat;
    std::initializer_list<int64_t> initial_stateShape;
    ge::DataType initial_stateDtype;
    ge::Format initial_stateFormat;
    std::initializer_list<int64_t> cu_seqlensShape;
    ge::DataType cu_seqlensDtype;
    ge::Format cu_seqlensFormat;
    std::initializer_list<int64_t> chunk_indicesShape;
    ge::DataType chunk_indicesDtype;
    ge::Format chunk_indicesFormat;
    std::initializer_list<int64_t> hShape;
    ge::DataType hDtype;
    ge::Format hFormat;
    std::initializer_list<int64_t> vShape;
    ge::DataType vDtype;
    ge::Format vFormat;
    std::initializer_list<int64_t> final_stateShape;
    ge::DataType final_stateDtype;
    ge::Format final_stateFormat;
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    uint64_t tilingDataMaxSize;
};

// TODO: 以下期望值基于初始模板实现，修改 tiling 逻辑后请更新：
//   expectTilingKey:  参考 op_kernel/chunk_gated_delta_rule_fwd_h_tiling_key.h 和 op_host/chunk_gated_delta_rule_fwd_h_tiling.cpp 中 tilingKey 的逻辑
//   expectTilingData: 参考 op_host/chunk_gated_delta_rule_fwd_h_tiling.cpp 中 TilingData 各字段的赋值
//   expectWorkspaces: 参考 op_host/chunk_gated_delta_rule_fwd_h_tiling.cpp 中 GetWorkspaceSize 的逻辑
static ChunkGatedDeltaRuleFwdHTestParam testCases[] = {
    {"chunk_gated_delta_rule_fwd_h_0", {1, 10016, 2, 128}, ge::DT_BF16, ge::FORMAT_ND, {1, 8, 10016, 128}, ge::DT_BF16, ge::FORMAT_ND, {1, 8, 10016, 128}, ge::DT_BF16, ge::FORMAT_ND, {1, 8, 10016}, ge::DT_FLOAT, ge::FORMAT_ND, {2, 8, 128, 128}, ge::DT_BF16, ge::FORMAT_ND, {3}, ge::DT_INT64, ge::FORMAT_ND, {158, 2}, ge::DT_INT64, ge::FORMAT_ND, {1, 8, 158, 128, 128}, ge::DT_BF16, ge::FORMAT_ND, {1, 8, 10016, 128}, ge::DT_BF16, ge::FORMAT_ND, {2, 8, 128, 128}, ge::DT_BF16, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 0UL, "0 1 0 ", {0}, 64, 262144, 4096},
};

class ChunkGatedDeltaRuleFwdHTilingTest : public testing::TestWithParam<ChunkGatedDeltaRuleFwdHTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "ChunkGatedDeltaRuleFwdHTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "ChunkGatedDeltaRuleFwdHTilingTest TearDown." << std::endl;
    }
};

struct ChunkGatedDeltaRuleFwdHCompileInfo {} compileInfo;

static void TestOneParamCase(const ChunkGatedDeltaRuleFwdHTestParam &param)
{
    gert::StorageShape kShape = {param.kShape, param.kShape};
    gert::StorageShape wShape = {param.wShape, param.wShape};
    gert::StorageShape uShape = {param.uShape, param.uShape};
    gert::StorageShape gShape = {param.gShape, param.gShape};
    gert::StorageShape initial_stateShape = {param.initial_stateShape, param.initial_stateShape};
    gert::StorageShape cu_seqlensShape = {param.cu_seqlensShape, param.cu_seqlensShape};
    gert::StorageShape chunk_indicesShape = {param.chunk_indicesShape, param.chunk_indicesShape};
    gert::StorageShape hShape = {param.hShape, param.hShape};
    gert::StorageShape vShape = {param.vShape, param.vShape};
    gert::StorageShape final_stateShape = {param.final_stateShape, param.final_stateShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{kShape, param.kDtype, param.kFormat},
        {wShape, param.wDtype, param.wFormat},
        {uShape, param.uDtype, param.uFormat},
        {gShape, param.gDtype, param.gFormat},
        {initial_stateShape, param.initial_stateDtype, param.initial_stateFormat},
        {cu_seqlensShape, param.cu_seqlensDtype, param.cu_seqlensFormat},
        {chunk_indicesShape, param.chunk_indicesDtype, param.chunk_indicesFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{hShape, param.hDtype, param.hFormat},
        {vShape, param.vDtype, param.vFormat},
        {final_stateShape, param.final_stateDtype, param.final_stateFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    attrs_.push_back(gert::TilingContextPara::OpAttr("chunk_size", Ops::Math::AnyValue::CreateFrom<int64_t>(64)));
    gert::TilingContextPara tilingContextPara(
        OP_NAME,
        inputTensorDesc_,
        outputTensorDesc_,
        attrs_,
        &compileInfo,
        param.maxAIVNum,
        param.ubSize,
        param.tilingDataMaxSize);
    ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey,
                    param.expectTilingData, param.expectWorkspaces);
}

TEST_P(ChunkGatedDeltaRuleFwdHTilingTest, tiling_test)
{
    const ChunkGatedDeltaRuleFwdHTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    ChunkGatedDeltaRuleFwdHTilingTests,
    ChunkGatedDeltaRuleFwdHTilingTest,
    testing::ValuesIn(testCases));

}
