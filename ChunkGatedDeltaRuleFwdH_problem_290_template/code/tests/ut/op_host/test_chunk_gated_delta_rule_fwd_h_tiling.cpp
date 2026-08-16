#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "chunk_gated_delta_rule_fwd_h_tiling_data.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace ChunkGatedDeltaRuleFwdHUT {
namespace {

using TensorDescription = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;

constexpr const char* OP_NAME = "ChunkGatedDeltaRuleFwdH";
constexpr uint64_t UB_SIZE = 262144;
constexpr uint64_t TILING_DATA_SIZE = 4096;

struct ChunkGatedDeltaRuleFwdHCompileInfo {} compileInfo;

TensorDescription Tensor(std::initializer_list<int64_t> dims, ge::DataType dtype)
{
    gert::StorageShape shape = {dims, dims};
    return TensorDescription(shape, dtype, ge::FORMAT_ND);
}

std::vector<OpAttr> DefaultAttrs()
{
    return {OpAttr("chunk_size", Ops::Math::AnyValue::CreateFrom<int64_t>(64))};
}

const ChunkGatedDeltaRuleFwdHTilingData& Decode(const TilingInfo& info)
{
    EXPECT_EQ(info.tilingDataSize, sizeof(ChunkGatedDeltaRuleFwdHTilingData));
    return *reinterpret_cast<const ChunkGatedDeltaRuleFwdHTilingData*>(info.tilingData.get());
}

TEST(ChunkGatedDeltaRuleFwdHTilingTest, VarLenAllOptionalPaths)
{
    std::vector<TensorDescription> inputs = {
        Tensor({1, 10016, 2, 128}, ge::DT_BF16),
        Tensor({1, 8, 10016, 128}, ge::DT_BF16),
        Tensor({1, 8, 10016, 128}, ge::DT_BF16),
        Tensor({1, 8, 10016}, ge::DT_FLOAT),
        Tensor({2, 8, 128, 128}, ge::DT_BF16),
        Tensor({3}, ge::DT_INT64),
        Tensor({158, 2}, ge::DT_INT64),
    };
    std::vector<TensorDescription> outputs = {
        Tensor({1, 8, 158, 128, 128}, ge::DT_BF16),
        Tensor({1, 8, 10016, 128}, ge::DT_BF16),
        Tensor({2, 8, 128, 128}, ge::DT_BF16),
    };
    gert::TilingContextPara context(
        OP_NAME, inputs, outputs, DefaultAttrs(), &compileInfo, 16, UB_SIZE, TILING_DATA_SIZE);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    EXPECT_EQ(info.tilingKey, 7);
    EXPECT_GT(info.blockNum, 0U);
    ASSERT_EQ(info.workspaceSizes.size(), 1U);

    const auto& tiling = Decode(info);
    EXPECT_EQ(tiling.batch, 1);
    EXPECT_EQ(tiling.totalTokens, 10016);
    EXPECT_EQ(tiling.sequenceCount, 2);
    EXPECT_EQ(tiling.chunkCount, 158);
    EXPECT_EQ(tiling.valueHeads, 8);
    EXPECT_EQ(tiling.keyHeads, 2);
    EXPECT_EQ(tiling.headRatio, 4);
    EXPECT_EQ(tiling.keyDim, 128);
    EXPECT_EQ(tiling.valueDim, 128);
    EXPECT_EQ(tiling.vTileSize, 64);
    EXPECT_EQ(tiling.vTileCount, 2);
    EXPECT_EQ(tiling.taskCount, 32);
    EXPECT_EQ(tiling.usedCoreNum, 16);
    EXPECT_EQ(tiling.taskCountPerCore, 2);
    EXPECT_EQ(tiling.taskTailCoreCount, 0);
    EXPECT_EQ(tiling.hasInitialState, 1);
    EXPECT_EQ(tiling.isVarLen, 1);
    EXPECT_EQ(tiling.storeFinalState, 1);
    EXPECT_EQ(tiling.mm1WorkspaceOffset, 0);
    EXPECT_EQ(tiling.mm2WorkspaceOffset, 8192);
    EXPECT_EQ(tiling.perCoreWorkspaceBytes, 24576);
    EXPECT_EQ(
        info.workspaceSizes[0],
        tiling.systemWorkspaceBytes + tiling.usedCoreNum * tiling.perCoreWorkspaceBytes);

    EXPECT_EQ(tiling.mm1Tiling.M, 64);
    EXPECT_EQ(tiling.mm1Tiling.N, 64);
    EXPECT_EQ(tiling.mm1Tiling.Ka, 128);
    EXPECT_EQ(tiling.mm2Tiling.M, 128);
    EXPECT_EQ(tiling.mm2Tiling.N, 64);
    EXPECT_EQ(tiling.mm2Tiling.Ka, 64);
}

TEST(ChunkGatedDeltaRuleFwdHTilingTest, FixedLengthWithoutOptionalStateOrFinal)
{
    std::vector<TensorDescription> inputs = {
        Tensor({1, 65, 2, 64}, ge::DT_BF16),
        Tensor({1, 4, 65, 64}, ge::DT_BF16),
        Tensor({1, 4, 65, 64}, ge::DT_BF16),
        Tensor({1, 4, 65}, ge::DT_FLOAT),
    };
    std::vector<TensorDescription> outputs = {
        Tensor({1, 4, 2, 64, 64}, ge::DT_BF16),
        Tensor({1, 4, 65, 64}, ge::DT_BF16),
    };
    std::vector<uint32_t> inputInstances = {1, 1, 1, 1, 0, 0, 0};
    std::vector<uint32_t> outputInstances = {1, 1, 0};
    gert::TilingContextPara context(
        OP_NAME, inputs, outputs, DefaultAttrs(), inputInstances, outputInstances,
        &compileInfo, 16, UB_SIZE, TILING_DATA_SIZE);

    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(context, info));
    EXPECT_EQ(info.tilingKey, 0);
    EXPECT_GT(info.blockNum, 0U);
    ASSERT_EQ(info.workspaceSizes.size(), 1U);

    const auto& tiling = Decode(info);
    EXPECT_EQ(tiling.sequenceCount, 1);
    EXPECT_EQ(tiling.chunkCount, 2);
    EXPECT_EQ(tiling.vTileSize, 64);
    EXPECT_EQ(tiling.vTileCount, 1);
    EXPECT_EQ(tiling.taskCount, 4);
    EXPECT_EQ(tiling.usedCoreNum, 4);
    EXPECT_EQ(tiling.taskCountPerCore, 1);
    EXPECT_EQ(tiling.taskTailCoreCount, 0);
    EXPECT_EQ(tiling.hasInitialState, 0);
    EXPECT_EQ(tiling.isVarLen, 0);
    EXPECT_EQ(tiling.storeFinalState, 0);
    EXPECT_EQ(tiling.mm2WorkspaceOffset, 8192);
    EXPECT_EQ(tiling.perCoreWorkspaceBytes, 16384);
    EXPECT_EQ(
        info.workspaceSizes[0],
        tiling.systemWorkspaceBytes + tiling.usedCoreNum * tiling.perCoreWorkspaceBytes);
}

TEST(ChunkGatedDeltaRuleFwdHTilingTest, RejectsHalfPresentVarLenMetadata)
{
    std::vector<TensorDescription> inputs = {
        Tensor({1, 64, 1, 64}, ge::DT_BF16),
        Tensor({1, 1, 64, 64}, ge::DT_BF16),
        Tensor({1, 1, 64, 64}, ge::DT_BF16),
        Tensor({1, 1, 64}, ge::DT_FLOAT),
        Tensor({2}, ge::DT_INT64),
    };
    std::vector<TensorDescription> outputs = {
        Tensor({1, 1, 1, 64, 64}, ge::DT_BF16),
        Tensor({1, 1, 64, 64}, ge::DT_BF16),
    };
    std::vector<uint32_t> inputInstances = {1, 1, 1, 1, 0, 1, 0};
    std::vector<uint32_t> outputInstances = {1, 1, 0};
    gert::TilingContextPara context(
        OP_NAME, inputs, outputs, DefaultAttrs(), inputInstances, outputInstances,
        &compileInfo, 16, UB_SIZE, TILING_DATA_SIZE);

    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}

}  // namespace
}  // namespace ChunkGatedDeltaRuleFwdHUT
