#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "infershape_case_executor.h"

namespace ChunkGatedDeltaRuleFwdHUT {
namespace {

const std::string OP_NAME = "ChunkGatedDeltaRuleFwdH";
using TensorDescription = gert::InfershapeContextPara::TensorDescription;
using OpAttr = gert::InfershapeContextPara::OpAttr;

gert::StorageShape MakeStorageShape(std::initializer_list<int64_t> dims)
{
    return gert::StorageShape(dims, dims);
}

TensorDescription MakeTensor(std::initializer_list<int64_t> dims, ge::DataType dtype)
{
    return TensorDescription(MakeStorageShape(dims), dtype, ge::FORMAT_ND);
}

gert::InfershapeContextPara MakeVarlenCase(
    bool withInitialState = true, bool withFinalState = true, int64_t chunkSize = 64)
{
    std::vector<TensorDescription> inputs = {
        MakeTensor({1, 10016, 2, 128}, ge::DT_BF16),
        MakeTensor({1, 8, 10016, 128}, ge::DT_BF16),
        MakeTensor({1, 8, 10016, 128}, ge::DT_BF16),
        MakeTensor({1, 8, 10016}, ge::DT_FLOAT),
    };
    std::vector<uint32_t> inputInstanceNum = {1, 1, 1, 1, 0, 1, 1};
    if (withInitialState) {
        inputs.emplace_back(MakeTensor({2, 8, 128, 128}, ge::DT_BF16));
        inputInstanceNum[4] = 1;
    }
    inputs.emplace_back(MakeTensor({3}, ge::DT_INT64));
    inputs.emplace_back(MakeTensor({158, 2}, ge::DT_INT64));

    std::vector<TensorDescription> outputs = {
        MakeTensor({1}, ge::DT_BF16),
        MakeTensor({1}, ge::DT_BF16),
    };
    std::vector<uint32_t> outputInstanceNum = {1, 1, 0};
    if (withFinalState) {
        outputs.emplace_back(MakeTensor({1}, ge::DT_BF16));
        outputInstanceNum[2] = 1;
    }

    std::vector<OpAttr> attrs = {
        OpAttr("chunk_size", Ops::Math::AnyValue::CreateFrom<int64_t>(chunkSize)),
    };
    return gert::InfershapeContextPara(
        OP_NAME, inputs, outputs, attrs, inputInstanceNum, outputInstanceNum);
}

gert::InfershapeContextPara MakeFixedLengthCase()
{
    std::vector<TensorDescription> inputs = {
        MakeTensor({1, 65, 2, 64}, ge::DT_BF16),
        MakeTensor({1, 4, 65, 64}, ge::DT_BF16),
        MakeTensor({1, 4, 65, 64}, ge::DT_BF16),
        MakeTensor({1, 4, 65}, ge::DT_FLOAT),
        MakeTensor({1, 4, 64, 64}, ge::DT_BF16),
    };
    std::vector<TensorDescription> outputs = {
        MakeTensor({1}, ge::DT_BF16),
        MakeTensor({1}, ge::DT_BF16),
        MakeTensor({1}, ge::DT_BF16),
    };
    std::vector<OpAttr> attrs = {
        OpAttr("chunk_size", Ops::Math::AnyValue::CreateFrom<int64_t>(64)),
    };
    return gert::InfershapeContextPara(
        OP_NAME,
        inputs,
        outputs,
        attrs,
        {1, 1, 1, 1, 1, 0, 0},
        {1, 1, 1});
}

gert::InfershapeContextPara MakeCuSeqlensOnlyCase()
{
    std::vector<TensorDescription> inputs = {
        MakeTensor({1, 65, 2, 64}, ge::DT_BF16),
        MakeTensor({1, 4, 65, 64}, ge::DT_BF16),
        MakeTensor({1, 4, 65, 64}, ge::DT_BF16),
        MakeTensor({1, 4, 65}, ge::DT_FLOAT),
        MakeTensor({2}, ge::DT_INT64),
    };
    std::vector<TensorDescription> outputs = {
        MakeTensor({1}, ge::DT_BF16),
        MakeTensor({1}, ge::DT_BF16),
        MakeTensor({1}, ge::DT_BF16),
    };
    std::vector<OpAttr> attrs = {
        OpAttr("chunk_size", Ops::Math::AnyValue::CreateFrom<int64_t>(64)),
    };
    return gert::InfershapeContextPara(
        OP_NAME,
        inputs,
        outputs,
        attrs,
        {1, 1, 1, 1, 0, 1, 0},
        {1, 1, 1});
}

TEST(ChunkGatedDeltaRuleFwdHInfershape, VarlenSuccess)
{
    auto param = MakeVarlenCase();
    ExecuteTestCase(
        param,
        ge::GRAPH_SUCCESS,
        {{1, 8, 158, 128, 128}, {1, 8, 10016, 128}, {2, 8, 128, 128}});
}

TEST(ChunkGatedDeltaRuleFwdHInfershape, OptionalInitialStateAndFinalState)
{
    auto param = MakeVarlenCase(false, false);
    ExecuteTestCase(param, ge::GRAPH_SUCCESS, {{1, 8, 158, 128, 128}, {1, 8, 10016, 128}});
}

TEST(ChunkGatedDeltaRuleFwdHInfershape, FixedLengthSuccess)
{
    auto param = MakeFixedLengthCase();
    ExecuteTestCase(
        param,
        ge::GRAPH_SUCCESS,
        {{1, 4, 2, 64, 64}, {1, 4, 65, 64}, {1, 4, 64, 64}});
}

TEST(ChunkGatedDeltaRuleFwdHInfershape, RejectsInvalidBatch)
{
    auto param = MakeVarlenCase();
    param.inputTensorDesc_[0].shape_ = MakeStorageShape({2, 10016, 2, 128});
    ExecuteTestCase(param, ge::GRAPH_FAILED);
}

TEST(ChunkGatedDeltaRuleFwdHInfershape, RejectsInvalidRank)
{
    auto param = MakeVarlenCase();
    param.inputTensorDesc_[3].shape_ = MakeStorageShape({1, 8, 10016, 1});
    ExecuteTestCase(param, ge::GRAPH_FAILED);
}

TEST(ChunkGatedDeltaRuleFwdHInfershape, RejectsMismatchedKeyAndValueDimensions)
{
    auto param = MakeVarlenCase();
    param.inputTensorDesc_[2].shape_ = MakeStorageShape({1, 8, 10016, 64});
    ExecuteTestCase(param, ge::GRAPH_FAILED);
}

TEST(ChunkGatedDeltaRuleFwdHInfershape, RejectsOutOfRangeHeadDimension)
{
    auto param = MakeVarlenCase();
    param.inputTensorDesc_[0].shape_ = MakeStorageShape({1, 10016, 2, 16});
    param.inputTensorDesc_[1].shape_ = MakeStorageShape({1, 8, 10016, 16});
    param.inputTensorDesc_[2].shape_ = MakeStorageShape({1, 8, 10016, 16});
    param.inputTensorDesc_[4].shape_ = MakeStorageShape({2, 8, 16, 16});
    ExecuteTestCase(param, ge::GRAPH_FAILED);
}

TEST(ChunkGatedDeltaRuleFwdHInfershape, RejectsNonDivisibleHeadRatio)
{
    auto param = MakeVarlenCase();
    param.inputTensorDesc_[1].shape_ = MakeStorageShape({1, 7, 10016, 128});
    param.inputTensorDesc_[2].shape_ = MakeStorageShape({1, 7, 10016, 128});
    param.inputTensorDesc_[3].shape_ = MakeStorageShape({1, 7, 10016});
    param.inputTensorDesc_[4].shape_ = MakeStorageShape({2, 7, 128, 128});
    ExecuteTestCase(param, ge::GRAPH_FAILED);
}

TEST(ChunkGatedDeltaRuleFwdHInfershape, RejectsMismatchedTokenDimension)
{
    auto param = MakeVarlenCase();
    param.inputTensorDesc_[1].shape_ = MakeStorageShape({1, 8, 10015, 128});
    ExecuteTestCase(param, ge::GRAPH_FAILED);
}

TEST(ChunkGatedDeltaRuleFwdHInfershape, RejectsInvalidInitialState)
{
    auto param = MakeVarlenCase();
    param.inputTensorDesc_[4].shape_ = MakeStorageShape({1, 8, 128, 128});
    ExecuteTestCase(param, ge::GRAPH_FAILED);
}

TEST(ChunkGatedDeltaRuleFwdHInfershape, RejectsInvalidChunkIndicesShape)
{
    auto param = MakeVarlenCase();
    param.inputTensorDesc_[6].shape_ = MakeStorageShape({158, 3});
    ExecuteTestCase(param, ge::GRAPH_FAILED);
}

TEST(ChunkGatedDeltaRuleFwdHInfershape, RejectsImpossibleChunkCount)
{
    auto param = MakeVarlenCase();
    param.inputTensorDesc_[6].shape_ = MakeStorageShape({160, 2});
    ExecuteTestCase(param, ge::GRAPH_FAILED);
}

TEST(ChunkGatedDeltaRuleFwdHInfershape, RejectsHalfInput)
{
    auto param = MakeVarlenCase();
    param.inputTensorDesc_[0].dtype_ = ge::DT_FLOAT16;
    ExecuteTestCase(param, ge::GRAPH_FAILED);
}

TEST(ChunkGatedDeltaRuleFwdHInfershape, RejectsUnsupportedChunkSize)
{
    auto param = MakeVarlenCase(true, true, 32);
    ExecuteTestCase(param, ge::GRAPH_FAILED);
}

TEST(ChunkGatedDeltaRuleFwdHInfershape, RejectsOnlyOneVarlenMetadataInput)
{
    auto param = MakeCuSeqlensOnlyCase();
    ExecuteTestCase(param, ge::GRAPH_FAILED);
}

} // namespace
} // namespace ChunkGatedDeltaRuleFwdHUT
