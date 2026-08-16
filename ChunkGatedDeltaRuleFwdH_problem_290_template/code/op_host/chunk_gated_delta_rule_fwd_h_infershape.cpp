/*!
 * \file chunk_gated_delta_rule_fwd_h_infershape.cpp
 * \brief ChunkGatedDeltaRuleFwdH shape inference and static validation
 */

#include <cstddef>
#include <cstdint>

#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
namespace {

constexpr size_t K_INPUT = 0;
constexpr size_t W_INPUT = 1;
constexpr size_t U_INPUT = 2;
constexpr size_t G_INPUT = 3;
constexpr size_t INITIAL_STATE_INPUT = 4;
constexpr size_t CU_SEQLENS_INPUT = 5;
constexpr size_t CHUNK_INDICES_INPUT = 6;

constexpr size_t H_OUTPUT = 0;
constexpr size_t V_OUTPUT = 1;
constexpr size_t FINAL_STATE_OUTPUT = 2;

constexpr int64_t CHUNK_SIZE = 64;
constexpr int64_t MAX_SEQUENCE_COUNT = 4;
constexpr int64_t MAX_TOTAL_TOKENS = 256 * 1024;
constexpr int64_t MAX_VALUE_HEADS = 128;
constexpr int64_t MAX_KEY_HEADS = 32;
constexpr int64_t MIN_HEAD_DIM = 32;
constexpr int64_t MAX_HEAD_DIM = 256;

const char* GetNodeName(const gert::InferShapeContext* context)
{
    const char* nodeName = context == nullptr ? nullptr : context->GetNodeName();
    return nodeName == nullptr ? "ChunkGatedDeltaRuleFwdH" : nodeName;
}

bool ResolveInputIndex(const gert::InferShapeContext* context, size_t irIndex, size_t& instanceIndex)
{
    const gert::AnchorInstanceInfo* instanceInfo = context->GetIrInputInstanceInfo(irIndex);
    if (instanceInfo == nullptr) {
        // Compatibility fallback for flat contexts which do not carry IR instance metadata.
        instanceIndex = irIndex;
        return context->GetInputShape(instanceIndex) != nullptr;
    }
    if (instanceInfo->GetInstanceNum() == 0) {
        return false;
    }
    instanceIndex = instanceInfo->GetInstanceStart();
    return true;
}

const gert::Shape* GetInputShape(const gert::InferShapeContext* context, size_t irIndex)
{
    size_t instanceIndex = 0;
    if (!ResolveInputIndex(context, irIndex, instanceIndex)) {
        return nullptr;
    }
    return context->GetInputShape(instanceIndex);
}

const gert::CompileTimeTensorDesc* GetInputDesc(const gert::InferShapeContext* context, size_t irIndex)
{
    size_t instanceIndex = 0;
    if (!ResolveInputIndex(context, irIndex, instanceIndex)) {
        return nullptr;
    }
    return context->GetInputDesc(instanceIndex);
}

gert::Shape* GetOutputShape(gert::InferShapeContext* context, size_t irIndex)
{
    const gert::AnchorInstanceInfo* instanceInfo = context->GetIrOutputInstanceInfo(irIndex);
    if (instanceInfo == nullptr) {
        return context->GetOutputShape(irIndex);
    }
    if (instanceInfo->GetInstanceNum() == 0) {
        return nullptr;
    }
    return context->GetOutputShape(instanceInfo->GetInstanceStart());
}

bool CheckRank(
    const gert::InferShapeContext* context, const gert::Shape* shape, size_t expectedRank, const char* inputName)
{
    if (shape == nullptr) {
        OP_LOGE(GetNodeName(context), "Required input %s is missing", inputName);
        return false;
    }
    if (shape->GetDimNum() != expectedRank) {
        OP_LOGE(
            GetNodeName(context),
            "Input %s must have rank %zu, but got %zu",
            inputName,
            expectedRank,
            shape->GetDimNum());
        return false;
    }
    return true;
}

bool CheckDtype(
    const gert::InferShapeContext* context,
    const gert::CompileTimeTensorDesc* desc,
    ge::DataType expectedDtype,
    const char* inputName)
{
    if (desc == nullptr) {
        OP_LOGE(GetNodeName(context), "Compile-time descriptor for input %s is missing", inputName);
        return false;
    }
    if (desc->GetDataType() != expectedDtype) {
        OP_LOGE(
            GetNodeName(context),
            "Input %s has invalid dtype %d, expected %d",
            inputName,
            static_cast<int32_t>(desc->GetDataType()),
            static_cast<int32_t>(expectedDtype));
        return false;
    }
    return true;
}

int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

ge::graphStatus InferShapeChunkGatedDeltaRuleFwdH(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* kShape = GetInputShape(context, K_INPUT);
    const gert::Shape* wShape = GetInputShape(context, W_INPUT);
    const gert::Shape* uShape = GetInputShape(context, U_INPUT);
    const gert::Shape* gShape = GetInputShape(context, G_INPUT);
    const gert::Shape* initialStateShape = GetInputShape(context, INITIAL_STATE_INPUT);
    const gert::Shape* cuSeqlensShape = GetInputShape(context, CU_SEQLENS_INPUT);
    const gert::Shape* chunkIndicesShape = GetInputShape(context, CHUNK_INDICES_INPUT);

    if (!CheckRank(context, kShape, 4, "k") || !CheckRank(context, wShape, 4, "w") ||
        !CheckRank(context, uShape, 4, "u") || !CheckRank(context, gShape, 3, "g")) {
        return ge::GRAPH_FAILED;
    }
    if (initialStateShape != nullptr && !CheckRank(context, initialStateShape, 4, "initial_state")) {
        return ge::GRAPH_FAILED;
    }
    if ((cuSeqlensShape == nullptr) != (chunkIndicesShape == nullptr)) {
        OP_LOGE(GetNodeName(context), "cu_seqlens and chunk_indices must be both present or both absent");
        return ge::GRAPH_FAILED;
    }
    if (cuSeqlensShape != nullptr &&
        (!CheckRank(context, cuSeqlensShape, 1, "cu_seqlens") ||
         !CheckRank(context, chunkIndicesShape, 2, "chunk_indices"))) {
        return ge::GRAPH_FAILED;
    }

    if (!CheckDtype(context, GetInputDesc(context, K_INPUT), ge::DT_BF16, "k") ||
        !CheckDtype(context, GetInputDesc(context, W_INPUT), ge::DT_BF16, "w") ||
        !CheckDtype(context, GetInputDesc(context, U_INPUT), ge::DT_BF16, "u") ||
        !CheckDtype(context, GetInputDesc(context, G_INPUT), ge::DT_FLOAT, "g")) {
        return ge::GRAPH_FAILED;
    }
    if (initialStateShape != nullptr &&
        !CheckDtype(context, GetInputDesc(context, INITIAL_STATE_INPUT), ge::DT_BF16, "initial_state")) {
        return ge::GRAPH_FAILED;
    }
    if (cuSeqlensShape != nullptr &&
        (!CheckDtype(context, GetInputDesc(context, CU_SEQLENS_INPUT), ge::DT_INT64, "cu_seqlens") ||
         !CheckDtype(context, GetInputDesc(context, CHUNK_INDICES_INPUT), ge::DT_INT64, "chunk_indices"))) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* chunkSize = attrs->GetInt(0);
        if (chunkSize != nullptr && *chunkSize != CHUNK_SIZE) {
            OP_LOGE(GetNodeName(context), "chunk_size must be %ld, but got %ld", CHUNK_SIZE, *chunkSize);
            return ge::GRAPH_FAILED;
        }
    }

    const int64_t batch = kShape->GetDim(0);
    const int64_t totalTokens = kShape->GetDim(1);
    const int64_t keyHeads = kShape->GetDim(2);
    const int64_t keyDim = kShape->GetDim(3);
    const int64_t valueHeads = uShape->GetDim(1);
    const int64_t valueDim = uShape->GetDim(3);

    if (batch != 1) {
        OP_LOGE(GetNodeName(context), "B must be 1, but got %ld", batch);
        return ge::GRAPH_FAILED;
    }
    if (totalTokens <= 0 || totalTokens > MAX_TOTAL_TOKENS) {
        OP_LOGE(GetNodeName(context), "T must be in [1, %ld], but got %ld", MAX_TOTAL_TOKENS, totalTokens);
        return ge::GRAPH_FAILED;
    }
    if (keyHeads <= 0 || keyHeads > MAX_KEY_HEADS || valueHeads <= 0 || valueHeads > MAX_VALUE_HEADS) {
        OP_LOGE(GetNodeName(context), "HK/HV are out of range: HK=%ld, HV=%ld", keyHeads, valueHeads);
        return ge::GRAPH_FAILED;
    }
    if (valueHeads % keyHeads != 0) {
        OP_LOGE(GetNodeName(context), "HV must be divisible by HK: HV=%ld, HK=%ld", valueHeads, keyHeads);
        return ge::GRAPH_FAILED;
    }
    if (keyDim < MIN_HEAD_DIM || keyDim > MAX_HEAD_DIM || valueDim != keyDim) {
        OP_LOGE(
            GetNodeName(context),
            "K and V must be equal and in [%ld, %ld], but got K=%ld, V=%ld",
            MIN_HEAD_DIM,
            MAX_HEAD_DIM,
            keyDim,
            valueDim);
        return ge::GRAPH_FAILED;
    }

    if (wShape->GetDim(0) != batch || wShape->GetDim(1) != valueHeads ||
        wShape->GetDim(2) != totalTokens || wShape->GetDim(3) != keyDim) {
        OP_LOGE(GetNodeName(context), "w must have shape [B, HV, T, K]");
        return ge::GRAPH_FAILED;
    }
    if (uShape->GetDim(0) != batch || uShape->GetDim(2) != totalTokens) {
        OP_LOGE(GetNodeName(context), "u must have shape [B, HV, T, V]");
        return ge::GRAPH_FAILED;
    }
    if (gShape->GetDim(0) != batch || gShape->GetDim(1) != valueHeads ||
        gShape->GetDim(2) != totalTokens) {
        OP_LOGE(GetNodeName(context), "g must have shape [B, HV, T]");
        return ge::GRAPH_FAILED;
    }

    int64_t sequenceCount = batch;
    int64_t chunkCount = CeilDiv(totalTokens, CHUNK_SIZE);
    if (cuSeqlensShape != nullptr) {
        const int64_t cuSeqlensLength = cuSeqlensShape->GetDim(0);
        if (cuSeqlensLength < 2 || cuSeqlensLength > MAX_SEQUENCE_COUNT + 1) {
            OP_LOGE(
                GetNodeName(context),
                "cu_seqlens length must be in [2, %ld], but got %ld",
                MAX_SEQUENCE_COUNT + 1,
                cuSeqlensLength);
            return ge::GRAPH_FAILED;
        }
        sequenceCount = cuSeqlensLength - 1;
        if (sequenceCount > totalTokens) {
            OP_LOGE(GetNodeName(context), "The number of non-empty sequences cannot exceed T");
            return ge::GRAPH_FAILED;
        }
        if (chunkIndicesShape->GetDim(1) != 2) {
            OP_LOGE(GetNodeName(context), "chunk_indices must have shape [NT, 2]");
            return ge::GRAPH_FAILED;
        }
        chunkCount = chunkIndicesShape->GetDim(0);
        const int64_t minChunkCount = CeilDiv(totalTokens, CHUNK_SIZE);
        const int64_t maxChunkCount = minChunkCount + sequenceCount - 1;
        if (chunkCount < minChunkCount || chunkCount > maxChunkCount) {
            OP_LOGE(
                GetNodeName(context),
                "NT=%ld is impossible for T=%ld and N=%ld; expected range [%ld, %ld]",
                chunkCount,
                totalTokens,
                sequenceCount,
                minChunkCount,
                maxChunkCount);
            return ge::GRAPH_FAILED;
        }
    }

    if (initialStateShape != nullptr &&
        (initialStateShape->GetDim(0) != sequenceCount || initialStateShape->GetDim(1) != valueHeads ||
         initialStateShape->GetDim(2) != keyDim || initialStateShape->GetDim(3) != valueDim)) {
        OP_LOGE(GetNodeName(context), "initial_state must have shape [N, HV, K, V]");
        return ge::GRAPH_FAILED;
    }

    gert::Shape* hOutputShape = GetOutputShape(context, H_OUTPUT);
    gert::Shape* vOutputShape = GetOutputShape(context, V_OUTPUT);
    if (hOutputShape == nullptr || vOutputShape == nullptr) {
        OP_LOGE(GetNodeName(context), "Required output shape is missing");
        return ge::GRAPH_FAILED;
    }
    *hOutputShape = gert::Shape({batch, valueHeads, chunkCount, keyDim, valueDim});
    *vOutputShape = gert::Shape({batch, valueHeads, totalTokens, valueDim});

    gert::Shape* finalStateOutputShape = GetOutputShape(context, FINAL_STATE_OUTPUT);
    if (finalStateOutputShape != nullptr) {
        *finalStateOutputShape = gert::Shape({sequenceCount, valueHeads, keyDim, valueDim});
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace

IMPL_OP_INFERSHAPE(ChunkGatedDeltaRuleFwdH).InferShape(InferShapeChunkGatedDeltaRuleFwdH);

} // namespace ops
