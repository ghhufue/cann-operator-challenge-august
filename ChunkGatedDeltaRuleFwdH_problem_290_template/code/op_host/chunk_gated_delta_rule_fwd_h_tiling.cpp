/*!
 * \file chunk_gated_delta_rule_fwd_h_tiling.cpp
 * \brief ChunkGatedDeltaRuleFwdH Host tiling implementation
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "tiling/matmul/matmul_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/chunk_gated_delta_rule_fwd_h_tiling_data.h"
#include "../op_kernel/chunk_gated_delta_rule_fwd_h_tiling_key.h"

namespace optiling {
namespace {

constexpr size_t K_INPUT = 0;
constexpr size_t U_INPUT = 2;
constexpr size_t INITIAL_STATE_INPUT = 4;
constexpr size_t CU_SEQLENS_INPUT = 5;
constexpr size_t CHUNK_INDICES_INPUT = 6;
constexpr size_t FINAL_STATE_OUTPUT = 2;

constexpr int64_t CHUNK_SIZE = 64;
constexpr int64_t DEFAULT_V_TILE = 64;
constexpr int64_t BF16_BYTES = 2;
constexpr int64_t WORKSPACE_ALIGNMENT = 512;

struct PlatformResources {
    platform_ascendc::PlatformAscendC platform;
    uint32_t aicCoreNum = 0;
    uint32_t aivCoreNum = 0;
    uint64_t ubSize = 0;
    uint64_t systemWorkspaceBytes = 0;

    explicit PlatformResources(fe::PlatFormInfos* platformInfo) : platform(platformInfo) {}
};

int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

int64_t AlignUp(int64_t value, int64_t alignment)
{
    return CeilDiv(value, alignment) * alignment;
}

int64_t ComputeDebugWorkspaceBytes(int64_t chunkSize, int64_t vTileSize, int64_t keyDim)
{
    const int64_t wsBytes = chunkSize * vTileSize * BF16_BYTES;
    const int64_t stateBytes = keyDim * vTileSize * BF16_BYTES;
    int64_t total = 0;
    auto addField = [&total](int64_t bytes) {
        total = AlignUp(total, WORKSPACE_ALIGNMENT);
        total += bytes;
    };
    addField(wsBytes);       // ws
    addField(wsBytes);       // v_new_fp16
    addField(chunkSize * BF16_BYTES); // gate_fp16
    addField(wsBytes);       // v_decay_bf16
    addField(stateBytes);    // h_decay_bf16
    addField(stateBytes);    // mm2_bf16
    addField(stateBytes);    // next_h_bf16
    return AlignUp(total, WORKSPACE_ALIGNMENT);
}

bool ResolveInputIndex(const gert::TilingContext* context, size_t irIndex, size_t& instanceIndex)
{
    const gert::AnchorInstanceInfo* instanceInfo = context->GetIrInputInstanceInfo(irIndex);
    if (instanceInfo == nullptr) {
        instanceIndex = irIndex;
        return context->GetInputShape(instanceIndex) != nullptr;
    }
    if (instanceInfo->GetInstanceNum() == 0) {
        return false;
    }
    instanceIndex = instanceInfo->GetInstanceStart();
    return true;
}

const gert::Shape* GetInputShape(const gert::TilingContext* context, size_t irIndex)
{
    size_t instanceIndex = 0;
    if (!ResolveInputIndex(context, irIndex, instanceIndex)) {
        return nullptr;
    }
    const gert::StorageShape* storageShape = context->GetInputShape(instanceIndex);
    return storageShape == nullptr ? nullptr : &storageShape->GetOriginShape();
}

bool HasOutput(const gert::TilingContext* context, size_t irIndex)
{
    const gert::AnchorInstanceInfo* instanceInfo = context->GetIrOutputInstanceInfo(irIndex);
    if (instanceInfo == nullptr) {
        return context->GetOutputShape(irIndex) != nullptr;
    }
    return instanceInfo->GetInstanceNum() != 0;
}

ge::graphStatus GetPlatformResources(gert::TilingContext* context, PlatformResources& resources)
{
    resources.aicCoreNum = resources.platform.GetCoreNumAic();
    resources.aivCoreNum = resources.platform.GetCoreNumAiv();
    resources.platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, resources.ubSize);
    resources.systemWorkspaceBytes = resources.platform.GetLibApiWorkSpaceSize();

    OP_CHECK_IF(
        resources.aicCoreNum == 0 || resources.aivCoreNum == 0,
        OP_LOGE(context, "MIX kernel requires AIC and AIV cores, got AIC=%u, AIV=%u",
                resources.aicCoreNum, resources.aivCoreNum),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(resources.ubSize == 0, OP_LOGE(context, "UB size is zero"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool BuildMatmulTiling(
    const platform_ascendc::PlatformAscendC& platform,
    int32_t m,
    int32_t n,
    int32_t k,
    matmul_tiling::TPosition aPosition,
    bool transposeA,
    AscendC::tiling::TCubeTiling& result)
{
    matmul_tiling::MatmulApiTiling tiling(platform);
    if (tiling.SetAType(
            aPosition,
            matmul_tiling::CubeFormat::ND,
            matmul_tiling::DataType::DT_BF16,
            transposeA) != 0 ||
        tiling.SetBType(
            matmul_tiling::TPosition::VECCALC,
            matmul_tiling::CubeFormat::ND,
            matmul_tiling::DataType::DT_BF16) != 0 ||
        tiling.SetCType(
            matmul_tiling::TPosition::GM,
            matmul_tiling::CubeFormat::ND,
            matmul_tiling::DataType::DT_BF16) != 0 ||
        tiling.SetBias(false) != 0 || tiling.SetShape(m, n, k) != 0 ||
        tiling.SetOrgShape(m, n, k) != 0) {
        return false;
    }
    return tiling.GetTiling(result) == 0;
}

ge::graphStatus SetWorkspace(
    gert::TilingContext* context,
    const PlatformResources& resources,
    ChunkGatedDeltaRuleFwdHTilingData& tiling)
{
    const int64_t mm1Bytes = CHUNK_SIZE * tiling.vTileSize * BF16_BYTES;
    const int64_t mm2Bytes = tiling.keyDim * tiling.vTileSize * BF16_BYTES;
    tiling.mm1WorkspaceOffset = 0;
    tiling.mm2WorkspaceOffset = AlignUp(mm1Bytes, WORKSPACE_ALIGNMENT);
    tiling.perCoreWorkspaceBytes =
        AlignUp(tiling.mm2WorkspaceOffset + mm2Bytes, WORKSPACE_ALIGNMENT);
    tiling.systemWorkspaceBytes = static_cast<int64_t>(resources.systemWorkspaceBytes);
    tiling.debugPerTaskBytes =
        ComputeDebugWorkspaceBytes(tiling.chunkSize, tiling.vTileSize, tiling.keyDim);
    tiling.debugWorkspaceBytes = tiling.debugPerTaskBytes * tiling.taskCount;

    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    const uint64_t userWorkspace =
        static_cast<uint64_t>(tiling.perCoreWorkspaceBytes) * static_cast<uint64_t>(tiling.usedCoreNum);
    tiling.debugWorkspaceOffset = AlignUp(
        static_cast<int64_t>(resources.systemWorkspaceBytes) +
            static_cast<int64_t>(userWorkspace),
        WORKSPACE_ALIGNMENT);
    workspace[0] = static_cast<uint64_t>(tiling.debugWorkspaceOffset) +
        static_cast<uint64_t>(tiling.debugWorkspaceBytes);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleFwdHTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    PlatformResources resources(platformInfo);
    OP_CHECK_IF(
        GetPlatformResources(context, resources) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Failed to query platform resources"),
        return ge::GRAPH_FAILED);

    const gert::Shape* kShape = GetInputShape(context, K_INPUT);
    const gert::Shape* uShape = GetInputShape(context, U_INPUT);
    OP_CHECK_IF(
        kShape == nullptr || uShape == nullptr || kShape->GetDimNum() != 4 || uShape->GetDimNum() != 4,
        OP_LOGE(context, "k and u must be rank-4 tensors"),
        return ge::GRAPH_FAILED);

    const gert::Shape* initialStateShape = GetInputShape(context, INITIAL_STATE_INPUT);
    const gert::Shape* cuSeqlensShape = GetInputShape(context, CU_SEQLENS_INPUT);
    const gert::Shape* chunkIndicesShape = GetInputShape(context, CHUNK_INDICES_INPUT);
    OP_CHECK_IF(
        (cuSeqlensShape == nullptr) != (chunkIndicesShape == nullptr),
        OP_LOGE(context, "cu_seqlens and chunk_indices must be both present or both absent"),
        return ge::GRAPH_FAILED);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* chunkSize = attrs->GetInt(0);
        OP_CHECK_IF(
            chunkSize != nullptr && *chunkSize != CHUNK_SIZE,
            OP_LOGE(context, "chunk_size must be %ld", CHUNK_SIZE),
            return ge::GRAPH_FAILED);
    }

    ChunkGatedDeltaRuleFwdHTilingData* tiling =
        context->GetTilingData<ChunkGatedDeltaRuleFwdHTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    tiling->batch = kShape->GetDim(0);
    tiling->totalTokens = kShape->GetDim(1);
    tiling->keyHeads = kShape->GetDim(2);
    tiling->keyDim = kShape->GetDim(3);
    tiling->valueHeads = uShape->GetDim(1);
    tiling->valueDim = uShape->GetDim(3);
    tiling->chunkSize = CHUNK_SIZE;
    tiling->hasInitialState = initialStateShape != nullptr ? 1 : 0;
    tiling->isVarLen = cuSeqlensShape != nullptr ? 1 : 0;
    tiling->storeFinalState = HasOutput(context, FINAL_STATE_OUTPUT) ? 1 : 0;

    if (tiling->isVarLen != 0) {
        OP_CHECK_IF(
            cuSeqlensShape->GetDimNum() != 1 || chunkIndicesShape->GetDimNum() != 2 ||
                chunkIndicesShape->GetDim(1) != 2,
            OP_LOGE(context, "invalid variable-length metadata rank"),
            return ge::GRAPH_FAILED);
        tiling->sequenceCount = cuSeqlensShape->GetDim(0) - 1;
        tiling->chunkCount = chunkIndicesShape->GetDim(0);
    } else {
        tiling->sequenceCount = tiling->batch;
        tiling->chunkCount = CeilDiv(tiling->totalTokens, CHUNK_SIZE);
    }

    OP_CHECK_IF(
        tiling->batch != 1 || tiling->totalTokens <= 0 || tiling->sequenceCount <= 0 ||
            tiling->chunkCount <= 0 || tiling->keyHeads <= 0 || tiling->valueHeads <= 0 ||
            tiling->keyDim <= 0 || tiling->valueDim <= 0 ||
            tiling->keyDim != tiling->valueDim || tiling->valueHeads % tiling->keyHeads != 0,
        OP_LOGE(context, "invalid dimensions for tiling"),
        return ge::GRAPH_FAILED);

    tiling->headRatio = tiling->valueHeads / tiling->keyHeads;
    tiling->vTileSize = std::min<int64_t>(DEFAULT_V_TILE, tiling->valueDim);
    tiling->vTileCount = CeilDiv(tiling->valueDim, tiling->vTileSize);
    tiling->taskCount = tiling->sequenceCount * tiling->valueHeads * tiling->vTileCount;
    tiling->usedCoreNum = std::min<int64_t>(tiling->taskCount, resources.aicCoreNum);
    tiling->taskCountPerCore = tiling->taskCount / tiling->usedCoreNum;
    tiling->taskTailCoreCount = tiling->taskCount % tiling->usedCoreNum;

    OP_CHECK_IF(
        !BuildMatmulTiling(
            resources.platform, static_cast<int32_t>(CHUNK_SIZE),
            static_cast<int32_t>(tiling->vTileSize), static_cast<int32_t>(tiling->keyDim),
            matmul_tiling::TPosition::VECCALC, false, tiling->mm1Tiling),
        OP_LOGE(context, "Failed to generate MM1 tiling"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !BuildMatmulTiling(
            resources.platform, static_cast<int32_t>(tiling->keyDim),
            static_cast<int32_t>(tiling->vTileSize), static_cast<int32_t>(CHUNK_SIZE),
            matmul_tiling::TPosition::VECCALC, true, tiling->mm2Tiling),
        OP_LOGE(context, "Failed to generate MM2 tiling"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        SetWorkspace(context, resources, *tiling) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "Failed to set workspace"),
        return ge::GRAPH_FAILED);

    const uint32_t usedAic = static_cast<uint32_t>(tiling->usedCoreNum);
    const uint32_t usedAiv = std::min<uint32_t>(resources.aivCoreNum, usedAic);
    context->SetBlockDim(resources.platform.CalcTschBlockDim(usedAic, usedAic, usedAiv));

    const uint64_t scheduleMode =
        static_cast<uint64_t>(tiling->hasInitialState) |
        (static_cast<uint64_t>(tiling->isVarLen) << 1U) |
        (static_cast<uint64_t>(tiling->storeFinalState) << 2U);
    context->SetTilingKey(GET_TPL_TILING_KEY(scheduleMode));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingParseForChunkGatedDeltaRuleFwdH([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

}  // namespace

struct ChunkGatedDeltaRuleFwdHCompileInfo {};

IMPL_OP_OPTILING(ChunkGatedDeltaRuleFwdH)
    .Tiling(ChunkGatedDeltaRuleFwdHTilingFunc)
    .TilingParse<ChunkGatedDeltaRuleFwdHCompileInfo>(TilingParseForChunkGatedDeltaRuleFwdH);

}  // namespace optiling
