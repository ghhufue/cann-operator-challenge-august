#include "kernel_tiling_bridge.h"

#include <cstring>
#include <memory>
#include <vector>

#include "base/registry/op_impl_space_registry_v2.h"
#include "tiling_case_executor.h"

namespace {

using TensorDescription = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;

constexpr char OP_NAME[] = "ChunkGatedDeltaRuleFwdH";
constexpr uint64_t UB_SIZE = 262144;
constexpr uint64_t TILING_DATA_CAPACITY = 4096;
constexpr int64_t BATCH = 1;
constexpr int64_t TOKENS = 5;
constexpr int64_t SEQUENCES = 1;
constexpr int64_t CHUNKS = 1;
constexpr int64_t VALUE_HEADS = 1;
constexpr int64_t KEY_HEADS = 1;
constexpr int64_t KEY_DIM = 32;
constexpr int64_t VALUE_DIM = 32;
constexpr int64_t CHUNK_SIZE = 64;

struct ChunkGatedDeltaRuleFwdHCompileInfo {} compileInfo;

TensorDescription Tensor(std::initializer_list<int64_t> dims, ge::DataType dtype)
{
    gert::StorageShape shape = {dims, dims};
    return TensorDescription(shape, dtype, ge::FORMAT_ND);
}

gert::TilingContextPara MakeTilingContext()
{
    std::vector<TensorDescription> inputs = {
        Tensor({BATCH, TOKENS, KEY_HEADS, KEY_DIM}, ge::DT_BF16),
        Tensor({BATCH, VALUE_HEADS, TOKENS, KEY_DIM}, ge::DT_BF16),
        Tensor({BATCH, VALUE_HEADS, TOKENS, VALUE_DIM}, ge::DT_BF16),
        Tensor({BATCH, VALUE_HEADS, TOKENS}, ge::DT_FLOAT),
        Tensor({SEQUENCES, VALUE_HEADS, KEY_DIM, VALUE_DIM}, ge::DT_BF16),
        Tensor({SEQUENCES + 1}, ge::DT_INT64),
        Tensor({CHUNKS, 2}, ge::DT_INT64),
    };
    std::vector<TensorDescription> outputs = {
        Tensor({BATCH, VALUE_HEADS, CHUNKS, KEY_DIM, VALUE_DIM}, ge::DT_BF16),
        Tensor({BATCH, VALUE_HEADS, TOKENS, VALUE_DIM}, ge::DT_BF16),
        Tensor({SEQUENCES, VALUE_HEADS, KEY_DIM, VALUE_DIM}, ge::DT_BF16),
    };
    std::vector<OpAttr> attrs = {
        OpAttr("chunk_size", Ops::Math::AnyValue::CreateFrom<int64_t>(CHUNK_SIZE)),
    };
    return gert::TilingContextPara(
        OP_NAME, inputs, outputs, attrs, &compileInfo, 16, UB_SIZE, TILING_DATA_CAPACITY);
}

bool InitializeRegistry()
{
    gert::OppSoDesc opHostDesc(
        {ge::AscendString(OP_HOST_UT_SO_PATH)}, "chunk_gated_delta_rule_fwd_h_op_host_ut");
    auto registry = std::make_shared<gert::OpImplSpaceRegistryV2>();
    if (registry->AddSoToRegistry(opHostDesc) != ge::GRAPH_SUCCESS) {
        return false;
    }
    gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(registry);
    return true;
}

}  // namespace

bool BuildKernelUtTiling(
    ChunkGatedDeltaRuleFwdHTilingData& tiling,
    size_t& workspaceBytes,
    int64_t& tilingKey)
{
    if (!InitializeRegistry()) {
        return false;
    }

    TilingInfo info;
    if (!ExecuteTiling(MakeTilingContext(), info) ||
        info.tilingDataSize != sizeof(ChunkGatedDeltaRuleFwdHTilingData) ||
        info.workspaceSizes.size() != 1U) {
        return false;
    }

    std::memcpy(&tiling, info.tilingData.get(), sizeof(tiling));
    workspaceBytes = info.workspaceSizes[0];
    tilingKey = info.tilingKey;
    return true;
}
