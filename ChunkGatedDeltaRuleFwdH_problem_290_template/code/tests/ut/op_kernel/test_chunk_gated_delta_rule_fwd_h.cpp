/*!
 * \file test_chunk_gated_delta_rule_fwd_h.cpp
 * \brief End-to-end CPU-simulator test for the ChunkGatedDeltaRuleFwdH kernel.
 */

#include "chunk_gated_delta_rule_fwd_h_tiling.h"
#include "../../../op_kernel/chunk_gated_delta_rule_fwd_h.cpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "kernel_tiling_bridge.h"
#include "tikicpulib.h"

namespace {

constexpr int64_t BATCH = 1;
constexpr int64_t TOKENS = 5;
constexpr int64_t SEQUENCES = 1;
constexpr int64_t CHUNKS = 1;
constexpr int64_t VALUE_HEADS = 1;
constexpr int64_t KEY_HEADS = 1;
constexpr int64_t KEY_DIM = 32;
constexpr int64_t VALUE_DIM = 32;
constexpr int64_t CHUNK_SIZE = 64;

constexpr size_t BF16_BYTES = 2;
constexpr size_t K_BYTES = BATCH * TOKENS * KEY_HEADS * KEY_DIM * BF16_BYTES;
constexpr size_t W_BYTES = BATCH * VALUE_HEADS * TOKENS * KEY_DIM * BF16_BYTES;
constexpr size_t U_BYTES = BATCH * VALUE_HEADS * TOKENS * VALUE_DIM * BF16_BYTES;
constexpr size_t G_BYTES = BATCH * VALUE_HEADS * TOKENS * sizeof(float);
constexpr size_t INITIAL_STATE_BYTES =
    SEQUENCES * VALUE_HEADS * KEY_DIM * VALUE_DIM * BF16_BYTES;
constexpr size_t CU_SEQLENS_BYTES = (SEQUENCES + 1) * sizeof(int64_t);
constexpr size_t CHUNK_INDICES_BYTES = CHUNKS * 2 * sizeof(int64_t);
constexpr size_t H_BYTES =
    BATCH * VALUE_HEADS * CHUNKS * KEY_DIM * VALUE_DIM * BF16_BYTES;
constexpr size_t V_BYTES = BATCH * VALUE_HEADS * TOKENS * VALUE_DIM * BF16_BYTES;
constexpr size_t FINAL_STATE_BYTES = INITIAL_STATE_BYTES;

std::vector<uint8_t> ReadBinary(const std::string& path, size_t expectedBytes)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("Cannot open input file: " + path);
    }
    const auto fileSize = static_cast<size_t>(stream.tellg());
    if (fileSize != expectedBytes) {
        throw std::runtime_error(
            "Unexpected byte size for " + path + ": expected " +
            std::to_string(expectedBytes) + ", got " + std::to_string(fileSize));
    }
    std::vector<uint8_t> data(fileSize);
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(fileSize));
    if (!stream) {
        throw std::runtime_error("Cannot read input file: " + path);
    }
    return data;
}

void WriteBinary(const std::string& path, const uint8_t* data, size_t bytes)
{
    std::ofstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Cannot open output file: " + path);
    }
    stream.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!stream) {
        throw std::runtime_error("Cannot write output file: " + path);
    }
}

class GmBuffer {
public:
    explicit GmBuffer(size_t bytes) : bytes_(bytes)
    {
        data_ = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bytes_));
        if (data_ == nullptr) {
            throw std::bad_alloc();
        }
        std::memset(data_, 0, bytes_);
    }

    ~GmBuffer()
    {
        if (data_ != nullptr) {
            AscendC::GmFree(data_);
        }
    }

    GmBuffer(const GmBuffer&) = delete;
    GmBuffer& operator=(const GmBuffer&) = delete;

    uint8_t* Data() const
    {
        return data_;
    }

    void CopyFrom(const std::vector<uint8_t>& source)
    {
        if (source.size() != bytes_) {
            throw std::runtime_error("GM input byte size mismatch");
        }
        std::memcpy(data_, source.data(), bytes_);
    }

private:
    uint8_t* data_ = nullptr;
    size_t bytes_ = 0;
};

TEST(ChunkGatedDeltaRuleFwdHKernelTest, ExecutesValidVarLenChunk)
{
    ChunkGatedDeltaRuleFwdHTilingData hostTiling{};
    size_t workspaceBytes = 0;
    int64_t tilingKey = -1;
    ASSERT_TRUE(BuildKernelUtTiling(hostTiling, workspaceBytes, tilingKey));
    ASSERT_EQ(tilingKey, 7);
    ASSERT_EQ(hostTiling.sequenceCount, SEQUENCES);
    ASSERT_EQ(hostTiling.chunkCount, CHUNKS);
    ASSERT_EQ(hostTiling.taskCount, 1);
    ASSERT_EQ(hostTiling.usedCoreNum, 1);
    ASSERT_EQ(hostTiling.taskCountPerCore, 1);
    ASSERT_EQ(hostTiling.vTileSize, VALUE_DIM);
    ASSERT_EQ(hostTiling.mm1Tiling.M, CHUNK_SIZE);
    ASSERT_EQ(hostTiling.mm1Tiling.N, VALUE_DIM);
    ASSERT_EQ(hostTiling.mm1Tiling.Ka, KEY_DIM);
    ASSERT_EQ(hostTiling.mm2Tiling.M, KEY_DIM);
    ASSERT_EQ(hostTiling.mm2Tiling.N, VALUE_DIM);
    ASSERT_EQ(hostTiling.mm2Tiling.Ka, CHUNK_SIZE);
    ASSERT_GE(workspaceBytes, static_cast<size_t>(hostTiling.perCoreWorkspaceBytes));

    GmBuffer k(K_BYTES);
    GmBuffer w(W_BYTES);
    GmBuffer u(U_BYTES);
    GmBuffer g(G_BYTES);
    GmBuffer initialState(INITIAL_STATE_BYTES);
    GmBuffer cuSeqlens(CU_SEQLENS_BYTES);
    GmBuffer chunkIndices(CHUNK_INDICES_BYTES);
    GmBuffer h(H_BYTES);
    GmBuffer v(V_BYTES);
    GmBuffer finalState(FINAL_STATE_BYTES);
    GmBuffer workspace(workspaceBytes);
    GmBuffer tiling(sizeof(hostTiling));

    k.CopyFrom(ReadBinary("bfloat16_input_chunk_gated_delta_rule_fwd_h_k.bin", K_BYTES));
    w.CopyFrom(ReadBinary("bfloat16_input_chunk_gated_delta_rule_fwd_h_w.bin", W_BYTES));
    u.CopyFrom(ReadBinary("bfloat16_input_chunk_gated_delta_rule_fwd_h_u.bin", U_BYTES));
    g.CopyFrom(ReadBinary("bfloat16_input_chunk_gated_delta_rule_fwd_h_g.bin", G_BYTES));
    initialState.CopyFrom(ReadBinary(
        "bfloat16_input_chunk_gated_delta_rule_fwd_h_initial_state.bin",
        INITIAL_STATE_BYTES));
    cuSeqlens.CopyFrom(ReadBinary(
        "bfloat16_input_chunk_gated_delta_rule_fwd_h_cu_seqlens.bin",
        CU_SEQLENS_BYTES));
    chunkIndices.CopyFrom(ReadBinary(
        "bfloat16_input_chunk_gated_delta_rule_fwd_h_chunk_indices.bin",
        CHUNK_INDICES_BYTES));
    std::memcpy(tiling.Data(), &hostTiling, sizeof(hostTiling));

    ICPU_SET_TILING_KEY(7);
    AscendC::SetKernelMode(KernelMode::MIX_AIC_1_1);
    ICPU_RUN_KF(
        (chunk_gated_delta_rule_fwd_h<7>), 1,
        k.Data(), w.Data(), u.Data(), g.Data(), initialState.Data(),
        cuSeqlens.Data(), chunkIndices.Data(), h.Data(), v.Data(),
        finalState.Data(), workspace.Data(), tiling.Data());

    WriteBinary("bfloat16_output_chunk_gated_delta_rule_fwd_h_0.bin", h.Data(), H_BYTES);
    WriteBinary("bfloat16_output_chunk_gated_delta_rule_fwd_h_1.bin", v.Data(), V_BYTES);
    WriteBinary(
        "bfloat16_output_chunk_gated_delta_rule_fwd_h_2.bin",
        finalState.Data(), FINAL_STATE_BYTES);
}

}  // namespace
