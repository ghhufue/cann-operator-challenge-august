/*!
 * \file test_chunk_gated_delta_rule_fwd_h.cpp
 * \brief ChunkGatedDeltaRuleFwdH 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "chunk_gated_delta_rule_fwd_h_tiling.h"
#include "../../../op_kernel/chunk_gated_delta_rule_fwd_h.cpp"

#include <array>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <new>
#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace std;

static uint16_t FloatToHalf(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ff;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7c00;
    return sign | (exp << 10) | mant;
}

static uint16_t FloatToBFloat16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    return (uint16_t)(bits >> 16);
}

class ChunkGatedDeltaRuleFwdHKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "ChunkGatedDeltaRuleFwdHKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "ChunkGatedDeltaRuleFwdHKernelTest TearDown" << endl;
    }
};

TEST_F(ChunkGatedDeltaRuleFwdHKernelTest, test_kernel_run)
{
    constexpr size_t size = 2564096;
    constexpr size_t tilingDataSize = sizeof(ChunkGatedDeltaRuleFwdHTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t kByteSize = 2564096 * 2;
    constexpr size_t wByteSize = 10256384 * 2;
    constexpr size_t uByteSize = 10256384 * 2;
    constexpr size_t gByteSize = 80128 * 4;
    constexpr size_t initial_stateByteSize = 262144 * 2;
    constexpr size_t cu_seqlensByteSize = 3 * 8;
    constexpr size_t chunk_indicesByteSize = 316 * 8;
    constexpr size_t hByteSize = 20709376 * 2;
    constexpr size_t vByteSize = 10256384 * 2;
    constexpr size_t final_stateByteSize = 262144 * 2;
    std::vector<float> kHost(2564096, 1);
    std::vector<float> wHost(10256384, 1);
    std::vector<float> uHost(10256384, 1);
    std::vector<float> gHost(80128, 1);
    std::vector<float> initial_stateHost(262144, 1);
    std::vector<int64_t> cu_seqlensHost(3, 1);
    std::vector<int64_t> chunk_indicesHost(316, 1);
    std::vector<float> hHost(20709376, 0);
    std::vector<float> vHost(10256384, 0);
    std::vector<float> final_stateHost(262144, 0);
    
    
    uint8_t* k = (uint8_t*)AscendC::GmAlloc(kByteSize);
    uint8_t* w = (uint8_t*)AscendC::GmAlloc(wByteSize);
    uint8_t* u = (uint8_t*)AscendC::GmAlloc(uByteSize);
    uint8_t* g = (uint8_t*)AscendC::GmAlloc(gByteSize);
    uint8_t* initial_state = (uint8_t*)AscendC::GmAlloc(initial_stateByteSize);
    uint8_t* cu_seqlens = (uint8_t*)AscendC::GmAlloc(cu_seqlensByteSize);
    uint8_t* chunk_indices = (uint8_t*)AscendC::GmAlloc(chunk_indicesByteSize);
    uint8_t* h = (uint8_t*)AscendC::GmAlloc(hByteSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(vByteSize);
    uint8_t* final_state = (uint8_t*)AscendC::GmAlloc(final_stateByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    
    for (size_t _i = 0; _i < 2564096; _i++) { uint16_t _b = FloatToBFloat16(kHost[_i]); memcpy(k + _i * 2, &_b, 2); }
    for (size_t _i = 0; _i < 10256384; _i++) { uint16_t _b = FloatToBFloat16(wHost[_i]); memcpy(w + _i * 2, &_b, 2); }
    for (size_t _i = 0; _i < 10256384; _i++) { uint16_t _b = FloatToBFloat16(uHost[_i]); memcpy(u + _i * 2, &_b, 2); }
    memcpy(g, gHost.data(), gByteSize);
    for (size_t _i = 0; _i < 262144; _i++) { uint16_t _b = FloatToBFloat16(initial_stateHost[_i]); memcpy(initial_state + _i * 2, &_b, 2); }
    memcpy(cu_seqlens, cu_seqlensHost.data(), cu_seqlensByteSize);
    memcpy(chunk_indices, chunk_indicesHost.data(), chunk_indicesByteSize);
    
    // Kernel implementation is still a stub, but keep this CPU-debug launch
    // structurally consistent with the current Host tiling contract.
    ChunkGatedDeltaRuleFwdHTilingData* tilingData =
        new (tiling) ChunkGatedDeltaRuleFwdHTilingData{};
    tilingData->batch = 1;
    tilingData->totalTokens = 10016;
    tilingData->sequenceCount = 2;
    tilingData->chunkCount = 158;
    tilingData->valueHeads = 8;
    tilingData->keyHeads = 2;
    tilingData->keyDim = 128;
    tilingData->valueDim = 128;
    tilingData->chunkSize = 64;
    tilingData->headRatio = 4;
    tilingData->vTileSize = 64;
    tilingData->vTileCount = 2;
    tilingData->taskCount = 32;
    tilingData->usedCoreNum = 1;
    tilingData->taskCountPerCore = 32;
    tilingData->hasInitialState = 1;
    tilingData->isVarLen = 1;
    tilingData->storeFinalState = 1;
    
    ICPU_SET_TILING_KEY(7);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((chunk_gated_delta_rule_fwd_h<7>), numBlocks, k, w, u, g, initial_state, cu_seqlens, chunk_indices, h, v, final_state, workspace, tiling);
    
    // 将动态输出的 packed buffer 拆回 individual buffers
    
    
    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(hHost.data(), h, hByteSize);
    { std::ofstream _ofs("bfloat16_output_chunk_gated_delta_rule_fwd_h_0.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(hHost.data()), hByteSize); }
    memcpy(vHost.data(), v, vByteSize);
    { std::ofstream _ofs("bfloat16_output_chunk_gated_delta_rule_fwd_h_1.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(vHost.data()), vByteSize); }
    memcpy(final_stateHost.data(), final_state, final_stateByteSize);
    { std::ofstream _ofs("bfloat16_output_chunk_gated_delta_rule_fwd_h_2.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(final_stateHost.data()), final_stateByteSize); }
    
    AscendC::GmFree(k);
    AscendC::GmFree(w);
    AscendC::GmFree(u);
    AscendC::GmFree(g);
    AscendC::GmFree(initial_state);
    AscendC::GmFree(cu_seqlens);
    AscendC::GmFree(chunk_indices);
    AscendC::GmFree(h);
    AscendC::GmFree(v);
    AscendC::GmFree(final_state);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
