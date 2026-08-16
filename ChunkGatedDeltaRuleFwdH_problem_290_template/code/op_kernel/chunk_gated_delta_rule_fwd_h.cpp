/*!
 * \file chunk_gated_delta_rule_fwd_h.cpp
 * \brief ChunkGatedDeltaRuleFwdH 算子 kernel 入口
 */

#include "chunk_gated_delta_rule_fwd_h.h"

enum class ChunkGatedDeltaRuleFwdHTilingKey : uint32_t
{
    TILING_KEY_CHUNKGATEDDELTARULEFWDH_MODE_0 = 0,
    TILING_KEY_CHUNKGATEDDELTARULEFWDH_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void chunk_gated_delta_rule_fwd_h(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR initial_state, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR h, GM_ADDR v, GM_ADDR final_state, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ChunkGatedDeltaRuleFwdHTilingData);
    GET_TILING_DATA_WITH_STRUCT(ChunkGatedDeltaRuleFwdHTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(ChunkGatedDeltaRuleFwdHTilingKey::TILING_KEY_CHUNKGATEDDELTARULEFWDH_MODE_0)) {
        NsChunkGatedDeltaRuleFwdH::ChunkGatedDeltaRuleFwdH<bfloat16_t> op;
        op.Init(k, w, u, g, initial_state, cu_seqlens, chunk_indices, h, v, final_state, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(ChunkGatedDeltaRuleFwdHTilingKey::TILING_KEY_CHUNKGATEDDELTARULEFWDH_MODE_1)) {
        NsChunkGatedDeltaRuleFwdH::ChunkGatedDeltaRuleFwdH<float> op;
        op.Init(k, w, u, g, initial_state, cu_seqlens, chunk_indices, h, v, final_state, &tilingData);
        op.Process();
    }
}
