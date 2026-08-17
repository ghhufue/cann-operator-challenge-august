/*!
 * \file chunk_gated_delta_rule_fwd_h.cpp
 * \brief ChunkGatedDeltaRuleFwdH 算子 kernel 入口
 */

#include "chunk_gated_delta_rule_fwd_h.h"

template <uint32_t schMode>
__global__ __aicore__ void chunk_gated_delta_rule_fwd_h(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR initial_state, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR h, GM_ADDR v, GM_ADDR final_state, GM_ADDR workspace, GM_ADDR tiling)
{
    static_assert(schMode <= CHUNKGATEDDELTARULEFWDH_TPL_SCH_MODE_7, "invalid tiling mode");
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    REGISTER_TILING_DEFAULT(ChunkGatedDeltaRuleFwdHTilingData);
    GET_TILING_DATA_WITH_STRUCT(ChunkGatedDeltaRuleFwdHTilingData, tilingData, tiling);

    AscendC::TPipe pipe;
    NsChunkGatedDeltaRuleFwdH::ChunkGatedDeltaRuleFwdH<bfloat16_t> op;
    op.Init(k, w, u, g, initial_state, cu_seqlens, chunk_indices,
            h, v, final_state, workspace, &tilingData, &pipe);
    op.Process();
}
