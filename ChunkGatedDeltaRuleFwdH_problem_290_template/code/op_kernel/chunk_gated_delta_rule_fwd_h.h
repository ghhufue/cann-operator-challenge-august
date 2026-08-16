/*!
 * \file chunk_gated_delta_rule_fwd_h.h
 * \brief ChunkGatedDeltaRuleFwdH 算子 kernel 类定义
 */

#ifndef CHUNKGATEDDELTARULEFWDH_H
#define CHUNKGATEDDELTARULEFWDH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "chunk_gated_delta_rule_fwd_h_tiling_data.h"
#include "chunk_gated_delta_rule_fwd_h_tiling_key.h"

namespace NsChunkGatedDeltaRuleFwdH {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class ChunkGatedDeltaRuleFwdH {
public:
    __aicore__ inline ChunkGatedDeltaRuleFwdH(){};

    __aicore__ inline void Init(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR initial_state, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR h, GM_ADDR v, GM_ADDR final_state, const ChunkGatedDeltaRuleFwdHTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void ChunkGatedDeltaRuleFwdH<T>::Init(GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR initial_state, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR h, GM_ADDR v, GM_ADDR final_state, const ChunkGatedDeltaRuleFwdHTilingData* tilingData)
{
    // TODO: 实现 Init 逻辑
}

template <typename T>
__aicore__ inline void ChunkGatedDeltaRuleFwdH<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyIn 逻辑
}

template <typename T>
__aicore__ inline void ChunkGatedDeltaRuleFwdH<T>::Compute(int64_t currentNum)
{
    // TODO: 实现 Compute 逻辑
}

template <typename T>
__aicore__ inline void ChunkGatedDeltaRuleFwdH<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    // TODO: 实现 CopyOut 逻辑
}

template <typename T>
__aicore__ inline void ChunkGatedDeltaRuleFwdH<T>::Process()
{
    // TODO: 实现 Process 逻辑
}

} // namespace NsChunkGatedDeltaRuleFwdH
#endif // CHUNKGATEDDELTARULEFWDH_H
