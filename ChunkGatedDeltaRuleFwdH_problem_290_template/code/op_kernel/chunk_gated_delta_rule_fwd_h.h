/*!
 * \file chunk_gated_delta_rule_fwd_h.h
 * \brief ChunkGatedDeltaRuleFwdH MIX kernel implementation.
 */

#ifndef CHUNKGATEDDELTARULEFWDH_H
#define CHUNKGATEDDELTARULEFWDH_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "chunk_gated_delta_rule_fwd_h_tiling_data.h"
#include "chunk_gated_delta_rule_fwd_h_tiling_key.h"

namespace NsChunkGatedDeltaRuleFwdH {

using namespace AscendC;

template <typename T>
class ChunkGatedDeltaRuleFwdH {
public:
    __aicore__ inline ChunkGatedDeltaRuleFwdH() = default;
    __aicore__ inline void Init(
        GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR initialState,
        GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR h, GM_ADDR v,
        GM_ADDR finalState, GM_ADDR workspace,
        const ChunkGatedDeltaRuleFwdHTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    struct TaskInfo {
        int64_t sequence;
        int64_t valueHead;
        int64_t keyHead;
        int64_t vStart;
        int64_t validV;
    };

    __aicore__ inline TaskInfo DecodeTask(int64_t taskId) const;
    __aicore__ inline void GetSequenceRange(
        int64_t sequence, int64_t& bos, int64_t& eos,
        int64_t& chunkOffset, int64_t& chunkNum) const;
    __aicore__ inline void InitState(const TaskInfo& task, const LocalTensor<T>& state);
    __aicore__ inline void StoreState(
        const GlobalTensor<T>& dst, int64_t baseOffset,
        const TaskInfo& task, const LocalTensor<T>& state);
    __aicore__ inline void PackW(
        const TaskInfo& task, int64_t tokenStart, int64_t actualLen,
        const LocalTensor<T>& chunkLocal);
    __aicore__ inline void PackK(
        const TaskInfo& task, int64_t tokenStart, int64_t actualLen,
        const LocalTensor<T>& chunkLocal);
    __aicore__ inline void ComputeValueAndDecay(
        const TaskInfo& task, int64_t tokenStart, int64_t actualLen,
        const LocalTensor<T>& stage, const LocalTensor<half>& halfWork,
        const LocalTensor<float>& calcWork, const LocalTensor<T>& chunkStorage);
    __aicore__ inline void MergeDelta(
        const LocalTensor<T>& state, const LocalTensor<T>& delta,
        const LocalTensor<half>& halfWork, const LocalTensor<float>& calcWork);
    __aicore__ inline void MatmulWStateP4(
        LocalTensor<float>& wFloat, LocalTensor<T>& state,
        LocalTensor<T>& out, LocalTensor<float>& scratch,
        int64_t actualLen);
    __aicore__ inline void MatmulKtVDecaySeq(
        LocalTensor<float>& kFloat, LocalTensor<T>& vDecay,
        LocalTensor<T>& out, LocalTensor<float>& scratch,
        int64_t actualLen);
    __aicore__ inline void ProcessTask(int64_t taskId);

private:
    GlobalTensor<T> kGm_;
    GlobalTensor<T> wGm_;
    GlobalTensor<T> uGm_;
    GlobalTensor<float> gGm_;
    GlobalTensor<T> initialStateGm_;
    GlobalTensor<int64_t> cuSeqlensGm_;
    GlobalTensor<int64_t> chunkIndicesGm_;
    GlobalTensor<T> hGm_;
    GlobalTensor<T> vGm_;
    GlobalTensor<T> finalStateGm_;

    TBuf<TPosition::VECCALC> stateBuf_;
    TBuf<TPosition::VECCALC> chunkBuf_;
    TBuf<TPosition::VECCALC> stageBuf_;
    TBuf<TPosition::VECCALC> halfWorkBuf_;
    TBuf<TPosition::VECCALC> calcWorkBuf_;
    TBuf<TPosition::VECCALC> matFloatBuf_;

    const ChunkGatedDeltaRuleFwdHTilingData* tiling_ = nullptr;
    int64_t blockIdx_ = 0;
    int64_t stateElements_ = 0;
    int64_t chunkElements_ = 0;
    int64_t stageElements_ = 0;
    int64_t halfWorkElements_ = 0;
};

template <typename T>
__aicore__ inline void ChunkGatedDeltaRuleFwdH<T>::Init(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR h, GM_ADDR v,
    GM_ADDR finalState, GM_ADDR workspace,
    const ChunkGatedDeltaRuleFwdHTilingData* tilingData, TPipe* pipe)
{
    tiling_ = tilingData;
    blockIdx_ = GetBlockIdx();
    stateElements_ = tiling_->keyDim * tiling_->vTileSize;
    chunkElements_ = tiling_->chunkSize * tiling_->keyDim;
    stageElements_ = tiling_->chunkSize * tiling_->vTileSize;
    const int64_t maxMatrixElements =
        stateElements_ > stageElements_ ? stateElements_ : stageElements_;
    halfWorkElements_ = maxMatrixElements + tiling_->vTileSize;

    kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(k));
    wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(w));
    uGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(u));
    gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(g));
    if (tiling_->hasInitialState != 0) {
        initialStateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(initialState));
    }
    if (tiling_->isVarLen != 0) {
        cuSeqlensGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(cuSeqlens));
        chunkIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(chunkIndices));
    }
    hGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(h));
    vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(v));
    if (tiling_->storeFinalState != 0) {
        finalStateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(finalState));
    }

    if ASCEND_IS_AIV {
        pipe->InitBuffer(stateBuf_, stateElements_ * sizeof(T));
        pipe->InitBuffer(chunkBuf_, chunkElements_ * sizeof(T));
        pipe->InitBuffer(stageBuf_, stageElements_ * sizeof(T));
        pipe->InitBuffer(halfWorkBuf_, halfWorkElements_ * sizeof(half));
        pipe->InitBuffer(calcWorkBuf_, 8 * tiling_->vTileSize * sizeof(float));
        pipe->InitBuffer(matFloatBuf_, chunkElements_ * sizeof(float));
    }
}

template <typename T>
__aicore__ inline void ChunkGatedDeltaRuleFwdH<T>::ComputeValueAndDecay(
    const TaskInfo& task, int64_t tokenStart, int64_t actualLen,
    const LocalTensor<T>& stage, const LocalTensor<half>& halfWork,
    const LocalTensor<float>& calcWork, const LocalTensor<T>& chunkStorage)
{
    if ASCEND_IS_AIV {
        const int64_t tile = tiling_->vTileSize;
        LocalTensor<float> whFloat = calcWork;
        LocalTensor<float> valueFloat = calcWork[tile];
        LocalTensor<half> scalarHalf = halfWork[halfWorkElements_ - tile];

        // Preserve round_to_fp16(WH) before stage is reused to load U.
        for (int64_t row = 0; row < actualLen; ++row) {
            Cast(whFloat, stage[row * tile], RoundMode::CAST_NONE, tile);
            Cast(halfWork[row * tile], whFloat, RoundMode::CAST_NONE, tile);
        }
        event_t vectorToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(vectorToMte2);
        WaitFlag<HardEvent::V_MTE2>(vectorToMte2);

        const int64_t uBase =
            (task.valueHead * tiling_->totalTokens + tokenStart) * tiling_->valueDim;
        DataCopyExtParams uCopy{1, static_cast<uint32_t>(task.validV * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> noPad{false, 0, 0, static_cast<T>(0)};
        for (int64_t row = 0; row < actualLen; ++row) {
            DataCopyPad(stage[row * tile],
                        uGm_[uBase + row * tiling_->valueDim + task.vStart],
                        uCopy, noPad);
        }
        event_t uReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(uReady);
        WaitFlag<HardEvent::MTE2_V>(uReady);

        // chunkStorage is no longer needed as W after MM1. Reinterpret its
        // first bytes as the gate vectors; PackK overwrites it afterwards.
        LocalTensor<float> gateFloat = chunkStorage.template ReinterpretCast<float>();
        LocalTensor<half> gateHalfStorage = chunkStorage.template ReinterpretCast<half>();
        LocalTensor<half> gateHalf = gateHalfStorage[128];
        LocalTensor<half> alphaHalf = gateHalfStorage[192];
        const int64_t gBase = task.valueHead * tiling_->totalTokens + tokenStart;
        const float gLast = gGm_.GetValue(gBase + actualLen - 1);
        DataCopyExtParams gCopy{1, static_cast<uint32_t>(actualLen * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> noFloatPad{false, 0, 0, 0.0f};
        DataCopyPad(gateFloat, gGm_[gBase], gCopy, noFloatPad);
        event_t gateReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(gateReady);
        WaitFlag<HardEvent::MTE2_V>(gateReady);
        Muls(gateFloat, gateFloat, -1.0f, actualLen);
        Adds(gateFloat, gateFloat, gLast, actualLen);
        Exp(gateFloat, gateFloat, actualLen);
        Cast(gateHalf, gateFloat, RoundMode::CAST_NONE, actualLen);

        // alpha = exp(round_to_fp16(g_last)), matching the reference's
        // explicit FP16 conversion before the state decay exponent.
        Duplicate(alphaHalf, static_cast<half>(gLast), tile);
        Exp(alphaHalf, alphaHalf, tile);
        event_t scalarReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(scalarReady);
        WaitFlag<HardEvent::V_S>(scalarReady);
        const float alpha = static_cast<float>(alphaHalf.GetValue(0));

        for (int64_t row = 0; row < actualLen; ++row) {
            // v_new_fp16 = fp16(fp16(U).float - fp16(WH).float)
            Cast(valueFloat, stage[row * tile], RoundMode::CAST_NONE, tile);
            Cast(scalarHalf, valueFloat, RoundMode::CAST_NONE, tile);
            Cast(valueFloat, scalarHalf, RoundMode::CAST_NONE, tile);
            Cast(whFloat, halfWork[row * tile], RoundMode::CAST_NONE, tile);
            Sub(valueFloat, valueFloat, whFloat, tile);
            Cast(scalarHalf, valueFloat, RoundMode::CAST_NONE, tile);

            // Save the un-decayed v_new output as BF16.
            Cast(valueFloat, scalarHalf, RoundMode::CAST_NONE, tile);
            Cast(stage[row * tile], valueFloat, RoundMode::CAST_RINT, tile);
            event_t valueToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(valueToMte3);
            WaitFlag<HardEvent::V_MTE3>(valueToMte3);
            const int64_t vDst =
                (task.valueHead * tiling_->totalTokens + tokenStart + row) *
                tiling_->valueDim + task.vStart;
            DataCopyPad(vGm_[vDst], stage[row * tile], uCopy);
            event_t valueStored = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(valueStored);
            WaitFlag<HardEvent::MTE3_V>(valueStored);

            // v_decay = fp16(v_new_fp16.float * fp16(exp(g_last-g))).
            const float gate = static_cast<float>(gateHalf.GetValue(row));
            Cast(valueFloat, scalarHalf, RoundMode::CAST_NONE, tile);
            Muls(valueFloat, valueFloat, gate, tile);
            Cast(scalarHalf, valueFloat, RoundMode::CAST_NONE, tile);
            Cast(valueFloat, scalarHalf, RoundMode::CAST_NONE, tile);
            Cast(stage[row * tile], valueFloat, RoundMode::CAST_RINT, tile);
        }
        if (actualLen < tiling_->chunkSize) {
            Duplicate(stage[actualLen * tile], static_cast<T>(0),
                      (tiling_->chunkSize - actualLen) * tile);
        }

        // State decay is deliberately done here while alpha is available.
        for (int64_t row = 0; row < tiling_->keyDim; ++row) {
            Cast(whFloat, stateBuf_.Get<T>()[row * tile], RoundMode::CAST_NONE, tile);
            Cast(halfWork[row * tile], whFloat, RoundMode::CAST_NONE, tile);
            Cast(whFloat, halfWork[row * tile], RoundMode::CAST_NONE, tile);
            Muls(whFloat, whFloat, alpha, tile);
            Cast(halfWork[row * tile], whFloat, RoundMode::CAST_NONE, tile);
            Cast(whFloat, halfWork[row * tile], RoundMode::CAST_NONE, tile);
            Cast(stateBuf_.Get<T>()[row * tile], whFloat, RoundMode::CAST_RINT, tile);
        }
    }
}

template <typename T>
__aicore__ inline void ChunkGatedDeltaRuleFwdH<T>::MergeDelta(
    const LocalTensor<T>& state, const LocalTensor<T>& delta,
    const LocalTensor<half>& halfWork, const LocalTensor<float>& calcWork)
{
    if ASCEND_IS_AIV {
        const int64_t tile = tiling_->vTileSize;
        LocalTensor<float> stateFloat = calcWork;
        LocalTensor<float> deltaFloat = calcWork[tile];
        for (int64_t row = 0; row < tiling_->keyDim; ++row) {
            Cast(deltaFloat, delta[row * tile], RoundMode::CAST_NONE, tile);
            Cast(halfWork[row * tile], deltaFloat, RoundMode::CAST_NONE, tile);
            Cast(deltaFloat, halfWork[row * tile], RoundMode::CAST_NONE, tile);
            Cast(delta[row * tile], deltaFloat, RoundMode::CAST_RINT, tile);
            Cast(stateFloat, state[row * tile], RoundMode::CAST_NONE, tile);
            Cast(deltaFloat, delta[row * tile], RoundMode::CAST_NONE, tile);
            Add(stateFloat, stateFloat, deltaFloat, tile);
            Cast(state[row * tile], stateFloat, RoundMode::CAST_RINT, tile);
        }
    }
}

template <typename T>
__aicore__ inline void ChunkGatedDeltaRuleFwdH<T>::MatmulWStateP4(
    LocalTensor<float>& wFloat, LocalTensor<T>& state,
    LocalTensor<T>& out, LocalTensor<float>& scratch,
    int64_t actualLen)
{
    if ASCEND_IS_AIV {
        const int64_t tile = tiling_->vTileSize;
        const int64_t kDim = tiling_->keyDim;
        LocalTensor<float> acc0 = scratch;
        LocalTensor<float> acc1 = scratch[tile];
        LocalTensor<float> acc2 = scratch[2 * tile];
        LocalTensor<float> acc3 = scratch[3 * tile];
        LocalTensor<float> prod = scratch[4 * tile];
        LocalTensor<float> tmp1 = scratch[5 * tile];
        LocalTensor<float> tmp2 = scratch[6 * tile];
        LocalTensor<float> rowF = scratch[7 * tile];

        for (int64_t m = 0; m < actualLen; ++m) {
            Duplicate(acc0, 0.0f, tile);
            Duplicate(acc1, 0.0f, tile);
            Duplicate(acc2, 0.0f, tile);
            Duplicate(acc3, 0.0f, tile);

            int64_t k = 0;
            for (; k + 4 <= kDim; k += 4) {
                Cast(rowF, state[k * tile], RoundMode::CAST_NONE, tile);
                Muls(prod, rowF, wFloat.GetValue(m * kDim + k), tile);
                Add(acc0, acc0, prod, tile);

                Cast(rowF, state[(k + 1) * tile], RoundMode::CAST_NONE, tile);
                Muls(prod, rowF, wFloat.GetValue(m * kDim + k + 1), tile);
                Add(acc1, acc1, prod, tile);

                Cast(rowF, state[(k + 2) * tile], RoundMode::CAST_NONE, tile);
                Muls(prod, rowF, wFloat.GetValue(m * kDim + k + 2), tile);
                Add(acc2, acc2, prod, tile);

                Cast(rowF, state[(k + 3) * tile], RoundMode::CAST_NONE, tile);
                Muls(prod, rowF, wFloat.GetValue(m * kDim + k + 3), tile);
                Add(acc3, acc3, prod, tile);
            }
            for (; k < kDim; ++k) {
                Cast(rowF, state[k * tile], RoundMode::CAST_NONE, tile);
                Muls(prod, rowF, wFloat.GetValue(m * kDim + k), tile);
                const int64_t lane = k & 3;
                if (lane == 0) {
                    Add(acc0, acc0, prod, tile);
                } else if (lane == 1) {
                    Add(acc1, acc1, prod, tile);
                } else if (lane == 2) {
                    Add(acc2, acc2, prod, tile);
                } else {
                    Add(acc3, acc3, prod, tile);
                }
            }

            Add(tmp1, acc0, acc1, tile);
            Add(tmp2, acc2, acc3, tile);
            Add(tmp1, tmp1, tmp2, tile);
            Cast(out[m * tile], tmp1, RoundMode::CAST_RINT, tile);
        }
    }
}

template <typename T>
__aicore__ inline void ChunkGatedDeltaRuleFwdH<T>::MatmulKtVDecaySeq(
    LocalTensor<float>& kFloat, LocalTensor<T>& vDecay,
    LocalTensor<T>& out, LocalTensor<float>& scratch,
    int64_t actualLen)
{
    if ASCEND_IS_AIV {
        const int64_t tile = tiling_->vTileSize;
        const int64_t kDim = tiling_->keyDim;
        LocalTensor<float> accN = scratch;
        LocalTensor<float> prod = scratch[tile];
        LocalTensor<float> rowF = scratch[2 * tile];

        for (int64_t m = 0; m < kDim; ++m) {
            Duplicate(accN, 0.0f, tile);
            for (int64_t t = 0; t < actualLen; ++t) {
                Cast(rowF, vDecay[t * tile], RoundMode::CAST_NONE, tile);
                Muls(prod, rowF, kFloat.GetValue(t * kDim + m), tile);
                Add(accN, accN, prod, tile);
            }
            Cast(out[m * tile], accN, RoundMode::CAST_RINT, tile);
        }
    }
}

template <typename T>
__aicore__ inline typename ChunkGatedDeltaRuleFwdH<T>::TaskInfo
ChunkGatedDeltaRuleFwdH<T>::DecodeTask(int64_t taskId) const
{
    TaskInfo task;
    const int64_t vTileId = taskId % tiling_->vTileCount;
    const int64_t sequenceAndHead = taskId / tiling_->vTileCount;
    task.valueHead = sequenceAndHead % tiling_->valueHeads;
    task.sequence = sequenceAndHead / tiling_->valueHeads;
    task.keyHead = task.valueHead / tiling_->headRatio;
    task.vStart = vTileId * tiling_->vTileSize;
    const int64_t remaining = tiling_->valueDim - task.vStart;
    task.validV = remaining < tiling_->vTileSize ? remaining : tiling_->vTileSize;
    return task;
}

template <typename T>
__aicore__ inline void ChunkGatedDeltaRuleFwdH<T>::GetSequenceRange(
    int64_t sequence, int64_t& bos, int64_t& eos,
    int64_t& chunkOffset, int64_t& chunkNum) const
{
    if (tiling_->isVarLen == 0) {
        bos = 0;
        eos = tiling_->totalTokens;
        chunkOffset = 0;
    } else {
        bos = cuSeqlensGm_.GetValue(sequence);
        eos = cuSeqlensGm_.GetValue(sequence + 1);
        chunkOffset = 0;
        for (int64_t n = 0; n < sequence; ++n) {
            const int64_t begin = cuSeqlensGm_.GetValue(n);
            const int64_t end = cuSeqlensGm_.GetValue(n + 1);
            chunkOffset += (end - begin + tiling_->chunkSize - 1) / tiling_->chunkSize;
        }
    }
    chunkNum = (eos - bos + tiling_->chunkSize - 1) / tiling_->chunkSize;
}

template <typename T>
__aicore__ inline void ChunkGatedDeltaRuleFwdH<T>::InitState(
    const TaskInfo& task, const LocalTensor<T>& state)
{
    if ASCEND_IS_AIV {
        Duplicate(state, static_cast<T>(0), stateElements_);
        event_t zeroDone = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(zeroDone);
        WaitFlag<HardEvent::V_MTE2>(zeroDone);
        if (tiling_->hasInitialState != 0) {
            const int64_t base =
                (task.sequence * tiling_->valueHeads + task.valueHead) *
                tiling_->keyDim * tiling_->valueDim;
            DataCopyExtParams copy{1, static_cast<uint32_t>(task.validV * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
            for (int64_t row = 0; row < tiling_->keyDim; ++row) {
                DataCopyPad(
                    state[row * tiling_->vTileSize],
                    initialStateGm_[base + row * tiling_->valueDim + task.vStart],
                    copy, pad);
            }
        }
    }
}

template <typename T>
__aicore__ inline void ChunkGatedDeltaRuleFwdH<T>::StoreState(
    const GlobalTensor<T>& dst, int64_t baseOffset,
    const TaskInfo& task, const LocalTensor<T>& state)
{
    if ASCEND_IS_AIV {
        event_t vectorToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(vectorToMte3);
        WaitFlag<HardEvent::V_MTE3>(vectorToMte3);
        event_t mte2ToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3);
        WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3);
        DataCopyExtParams copy{1, static_cast<uint32_t>(task.validV * sizeof(T)), 0, 0, 0};
        for (int64_t row = 0; row < tiling_->keyDim; ++row) {
            DataCopyPad(
                dst[baseOffset + row * tiling_->valueDim + task.vStart],
                state[row * tiling_->vTileSize], copy);
        }
        event_t storeDone = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(storeDone);
        WaitFlag<HardEvent::MTE3_V>(storeDone);
    }
}

template <typename T>
__aicore__ inline void ChunkGatedDeltaRuleFwdH<T>::PackW(
    const TaskInfo& task, int64_t tokenStart, int64_t actualLen,
    const LocalTensor<T>& chunkLocal)
{
    if ASCEND_IS_AIV {
        Duplicate(chunkLocal, static_cast<T>(0), chunkElements_);
        event_t zeroDone = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(zeroDone);
        WaitFlag<HardEvent::V_MTE2>(zeroDone);
        const int64_t base =
            (task.valueHead * tiling_->totalTokens + tokenStart) * tiling_->keyDim;
        DataCopyExtParams copy{1, static_cast<uint32_t>(tiling_->keyDim * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        for (int64_t row = 0; row < actualLen; ++row) {
            DataCopyPad(chunkLocal[row * tiling_->keyDim],
                        wGm_[base + row * tiling_->keyDim], copy, pad);
        }
    }
}

template <typename T>
__aicore__ inline void ChunkGatedDeltaRuleFwdH<T>::PackK(
    const TaskInfo& task, int64_t tokenStart, int64_t actualLen,
    const LocalTensor<T>& chunkLocal)
{
    if ASCEND_IS_AIV {
        Duplicate(chunkLocal, static_cast<T>(0), chunkElements_);
        event_t zeroDone = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(zeroDone);
        WaitFlag<HardEvent::V_MTE2>(zeroDone);
        DataCopyExtParams copy{1, static_cast<uint32_t>(tiling_->keyDim * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        for (int64_t row = 0; row < actualLen; ++row) {
            const int64_t src =
                (tokenStart + row) * tiling_->keyHeads * tiling_->keyDim +
                task.keyHead * tiling_->keyDim;
            DataCopyPad(chunkLocal[row * tiling_->keyDim], kGm_[src], copy, pad);
        }
    }
}

template <typename T>
__aicore__ inline void ChunkGatedDeltaRuleFwdH<T>::ProcessTask(int64_t taskId)
{
    const TaskInfo task = DecodeTask(taskId);
    int64_t bos = 0;
    int64_t eos = 0;
    int64_t chunkOffset = 0;
    int64_t chunkNum = 0;
    GetSequenceRange(task.sequence, bos, eos, chunkOffset, chunkNum);

    LocalTensor<T> state;
    LocalTensor<T> chunkLocal;
    LocalTensor<T> stage;
    LocalTensor<T> deltaBf16;
    LocalTensor<half> halfWork;
    LocalTensor<float> calcWork;
    LocalTensor<float> matFloat;
    if ASCEND_IS_AIV {
        state = stateBuf_.Get<T>();
        chunkLocal = chunkBuf_.Get<T>();
        stage = stageBuf_.Get<T>();
        LocalTensor<half> halfHandle = halfWorkBuf_.Get<half>();
        deltaBf16 = halfHandle.ReinterpretCast<T>();
        halfWork = halfWorkBuf_.Get<half>();
        calcWork = calcWorkBuf_.Get<float>();
        matFloat = matFloatBuf_.Get<float>();
    }
    InitState(task, state);

    for (int64_t chunk = 0; chunk < chunkNum; ++chunk) {
        const int64_t globalChunk = chunkOffset + chunk;
        const int64_t tokenStart = bos + chunk * tiling_->chunkSize;
        const int64_t remaining = eos - tokenStart;
        const int64_t actualLen = remaining < tiling_->chunkSize ? remaining : tiling_->chunkSize;

        const int64_t hBase =
            (task.valueHead * tiling_->chunkCount + globalChunk) *
            tiling_->keyDim * tiling_->valueDim;
        StoreState(hGm_, hBase, task, state);

        PackW(task, tokenStart, actualLen, chunkLocal);
        Cast(matFloat, chunkLocal, RoundMode::CAST_NONE, chunkElements_);
        event_t matFloatReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(matFloatReady);
        WaitFlag<HardEvent::V_S>(matFloatReady);
        MatmulWStateP4(matFloat, state, stage, calcWork, actualLen);
        ComputeValueAndDecay(
            task, tokenStart, actualLen, stage, halfWork, calcWork, chunkLocal);

        PackK(task, tokenStart, actualLen, chunkLocal);
        Cast(matFloat, chunkLocal, RoundMode::CAST_NONE, chunkElements_);
        event_t matFloatReady = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(matFloatReady);
        WaitFlag<HardEvent::V_S>(matFloatReady);
        MatmulKtVDecaySeq(matFloat, stage, deltaBf16, calcWork, actualLen);
        MergeDelta(state, deltaBf16, halfWork, calcWork);
    }

    if (tiling_->storeFinalState != 0) {
        const int64_t finalBase =
            (task.sequence * tiling_->valueHeads + task.valueHead) *
            tiling_->keyDim * tiling_->valueDim;
        StoreState(finalStateGm_, finalBase, task, state);
    }
}

template <typename T>
__aicore__ inline void ChunkGatedDeltaRuleFwdH<T>::Process()
{
    const int64_t localTaskCount = tiling_->taskCountPerCore +
        (blockIdx_ < tiling_->taskTailCoreCount ? 1 : 0);
    const int64_t taskStart = blockIdx_ * tiling_->taskCountPerCore +
        (blockIdx_ < tiling_->taskTailCoreCount ? blockIdx_ : tiling_->taskTailCoreCount);
    for (int64_t i = 0; i < localTaskCount; ++i) {
        ProcessTask(taskStart + i);
    }
}

} // namespace NsChunkGatedDeltaRuleFwdH
#endif // CHUNKGATEDDELTARULEFWDH_H
