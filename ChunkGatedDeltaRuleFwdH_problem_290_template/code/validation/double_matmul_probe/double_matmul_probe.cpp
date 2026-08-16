/*!
 * \file double_matmul_probe.cpp
 * \brief Compile-only probe for the two Matmul pipeline used by
 *        ChunkGatedDeltaRuleFwdH.
 *
 * This file is intentionally kept outside op_kernel/. It is compiled with
 * the real operator's generated compile metadata, but is never packaged as
 * the production kernel. The fixed shapes are:
 *
 *   MM1: [64, 128] x [128, 64] -> [64, 64]
 *   MM2: [128, 64] x [64, 64]  -> [128, 64]
 *
 * The probe verifies the interfaces needed by the planned kernel:
 *   - two high-level Matmul objects in one MIX kernel;
 *   - a VECCALC LocalTensor used as a Matmul input;
 *   - a synchronous Matmul result consumed by a Vector-side data movement;
 *   - the Vector-produced LocalTensor used by the second Matmul.
 *
 * The output of each Matmul is deliberately placed in GM workspace/output.
 * CANN 9.1's IterateAll(LocalTensor) requires a C tensor in L1 (A1/B1), so a
 * direct Cube-to-UB result is not a valid generic path on Ascend 910B.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace DoubleMatmulProbe {

constexpr uint32_t CHUNK_SIZE = 64;
constexpr uint32_t K_DIM = 128;
constexpr uint32_t V_TILE = 64;
constexpr uint32_t STATE_ELEMENTS = K_DIM * V_TILE;
constexpr uint32_t K_CHUNK_ELEMENTS = CHUNK_SIZE * K_DIM;
constexpr uint32_t CHUNK_TILE_ELEMENTS = CHUNK_SIZE * V_TILE;
constexpr uint32_t STATE_BYTES = STATE_ELEMENTS * sizeof(bfloat16_t);
constexpr uint32_t K_CHUNK_BYTES = K_CHUNK_ELEMENTS * sizeof(bfloat16_t);
constexpr uint32_t CHUNK_TILE_BYTES = CHUNK_TILE_ELEMENTS * sizeof(bfloat16_t);

struct ProbeTilingData {
    TCubeTiling mm1Tiling;
    TCubeTiling mm2Tiling;
};

using Mm1AType = MatmulType<TPosition::VECCALC, CubeFormat::ND, bfloat16_t>;
using Mm1BType = MatmulType<TPosition::VECCALC, CubeFormat::ND, bfloat16_t>;
using Mm1CType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;

using Mm2AType = MatmulType<TPosition::VECCALC, CubeFormat::ND, bfloat16_t, true>;
using Mm2BType = MatmulType<TPosition::VECCALC, CubeFormat::ND, bfloat16_t>;
using Mm2CType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;

using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

class ProbeKernel {
public:
    matmul::Matmul<Mm1AType, Mm1BType, Mm1CType, BiasType> mm1;
    matmul::Matmul<Mm2AType, Mm2BType, Mm2CType, BiasType> mm2;

    __aicore__ inline void Init(GM_ADDR k, GM_ADDR w, GM_ADDR initialState,
                                GM_ADDR h, GM_ADDR workspace, TPipe* pipe)
    {
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(k), CHUNK_SIZE * K_DIM);
        wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(w), CHUNK_SIZE * K_DIM);
        initialStateGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(initialState), STATE_ELEMENTS);
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(h), STATE_ELEMENTS);

        __gm__ uint8_t* userWorkspace = GetUserWorkspace(workspace);
        mm1ResultGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(userWorkspace), CHUNK_TILE_ELEMENTS);
        vDecayGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(userWorkspace + CHUNK_TILE_BYTES),
            CHUNK_TILE_ELEMENTS);

        if ASCEND_IS_AIV {
            pipe->InitBuffer(stateBuf_, STATE_BYTES);
            pipe->InitBuffer(kChunkBuf_, K_CHUNK_BYTES);
            pipe->InitBuffer(stageBuf_, CHUNK_TILE_BYTES);
        }
    }

    __aicore__ inline void Process()
    {
        LocalTensor<bfloat16_t> stateLocal;
        LocalTensor<bfloat16_t> kChunkLocal;
        LocalTensor<bfloat16_t> stageLocal;
        if ASCEND_IS_AIV {
            stateLocal = stateBuf_.Get<bfloat16_t>();
            kChunkLocal = kChunkBuf_.Get<bfloat16_t>();
            stageLocal = stageBuf_.Get<bfloat16_t>();
            DataCopy(stateLocal, initialStateGm_, STATE_ELEMENTS);
            DataCopy(kChunkLocal, wGm_, K_CHUNK_ELEMENTS);
        }

        // MM1: W[64, 128] x H[128, 64] -> WH[64, 64].
        mm1.SetTensorA(kChunkLocal);
        mm1.SetTensorB(stateLocal);
        mm1.template IterateAll<true>(mm1ResultGm_);
        mm1.End();

        // Compile-time stand-in for the real elementwise gate/decay stage.
        // The synchronous MM1 makes its GM result visible before this copy.
        if ASCEND_IS_AIV {
            DataCopy(stageLocal, mm1ResultGm_, CHUNK_TILE_ELEMENTS);
            DataCopy(vDecayGm_, stageLocal, CHUNK_TILE_ELEMENTS);
        }

        // MM2: K^T[128, 64] x V_decay[64, 64] -> deltaH[128, 64].
        // Passing stageLocal confirms that Vector-produced local data can be
        // consumed by the second high-level Matmul object.
        if ASCEND_IS_AIV {
            DataCopy(kChunkLocal, kGm_, K_CHUNK_ELEMENTS);
        }
        mm2.SetTensorA(kChunkLocal, true);
        mm2.SetTensorB(stageLocal);
        mm2.template IterateAll<true>(outputGm_);
        mm2.End();
    }

private:
    GlobalTensor<bfloat16_t> kGm_;
    GlobalTensor<bfloat16_t> wGm_;
    GlobalTensor<bfloat16_t> initialStateGm_;
    GlobalTensor<bfloat16_t> mm1ResultGm_;
    GlobalTensor<bfloat16_t> vDecayGm_;
    GlobalTensor<bfloat16_t> outputGm_;

    TBuf<TPosition::VECCALC> stateBuf_;
    TBuf<TPosition::VECCALC> kChunkBuf_;
    TBuf<TPosition::VECCALC> stageBuf_;
};

}  // namespace DoubleMatmulProbe

__global__ __aicore__ void chunk_gated_delta_rule_fwd_h(
    GM_ADDR k, GM_ADDR w, GM_ADDR u, GM_ADDR g, GM_ADDR initialState,
    GM_ADDR cuSeqlens, GM_ADDR chunkIndices, GM_ADDR h, GM_ADDR v,
    GM_ADDR finalState, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)u;
    (void)g;
    (void)cuSeqlens;
    (void)chunkIndices;
    (void)v;
    (void)finalState;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(DoubleMatmulProbe::ProbeTilingData);
    GET_TILING_DATA_WITH_STRUCT(DoubleMatmulProbe::ProbeTilingData, tilingData, tiling);

    TPipe pipe;
    DoubleMatmulProbe::ProbeKernel op;
    const TCubeTiling* mm1Tiling = &tilingData.mm1Tiling;
    const TCubeTiling* mm2Tiling = &tilingData.mm2Tiling;
    REGIST_MATMUL_OBJ(
        &pipe, GetSysWorkSpacePtr(), op.mm1, mm1Tiling, op.mm2, mm2Tiling);
    op.Init(k, w, initialState, h, workspace, &pipe);
    op.Process();
}
