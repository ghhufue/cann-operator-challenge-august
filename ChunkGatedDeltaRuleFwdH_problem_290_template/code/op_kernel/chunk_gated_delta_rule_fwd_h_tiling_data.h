/*!
 * \file chunk_gated_delta_rule_fwd_h_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _CHUNKGATEDDELTARULEFWDH_TILING_DATA_H_
#define _CHUNKGATEDDELTARULEFWDH_TILING_DATA_H_

#include "kernel_tiling/kernel_tiling.h"

struct ChunkGatedDeltaRuleFwdHTilingData {
    int64_t batch = 0;
    int64_t totalTokens = 0;
    int64_t sequenceCount = 0;
    int64_t chunkCount = 0;
    int64_t valueHeads = 0;
    int64_t keyHeads = 0;
    int64_t keyDim = 0;
    int64_t valueDim = 0;
    int64_t chunkSize = 64;
    int64_t headRatio = 0;

    // One logical task owns one (sequence, value_head, v_tile) recurrence.
    int64_t vTileSize = 0;
    int64_t vTileCount = 0;
    int64_t taskCount = 0;
    int64_t usedCoreNum = 0;
    int64_t taskCountPerCore = 0;
    int64_t taskTailCoreCount = 0;

    int64_t hasInitialState = 0;
    int64_t isVarLen = 0;
    int64_t storeFinalState = 0;

    // One reusable Cube/Vector interchange slot is reserved per MIX core.
    int64_t systemWorkspaceBytes = 0;
    int64_t perCoreWorkspaceBytes = 0;
    int64_t mm1WorkspaceOffset = 0;
    int64_t mm2WorkspaceOffset = 0;
    int64_t debugWorkspaceOffset = 0;
    int64_t debugWorkspaceBytes = 0;

    TCubeTiling mm1Tiling;  // [64, K] @ [K, vTile]
    TCubeTiling mm2Tiling;  // [K, 64] @ [64, vTile]
};
#endif
