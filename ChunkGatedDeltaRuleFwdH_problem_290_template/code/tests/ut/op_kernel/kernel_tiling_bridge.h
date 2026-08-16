#ifndef CHUNK_GATED_DELTA_RULE_FWD_H_KERNEL_TILING_BRIDGE_H
#define CHUNK_GATED_DELTA_RULE_FWD_H_KERNEL_TILING_BRIDGE_H

#include <cstddef>
#include <cstdint>

#include "../../../op_kernel/chunk_gated_delta_rule_fwd_h_tiling_data.h"

bool BuildKernelUtTiling(
    ChunkGatedDeltaRuleFwdHTilingData& tiling,
    size_t& workspaceBytes,
    int64_t& tilingKey);

#endif
