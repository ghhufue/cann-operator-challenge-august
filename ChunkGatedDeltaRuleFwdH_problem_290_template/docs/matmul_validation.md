# 双 Matmul 与 MIX Kernel 验证结论

## 验证目标

在实现正式 Kernel 前，先用固定形状确认下面的数据通路能够通过 CANN 9.1.0 的 Ascend910B 设备编译：

```text
H LocalTensor、W 补零后的 LocalTensor
    ↓
MM1: W[64,K] @ H[K,VTile]
    ↓ GM workspace
Vector 中间处理
    ↓ LocalTensor
MM2: K^T[K,64] @ V_decay[64,VTile]
    ↓ GM workspace
Vector 更新 H
```

验证形状为 `K=128、VTile=64、chunkSize=64`，验证代码位于：

```text
code/validation/double_matmul_probe/
```

## 已验证结果

在本机 WSL 的 CANN 9.1.0 环境中，探针已成功生成 Ascend910B 设备目标文件。确认可用的组合包括：

- `KERNEL_TYPE_MIX_AIC_1_2`；
- 同一个 Kernel 通过 `REGIST_MATMUL_OBJ` 注册两个高阶 Matmul；
- MM1 的 W/H 和 MM2 的 K/Vdecay 均可由 `VECCALC LocalTensor` 输入；
- MM1 同步完成后由 Vector 侧消费结果；
- Vector 侧产生的 LocalTensor 作为 MM2 输入；
- 两个 Matmul 依次调用 `IterateAll<true>` 和 `End()`。

验证命令：

```bash
cd code/validation/double_matmul_probe
bash compile_probe.sh
```

## 发现的限制

CANN 9.1 的 `IterateAll(LocalTensor)` 不是“结果直接写普通 UB”的通用接口。该接口要求 Matmul 的 C 类型位于 L1 的 A1/B1 位置。因此首版正式 Kernel 不能假设以下路径天然成立：

```text
Cube MM result → 普通 VECCALC UB
```

首版采用保守且已经编译验证的数据通路：

```text
Cube → 每核 GM workspace → Vector UB
Vector UB → Cube → 每核 GM workspace → Vector UB
```

正式接入后又补做了 MM1 LocalTensor A 输入验证。原因是尾 chunk 不足 64 行时，直接让 Cube 从原始 W 的 GM 地址读取固定 `[64,K]` 会越过当前序列，张量末尾甚至可能越界。当前实现先把有效 W 行搬入 LocalTensor，其余行补零；MM1 完成后复用同一块 UB 整理按 key head 抽取的 K。

这会产生额外的 GM 往返，但不改变一个逻辑任务持有整条 chunk 状态链的任务模型。后续可以单独验证 L1 输出再搬到 UB，以减少往返。

## 对正式 Tiling 的影响

任务粒度保持为：

```text
(sequence, value_head, v_tile)
```

每个活跃 MIX 核组复用一个 workspace slot：

```text
slot base
├── MM1 result [64, VTile] BF16
└── MM2 result [K, VTile] BF16
```

当前 `VTile=min(V,64)`。每核空间按 512 字节对齐：

```text
mm1Bytes = 64 * VTile * 2
mm2Bytes = K  * VTile * 2
perCoreWorkspace = align512(align512(mm1Bytes) + mm2Bytes)
```

总 workspace 为高阶 API 系统空间加所有活跃逻辑核的用户空间：

```text
systemWorkspace + usedCoreNum * perCoreWorkspace
```

## 仍需真实 NPU 验证的内容

本机只能编译，不能运行设备 Kernel，因此以下内容尚未得到硬件验证：

- AIC/AIV 同步的运行时正确性；
- BF16 中间舍入与题目参考实现的逐级精度一致性；
- workspace 往返的实际性能；
- `VTile=64` 是否是所有 K/V 规格的最佳选择；
- MIX 1:2 下两个 AIV 子核的最终工作划分。

## 生产接入状态

截至当前版本，探针结论已经用于生产文件：

```text
op_kernel/chunk_gated_delta_rule_fwd_h.cpp
op_kernel/chunk_gated_delta_rule_fwd_h.h
op_host/chunk_gated_delta_rule_fwd_h_tiling.cpp
```

生产 Kernel 已通过 CANN 9.1.0 的 Ascend910B OPC 编译，并且 Host 侧 18 个 shape/tiling 测试全部通过。这里的“通过”仅代表接口、模板、设备指令和 Tiling 可以生成，不代表已完成真实 NPU 上的最终精度验收。
