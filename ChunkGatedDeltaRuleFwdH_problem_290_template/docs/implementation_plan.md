# ChunkGatedDeltaRuleFwdH 实现计划

## 当前进度

- 阶段 1 已完成：Golden 生成器、4 个小型用例和 manifest 校验已实现。
- 阶段 2 的 shape inference 与静态参数校验已完成：支持标准变长输入、固定长度输入和可选 `initial_state/final_state`，并将公开数据类型收紧为题目要求的 BF16 主路径。
- 阶段 2 的 tiling 数据结构和 Host tiling 已完成：任务按 `(sequence, value_head, v_tile)` 划分，生成两份 `TCubeTiling`，并规划每核 Cube/Vector 交接 workspace。
- 固定形状双 Matmul MIX 编译探针已完成，确认两个 Matmul 对象和 Vector LocalTensor 输入组合可在 Ascend910B 上生成设备目标文件；详细结论见 `matmul_validation.md`。
- 生产 Kernel 的接入版已实现并通过 Ascend910B 设备编译：入口为 `MIX_AIC_1_2`，注册两个高阶 Matmul，完成任务解码、变长 chunk 寻址、每核 workspace、W/K 补零整理和片上状态递推。
- Vector 数值路径已按参考顺序接入：`U-WH`、保存 `v_new`、门控衰减、旧状态衰减和 `Hnext` 合并均保留 FP16/FP32/BF16 舍入点。
- WSL CANN 9.1.0 已成功编译完整算子包；15 个 Shape inference Host UT 和 3 个 Tiling Host UT 全部通过。该环境没有真实 NPU，因此尚不能把阶段 3 标为“数值正确性完成”。

当前阶段 3 的状态是“实现和编译完成、等待真机数值验证”。首轮真机验证必须重点检查：Matmul 结果中的 NaN 清零、`cast_to_float16` 对有限溢出的饱和，以及 `MIX_AIC_1_2` 两个 AIV 子核的执行行为。

## 1. 实现原则

整体分为“正确性基线”和“性能优化”两轮：

1. 先建立可信的 PyTorch Golden、明确布局和舍入边界；
2. 完成 shape inference、参数校验和 tiling；
3. 实现能够覆盖全部合法 shape 的 BF16 Kernel；
4. 在输出保持一致的前提下使用 Cube、片上状态常驻和流水线优化。

不采用跨 chunk 的并行前缀扫描作为首版方案。虽然递推可以在数学上改写为仿射变换，但会引入额外的 `K x K` 矩阵乘、额外存储和不同的浮点运算顺序，不利于参考精度对齐。

## 2. 阶段划分

| 阶段 | 内容 | 完成标准 |
|---|---|---|
| 1 | Golden 生成器和小尺寸用例 | 可重复生成输入、`h/v/final_state` 和 manifest；自检通过 |
| 2 | Shape inference、校验、tiling 数据 | Host UT 覆盖合法与非法 shape，输出 shape 正确 |
| 3 | BF16 正确性 Kernel | 小尺寸和 chunk 边界结果与 Golden 对齐 |
| 4 | 全规格和可选输入 | 覆盖 K/V 32 到 256、不同 HV/HK、变长和尾块 |
| 5 | Cube 与流水线优化 | 精度不回退，核心矩阵乘进入高性能路径 |
| 6 | 大规模与稳定性测试 | 长序列不越界、不爆 UB，性能曲线稳定 |

## 3. 阶段 1：Golden 基础设施

新增独立工具：

```text
code/tests/golden/generate_golden.py
```

职责：

- 按题目 3.1 的精度路径实现参考计算；
- 正确生成 `cu_seqlens`、`chunk_indices` 和 chunk offsets；
- 构造 chunk 内严格递减且为负数的 `g`；
- 覆盖部分 chunk、完整 chunk、跨 chunk 状态传递和 grouped heads；
- 保存原始 BF16 位模式，而不是先转 FP32 再落盘；
- 为每个 tensor 保存 shape、dtype、元素数、字节数和 SHA-256；
- 提供不落盘的自检模式。

计划用例：

- `smoke_single_partial`：一条长度 5 的部分 chunk；
- `boundary_63_64_65`：验证 63、64、65 三个关键边界；
- `boundary_127_128_129`：验证两个完整 chunk 附近的状态传递；
- `grouped_heads_packed`：4 条打包序列，`HV=8、HK=2`，验证 key head 映射。

运行方式：

```bash
python code/tests/golden/generate_golden.py --validate-only
python code/tests/golden/generate_golden.py --case smoke_single_partial --output-dir /tmp/chunk_golden
python code/tests/golden/generate_golden.py --list-cases
```

## 4. 阶段 2：Host 侧

### 4.1 Shape inference

从输入读取：

```text
B,T,HK,K = k.shape
HV,V      = u.shape[1], u.shape[3]
N         = cu_seqlens.shape[0] - 1
NT        = chunk_indices.shape[0]
```

推导：

```text
h           [B,HV,NT,K,V]
v           与 u 相同
final_state [N,HV,K,V]
```

校验：

- `B == 1`；
- `chunk_size == 64`；
- `K == V`；
- `HV % HK == 0`；
- `w/u/g` 的 B、HV、T 一致；
- `initial_state` 的 N、HV、K、V 一致；
- `cu_seqlens` 为 int64，`chunk_indices` 为 `[NT,2]` int64。

### 4.2 Tiling data

替换占位字段，至少携带：

```cpp
B, T, N, NT;
HV, HK, K, V;
chunkSize;
headRatio;
vTileSize;
vTileCount;
taskCount;
hasInitialState;
isVarLen;
storeFinalState;
```

后续根据 Matmul API 的要求补充 Cube tiling 和 workspace 数据。

## 5. 阶段 3：正确性 Kernel

### 5.1 并行任务

任务粒度：

```text
(sequence, value_head, v_tile)
```

任务总数：

\[
N\times HV\times\lceil V/VTile\rceil
\]

每个任务只维护状态的一个 V 列块：

\[
H[:,vStart:vEnd]
\]

任务内部按顺序遍历当前序列的所有 chunk。

### 5.2 单个 chunk 的设备流水

```text
保存当前 H 到输出 h
        ↓
MM1：W @ H
        ↓
Vector：精度转换、U-WH、保存 v、计算门控
        ↓
MM2：K^T @ Vdecay
        ↓
Vector：旧状态衰减、状态相加和舍入
        ↓
下一个 chunk
```

当前接入版使用 AscendC 高阶 Matmul/Cube 完成两个矩阵乘。由于普通 UB 不能作为通用的 `IterateAll` 输出，两个结果先落到每核 GM workspace，再由 Vector 搬入 UB。流水重叠和减少 workspace 往返延后到阶段 5。

### 5.3 片上状态

状态 tile 在整个 chunk 循环中常驻片上：

- `initial_state` 只读取一次；
- 每个 chunk 将当前状态写一次到 `h`；
- 不从 GM 重新读取上一个 chunk 的状态；
- 最终状态只写一次。

例如 `K=256、VTile=64` 时，BF16 状态占 32 KiB，适合作为初始评估点。实际 `VTile` 需要结合中间张量、Matmul 缓冲和 UB/L1 大小计算。

### 5.4 变长寻址

设备侧读取：

```text
bos = cu_seqlens[n]
eos = cu_seqlens[n+1]
seqLen = eos - bos
numChunks = ceil(seqLen / 64)
```

由于 `N<=4`，可以通过前面序列的长度在设备侧计算该序列的全局 chunk offset，无需新增输入：

```text
chunkOffset = sum(ceil(seqLen[i] / 64), i < n)
```

序列内第 `c` 个 chunk：

```text
globalChunk = chunkOffset + c
tokenStart  = bos + c * 64
actualLen   = min(64, eos - tokenStart)
```

最后一个 chunk 的 `g_last` 必须取 `actualLen-1`，无效行参与矩阵乘时补零。

## 6. 精度对齐策略

为每个中间量规定明确的舍入点，按照 Golden 逐级比较：

```text
WH
Vnew
gate
Vdecay
Hdecay
K^T @ Vdecay
Hnext
```

重点处理：

- `cast_to_float16` 对有限溢出值的饱和；
- 原始 `NaN/Inf` 的保留；
- 参考实现中矩阵乘结果 `NaN -> 0`；
- BF16 Matmul 输出与 FP16 转换顺序；
- 最终两项先转 BF16再相加的顺序。

若最终输出不一致，先用小用例找出第一个不一致的中间量，不直接在大规模最终输出上猜测。

## 7. 阶段 4：规格扩展

按以下顺序扩展：

1. 官方主路径：`B=1`、BF16、变长输入存在、`initial_state` 存在；
2. `initial_state` 为空时从零状态开始；
3. 最后一个 chunk 的任意实际长度；
4. `K=V` 的 32、64、128、256；
5. `HV/HK` 的不同整数比例；
6. 若评测真实覆盖，再支持定义文件中声明的 FP16 和等长可选输入路径。

定义文件当前同时声明 BF16/FP16，但题面核心约束只要求 BF16。实现时不能让 FP16 错误进入 BF16 模板；要么增加独立 tiling key 和模板实例，要么收紧公开类型声明。

## 8. 阶段 5：性能优化

优化顺序：

1. 状态跨 chunk 常驻片上；
2. 两个矩阵乘使用 Cube；
3. 当前 chunk 计算与下一 chunk 搬运双缓冲；
4. Cube 和 Vector 阶段流水重叠；
5. 根据 `N*HV` 和维度自适应选择 `VTile`；
6. 减少同一 key group 中 `k` 的重复搬运；
7. 针对 `K/V=128` 等高频规格提供专用 tiling key。

`VTile` 的基本权衡：

- tile 大：`w/g/k` 重复读取少，但任务数少、片上占用大；
- tile 小：并行度高，但同一 chunk 的共享输入会被更多任务重复读取。

## 9. 测试矩阵

正确性测试至少覆盖：

| 类别 | 取值 |
|---|---|
| 序列长度 | 1、5、63、64、65、127、128、129 |
| token batch | 1、2、4 |
| HV/HK | 1/1、2/2、4/2、8/2、16/2 |
| K=V | 32、64、128、256 |
| 初始状态 | 有、无 |
| chunk 尾部 | 完整、部分 |

性能测试使用更长序列，并分别观察：

- `N*HV` 足够大时的吞吐；
- 单序列、少 head 时 V 分块能否提供足够并行度；
- `h` 巨大输出带来的 GM 带宽上限；
- key 共享场景中的重复读取开销。

## 10. 文件实施顺序

1. `code/tests/golden/generate_golden.py`
2. `op_host/chunk_gated_delta_rule_fwd_h_infershape.cpp`
3. `op_kernel/chunk_gated_delta_rule_fwd_h_tiling_data.h`
4. `op_host/chunk_gated_delta_rule_fwd_h_tiling.cpp`
5. `tests/ut/op_host/test_chunk_gated_delta_rule_fwd_h_tiling.cpp`
6. `op_kernel/chunk_gated_delta_rule_fwd_h.h`
7. `op_kernel/chunk_gated_delta_rule_fwd_h.cpp`
8. Kernel Golden 对比与性能用例

每个阶段都以已有 Golden 为回归基线，不能为了性能改变可观察的输出语义。
