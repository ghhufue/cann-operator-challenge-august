# ChunkGatedDeltaRuleFwdH 后续优化方向

## 1. 文档目的

本文记录当前生产 Kernel 完成正确性接入和设备编译后，可以继续研究的性能优化方向。

本阶段只保留优化设计，不立即修改代码。后续实施必须遵守以下原则：

- 不改变题目数学公式；
- 不改变 `h`、`v`、`final_state` 的输出含义和保存时机；
- 不改变参考实现规定的 FP16、FP32、BF16 舍入顺序；
- 一个 `(sequence, value_head, v_tile)` 任务内部仍然顺序执行全部 chunk；
- 每次只优化一条数据链路，并在编译和正确性验证通过后再进入下一步；
- 不在没有探针验证的情况下假设某个 `TPosition`、队列绑定或 Matmul 接口可用。

## 2. 当前实现基线

### 2.1 任务划分

当前逻辑任务为：

```text
(sequence, value_head, v_tile)
```

其中：

```text
VTile = min(V, 64)
```

每个任务持有一个 `[K,VTile]` 状态块，并在任务内部顺序遍历对应 sequence 的全部 chunk：

```text
H0 → chunk0 → H1 → chunk1 → H2 → ...
```

这部分符合真实递推依赖，后续优化不改变任务语义。

### 2.2 当前核心数据流

```text
H stateBuf_ Local
    ├─→ 写 h 输出
    └─→ MM1 B

W GM → 补零整理到 chunkBuf_ Local → MM1 A

MM1
 ↓
mm1ResultGm_（每核 workspace）
 ↓ DataCopy
stageBuf_ Local
 ↓
U-WH、保存 Vnew、门控衰减、H 衰减
 ↓
V_decay 保留在 stageBuf_ Local
 ↓
MM2 B

K GM → 按 key head 整理到 chunkBuf_ Local → MM2 A

MM2
 ↓
mm2ResultGm_（每核 workspace）
 ↓ DataCopy
chunkBuf_ Local
 ↓
Hnext = H_decay + deltaH
 ↓
stateBuf_ Local，继续下一个 chunk
```

当前已经具备：

- `MIX_AIC_1_2` Kernel；
- 两个 Ascend C 高阶 Matmul；
- H 状态跨 chunk 保持 Local，不从 GM 重新加载；
- `Vnew` 在 Local 中直接继续计算 `V_decay`；
- `V_decay` 通过 `SetTensorB(LocalTensor)` 直接进入 MM2，没有中间 GM 往返；
- W/K、门控临时量和 MM2 结果对 `chunkBuf_` 的生命周期复用；
- WH、U、Vnew、Vdecay 对 `stageBuf_` 的生命周期复用。

当前最主要的中间 GM 往返只有：

```text
MM1 → GM workspace → Vector
MM2 → GM workspace → Vector
```

## 3. 优先优化目标

后续优化的核心目标是逐条消除两个 Matmul 输出的 GM 往返：

```text
MM1 Cube
  ↓
CO2 → VECIN
  ↓
Vector Sub / Decay
  ↓
LocalTensor → MM2
  ↓
MM2 Cube
  ↓
CO2 → VECIN
  ↓
Vector Add
```

如果两条输出链路都能融合，理论上可以删除当前每核用户 workspace 中的：

```text
MM1 result [64,VTile] BF16
MM2 result [K,VTile] BF16
```

最大规格 `K=256、VTile=64` 下，每核用户 workspace 可由约 40 KiB 降到 0。高阶 Matmul 注册所需的 system workspace 仍可能保留。

## 4. 第一优先级：验证 Matmul 本地输出接口

### 4.1 为什么必须先做探针

CANN 9.1 高阶 Matmul 的：

```cpp
IterateAll(LocalTensor)
```

对 C Tensor 位置有限制，不能直接假设它可以输出到普通 Vector UB。

另一方面，当前版本还提供：

```cpp
Iterate(...)
GetTensorC(LocalTensor)
```

该接口是研究 `CO2 → VECIN` 的主要入口，但本地输出涉及 C 类型位置、NZ/ND 格式和 MIX 同步约束，必须先通过独立固定形状探针验证。

### 4.2 探针需要验证的内容

分别为 MM1 和 MM2 验证：

```text
MM1: [64,K] × [K,VTile] → [64,VTile]
MM2: [K,64] × [64,VTile] → [K,VTile]
```

检查以下项目：

1. `Iterate + GetTensorC(LocalTensor)` 能否在 Ascend910B 上生成设备目标；
2. C 类型应使用 `VECIN`、`CO2`、A1/B1 还是其他位置；
3. `TQueBind<TPosition::CO2,TPosition::VECIN>` 是否能与高阶 Matmul 配合；
4. 本地 C 输出是否只支持 NZ，是否支持 ND 或 ND_ALIGN；
5. 如果输出为 NZ，Vector 如何按正确逻辑读取 `[M,N]`；
6. 是否需要 NZ→ND 转换，以及转换开销是否仍低于 GM 往返；
7. `K=32/64/128/256`、`VTile<=64` 和最后一个 V tile 是否都能使用；
8. `MIX_AIC_1_2` 下两个 AIV 子核的同步和数据归属；
9. MM1、MM2 两个 Matmul 对象连续使用本地输出时是否存在事件冲突；
10. 本地输出是否仍需要隐藏的缓存 workspace。

探针只用于确认接口和编译能力，不直接替换生产 Kernel。

## 5. 第二优先级：优化 MM1 → Vector

当前路径：

```text
MM1
 ↓
mm1ResultGm_
 ↓ DataCopy
stageBuf_
 ↓
Vector
```

目标路径：

```text
MM1
 ↓
CO2 → VECIN
 ↓
stage/MMOut LocalTensor
 ↓
Vector
```

实施时只修改 MM1 输出，不同时改 MM2。必须保持：

- MM1 输出的 BF16 数据类型；
- MM1 结果转 FP16前的 NaN 处理；
- U、WH、Vnew 的参考舍入点；
- 尾 chunk 的无效行为零；
- `v` 保存未衰减的 `Vnew`，不能保存 `V_decay`。

成功后可以删除：

```text
mm1ResultGm_
mm1WorkspaceOffset
MM1 对应的每核 workspace 大小
MM1 结果的 GM→UB DataCopy
```

如果直接 ND 本地输出不可用，但 NZ 输出可用，需要先比较：

```text
CO2→VECIN + 本地布局转换
```

与：

```text
Cube→GM→UB
```

的成本，再决定是否采用。

## 6. 第三优先级：优化 MM2 → Vector

当前路径：

```text
MM2
 ↓
mm2ResultGm_
 ↓ DataCopy
chunkBuf_
 ↓
Vector Add
```

目标路径：

```text
MM2
 ↓
CO2 → VECIN
 ↓
deltaH LocalTensor
 ↓
Hnext = H_decay + deltaH
```

MM2 输出形状为 `[K,VTile]`，需要验证不同 K 下的本地输出布局。

成功后可以删除：

```text
mm2ResultGm_
mm2WorkspaceOffset
MM2 对应的每核 workspace 大小
MM2 结果的 GM→UB DataCopy
```

必须保持参考实现的合并顺序：

```text
MM2 result
→ FP16
→ NaN 转 0
→ BF16

H_decay
→ FP16
→ BF16

两项以 BF16 语义相加
→ Hnext
```

## 7. Vector → MM2 的后续调整

当前 `V_decay` 已经通过 `SetTensorB(stage LocalTensor)` 直接进入 MM2，没有 GM 往返。因此这条链路不需要作为首要重构对象。

后续可以研究显式的：

```text
VECOUT → B1/B2
```

或绑定队列，以减少同步开销、明确生产者/消费者关系。但修改前必须比较它与当前 `VECCALC LocalTensor` 输入路径的实际差异。

不能为了“看起来使用了 CV Fusion”而把当前已经无 GM 的数据流改得更复杂。

## 8. K 输入路径保持策略

原始 K 布局为：

```text
[B,T,HK,K]
```

选择一个 key head 后，相邻 token 行之间的跨度是：

```text
HK × K
```

因此目标 key head 的 `[64,K]` 数据不是紧凑矩阵。当前先整理到 `chunkBuf_` 再作为转置 A 输入是安全路径。

除非通过探针确认高阶 Matmul 可以正确描述该动态 leading stride，否则保留：

```text
K GM → PackK LocalTensor → MM2 A
```

不能未经验证直接改成 `Kᵀ GM → Cube`，否则会读到其他 key head 或错误的 token 行。

## 9. H 状态策略

当前 H 已经在 `stateBuf_ [K,VTile]` 中跨 chunk 常驻。每个 chunk 开头写入 GM 的 H 是题目要求的输出 `h`，不是计算状态 spill。

当前路径为：

```text
stateBuf_ 当前 H
├─→ StoreState 到 h
├─→ MM1
├─→ 原地计算 H_decay
└─→ 加入 deltaH，形成下一 chunk 的 H
```

因此暂不引入 Hcur/Hnext 双缓冲。最大规格下第二份 H 会额外占用 32 KiB UB，只有在以下情况得到证据后才考虑：

- 本地 MM2 输出无法与 stateBuf_ 安全合并；
- ping-pong 可以显著减少同步；
- 能形成真实 Cube/Vector 重叠；
- UB 预算允许且性能数据证明收益大于占用。

## 10. Local Memory 优化方向

### 10.1 当前主要 Local Buffer

| 名称 | Shape | Dtype | 当前位置 | 生命周期/复用 |
|---|---:|---|---|---|
| `stateBuf_` | `[K,VTile]` | BF16 | VECCALC | 整个 task，跨全部 chunk |
| `chunkBuf_` | `[64,K]` | BF16 | VECCALC | 复用为 W、门控临时区、K、deltaH |
| `stageBuf_` | `[64,VTile]` | BF16 | VECCALC | 复用为 WH、U、Vnew、Vdecay |
| `halfWorkBuf_` | `max(K×VTile,64×VTile)+VTile` | FP16 | VECCALC | 保存参考舍入点和逐行临时量 |
| `calcWorkBuf_` | `[2,VTile]` | FP32 | VECCALC | 逐行 Sub、Mul、Add 临时量 |

最大 `K=256、VTile=64` 时，当前显式 UB 使用约 104.6 KiB。

### 10.2 可继续研究的复用

在完成 Matmul 本地输出后，检查：

- MM1Out 与 MM2Out 生命周期不重叠，能否复用同一个队列或 Local Buffer；
- `stageBuf_` 是否可以直接承接 MM1 本地输出；
- MM2 本地输出是否可以复用 MM1 已释放的输出区；
- `chunkBuf_` 的 W、门控、K、deltaH 复用在删除 GM 输出后是否仍安全；
- `halfWorkBuf_` 是否可以按行缩小，而不是保存完整 WH FP16；
- 是否存在因频繁逐行事件同步造成的性能损失。

在正确性稳定前，不加入多组 double buffer 或异步流水。

## 11. 精度前置工作

进行 CV Fusion 前，必须先明确并补齐当前基线中的特殊值语义。参考实现要求：

```text
MM1结果 → FP16 → NaN转0
U → finite overflow饱和的FP16
U-WH → finite overflow饱和的FP16
gate → finite overflow饱和的FP16
V_decay → finite overflow饱和的FP16
H_decay → finite overflow饱和的FP16
MM2结果 → FP16 → NaN转0
H_decay与deltaH分别转BF16后相加
```

当前实现尚未显式完成：

- MM1 结果 NaN 清零；
- MM2 结果 NaN 清零；
- `cast_to_float16` 对有限溢出值的显式饱和；
- NaN/Inf 在 Cast 中的真机行为验证。

这些问题应在性能改造前建立独立测试，否则 Fusion 后出现误差时无法判断根因。

当前算子公开输入只支持 BF16，因此后续测试范围是：

```text
BF16 输入/输出
+ 内部 FP16 舍入语义
```

暂不扩展 FP16 输入类型。

## 12. 测试基础建设

当前已经有 Shape/Tiling Host UT 和独立 Golden 生成器，但生产 Kernel 还没有可用的端到端 correctness 回归。

现有 Kernel CPU UT 中的 workspace、Matmul tiling 和变长数据仍是旧占位配置，不能用于验证当前双 Matmul MIX Kernel。

正式性能修改前应先建立：

1. 合法的小尺寸固定长度用例；
2. 合法的多 sequence 变长用例；
3. `K=V` 为 32、64、128、256；
4. 长度 1、63、64、65、127、128、129；
5. `HV/HK` 为 1、2、4 等不同比例；
6. `initial_state` 有/无；
7. `final_state` 有/无；
8. 最后一个不足 64 的 chunk；
9. NaN、Inf、有限值 FP16 overflow；
10. 能逐级比较 WH、Vnew、Vdecay、Hdecay、deltaH 和 Hnext 的调试用例。

本机环境只能完成设备编译，不能真实运行 NPU。每一轮优化至少要完成编译探针和完整设备编译；最终 correctness 和性能必须在真实 Ascend910B/A2 环境验证。

## 13. 分阶段实施顺序

### 阶段 A：输出 Fusion 探针

- 不修改生产 Kernel；
- 验证 MM1/MM2 的 `Iterate + GetTensorC(LocalTensor)`；
- 确认 CO2、VECIN、NZ/ND 和 MIX 同步关系；
- 记录是否仍有隐藏 GM workspace。

### 阶段 B：只优化 MM1 输出

- MM1 改为本地输出；
- MM2 和其他逻辑保持不变；
- 编译并进行全部可用正确性回归；
- 删除 MM1 用户 workspace。

### 阶段 C：只优化 MM2 输出

- MM2 改为本地输出；
- 保持 Vector→MM2 现有 LocalTensor 输入；
- 编译并进行全部可用正确性回归；
- 删除 MM2 用户 workspace。

### 阶段 D：同步与 Buffer 优化

- 研究显式 TQue/TQueBind；
- 复用 MM1Out/MM2Out；
- 减少逐行事件；
- 根据真实性能数据决定是否引入 double buffer。

### 阶段 E：流水和规格优化

- 评估异步 Iterate/GetTensorC；
- 评估 Cube/Vector 跨阶段重叠；
- 针对 K/V=128 等高频规格选择专用 tiling；
- 评估更合适的 VTile。

## 14. 回退原则

如果某条 Fusion 链路在当前 CANN 9.1/Ascend910B 上无法实现，只回退该边。

例如 MM1 可以本地输出、MM2 不可以时，允许：

```text
MM1
 ↓ CO2→VECIN
Vector
 ↓ LocalTensor
MM2
 ↓ GM fallback
Vector
```

必须明确记录：

- 哪条边仍使用 GM；
- 对应的 API、格式或硬件限制；
- 已验证过哪些 TPosition 和接口；
- 下一步还可以验证什么。

不能因为一条边无法融合，就退回整个算子全部使用 GM 中转。

## 15. 后续每轮修改的汇报内容

每一轮优化完成后至少记录：

1. 修改前和修改后的实际数据流；
2. 哪些链路是 `CO2→VECIN`、LocalTensor 输入或 GM fallback；
3. 修改的文件、函数和 TPosition；
4. LocalTensor 的 shape、dtype、位置、生命周期和复用；
5. 修改前后的用户 workspace 与 system workspace；
6. 已执行的编译、正确性和特殊值测试；
7. MM1、MM2、Vector、GM 和同步的性能数据；
8. 尚未完成的 Fusion 链路及其具体原因。

最终优化目标是：

```text
H Local
   ↓
Cube MM1
   ↓
CO2 → VECIN
   ↓
Vector Sub / Decay
   ↓
LocalTensor → Cube MM2
   ↓
CO2 → VECIN
   ↓
Vector Add
   ↓
H Local
   ↓
下一 chunk
```

但是否采用每一条 Fusion 路径，以当前 CANN/硬件的探针结果、正确性和真实性能收益为准。
