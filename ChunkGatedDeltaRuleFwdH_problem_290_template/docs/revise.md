你现在需要优化 `ChunkGatedDeltaRuleFwdH` 算子的**核心 Kernel 计算流程**。

## 目标

先完整检查现有实现，在**不改变题目计算语义、输出语义和精度语义**的前提下，尽可能将当前实现优化为 Ascend 910B/A2 上的 **MIX + CV Fusion** 实现。

核心目标不是重写整个算子，而是减少：

```text
Cube → GM → Vector
Vector → GM → Cube
```

这类中间结果落 GM 的开销。

优先实现：

```text
Cube MM1
  ↓
CO2 → VECIN
  ↓
Vector
  ↓
VECOUT → Cube
  ↓
Cube MM2
  ↓
CO2 → VECIN
  ↓
Vector
```

---

## 一、第一步：先检查现有实现

不要立即修改代码。

先阅读并梳理当前项目中与该算子有关的：

* Host / Tiling
* Kernel
* MM1 Matmul 配置
* MM2 Matmul 配置
* workspace
* MIX/AIC/AIV 配置
* Queue / TBuf / LocalTensor
* 精度转换
* UT / correctness test

输出一个简短检查结果：

```text
1. 当前任务粒度是什么
2. MM1 输入输出目前放在哪里
3. MM1 → Vector 当前怎么传数据
4. V_decay → MM2 当前怎么传数据
5. MM2 → Vector 当前怎么传数据
6. H 当前存在哪里
7. 每个 chunk 是否把 H 写回 GM
8. 当前有哪些中间 workspace
9. 当前是否已经使用 MIX / CV Fusion
10. 哪些地方是主要 GM 往返
```

不要只根据变量名判断，必须跟踪实际 `LocalTensor / GlobalTensor / TPosition / DataCopy / Matmul` 数据流。

---

## 二、必须保持的计算语义

每个任务建议继续对应：

```text
(sequence, value_head, v_tile)
```

其中：

```text
VTile <= 64
```

每个任务内部必须**顺序执行该 sequence/value head 的所有 chunk**，因为：

```text
H0 → chunk0 → H1 → chunk1 → H2
```

存在真实递推依赖。

单个 chunk 的语义为：

```text
保存当前 H 到 h

MM1:
WH = W @ H

Vector:
Vnew = U - WH

保存：
v = Vnew

Vector:
V_decay = Vnew * exp(g_last - g)
H_decay = H * exp(g_last)

MM2:
deltaH = K^T @ V_decay

Vector:
Hnext = H_decay + deltaH

H = Hnext
```

最后：

```text
final_state = 最后一个 chunk 更新后的 H
```

必须注意：

```text
h 保存 chunk 开始前的 H
v 保存 Vnew = U - W@H
```

**不能把 V_decay 写到输出 v。**

这些语义必须以题目参考实现为准。

---

## 三、目标核心数据流

优先把当前 Kernel 改造成下面的结构：

```text
                 Hcur Local
                    │
          ┌─────────┴─────────┐
          │                   │
          ↓                   ↓
       h output              MM1
                              │
                       W × Hcur
                              │
                        CO2 → VECIN
                              │
                              ↓
                         WH Local
                              │
                   U ──────── Sub
                              │
                              ↓
                           Vnew
                         /       \
                        /         \
                  v output       decay
                                  │
                                  ↓
                              V_decay
                                  │
                           VECOUT → Cube
                                  │
                                  ↓
                                 MM2
                          Kᵀ × V_decay
                                  │
                            CO2 → VECIN
                                  │
                                  ↓
                               deltaH
                                  │
                         H_decay ─ Add
                                  │
                                  ↓
                               Hnext
                                  │
                           下一 chunk
```

重点检查是否可以使用：

```text
CO2 → VECIN
VECOUT → A1/A2
VECOUT → B1/B2
```

等 CV Fusion 数据路径。

---

## 四、MM1：优先实现 Cube → Vector Fusion

MM1：

```text
W[L,K] @ H[K,VTile]
      ↓
WH[L,VTile]
```

当前如果是：

```text
Cube
↓
GM workspace
↓
DataCopy
↓
UB
↓
Vector
```

优先修改。

目标是让 Matmul 的输出直接成为 Vector 侧 LocalTensor，类似：

```cpp
TQueBind<TPosition::CO2, TPosition::VECIN> mm1OutQueue;
```

并通过 Matmul 的 LocalTensor 输出接口取得 MM1 结果。

目标数据流：

```text
MM1 Cube
   ↓
  CO2
   ↓
 VECIN
   ↓
Vector Sub
```

不要为了 Fusion 而手写底层 Cube 指令，优先继续使用 Ascend C Matmul 高阶 API。

---

## 五、Vector：完成 Vnew 和 decay

得到 MM1 输出以后：

```text
WH
 ↓
Vnew = U - WH
```

这里需要：

1. 保持参考实现规定的 cast / FP16 舍入；
2. `Vnew` 写入最终输出 `v`；
3. 同一份 `Vnew` 继续用于计算：

```text
V_decay
```

尽量不要：

```text
Vnew → GM → 再读回来做 decay
```

优先在本地 Buffer 内完成：

```text
WH
↓
Vnew
├─→ CopyOut v
└─→ decay
```

---

## 六、MM2：重点实现 Vector → Cube Fusion

MM2：

```text
Kᵀ[K,L] @ V_decay[L,VTile]
          ↓
      deltaH[K,VTile]
```

这里是本次优化的重点。

当前如果：

```text
V_decay UB
↓
GM workspace
↓
Cube重新读取
```

优先尝试改成：

```text
V_decay
  ↓
VECOUT
  ↓
Cube Matmul input
```

由于：

```text
A = Kᵀ
B = V_decay
```

优先研究让 `V_decay` 作为 MM2 的 B 矩阵 LocalTensor 输入。

即：

```text
Kᵀ : GM → Cube A side

V_decay:
Vector
 ↓
VECOUT
 ↓
Cube B side
```

优先确认当前 CANN / AscendC 版本：

```text
Matmul::SetTensorB(LocalTensor)
```

是否支持当前数据位置和 dtype。

如果 B 方向不适合，再评估交换 MM2 表达方式、A/B 位置或其他官方支持的 CV Fusion 路径。

**禁止未经验证直接假设某个 TPosition 可用。**

---

## 七、MM2 → Vector

MM2 输出：

```text
deltaH[K,VTile]
```

后面立即：

```text
Hnext = H_decay + deltaH
```

因此与 MM1 相同：

```text
MM2
 ↓
CO2
 ↓
VECIN
 ↓
Vector Add
```

尽量不要：

```text
MM2 → GM → UB → Add
```

---

## 八、H 状态尽量保持 Local

重点检查当前实现是否每个 chunk 都：

```text
H → GM
下一 chunk
GM → H
```

如果存在这种行为，尝试消除。

因为单个 task 的状态为：

```text
H = [K, VTile]

K <= 256
VTile <= 64
```

优先尝试设计：

```text
Hcur
Hnext
```

两个 Local Buffer ping-pong：

```text
chunk0:
Hcur → Hnext

chunk1:
Hnext → Hcur

chunk2:
Hcur → Hnext
```

每个 chunk 开头：

```text
Hcur
 ├─→ 写 h 输出
 └─→ 继续参与 MM1
```

不要因为需要输出 `h`，就把计算状态本身强制落 GM。

最后一个 chunk 完成以后再：

```text
Hcur/Hnext → final_state
```

---

## 九、Local Memory 重新规划

检查 UB / Local Memory 是否可以围绕下面的生命周期复用：

```text
Hcur      [K,VTile]
Hnext     [K,VTile]

MMOut     max(
              MM1 [L,VTile],
              MM2 [K,VTile]
          )

VTemp     [L,VTile]

U / G 临时 Buffer
```

重点利用：

```text
MM1Out 和 MM2Out 生命周期不重叠
```

因此优先研究是否能够复用同一块 Local Buffer。

同时检查：

* double buffer 是否真的有收益；
* 是否因为过度 double-buffer 导致 UB 不够；
* MM1/MM2 是否可以共享 Buffer；
* Hcur/Hnext 是否必须双 Buffer；
* `Vnew` 和 `V_decay` 是否可以原地复用。

---

## 十、不要破坏精度语义

本题不能简单改成：

```text
全部 FP32
最后一次性转 BF16
```

必须检查参考实现要求的每一个可观察舍入点。

尤其包括：

```text
MM1结果 → FP16
U → FP16
U - WH → FP16

门控系数 → FP16
V_decay → FP16

H_decay → FP16

MM2结果 → FP16

H_decay / MM2结果
→ BF16
→ Add
```

还需要保持：

* FP16 finite overflow saturation；
* NaN / Inf 行为；
* Matmul 结果 NaN 的参考处理逻辑。

如果 CV Fusion 改变了中间 Tensor dtype，必须主动插入 Cast，使结果保持参考实现语义。

---

## 十一、实现策略

按以下顺序修改，禁止一次性大改：

### Step 1：只整理数据流

先标出当前：

```text
GM
L1
L0
CO2
VECIN
VECOUT
UB
```

各 Tensor 所在位置。

不修改算法。

### Step 2：优化 MM1 → Vector

优先实现：

```text
MM1 CO2 → VECIN → Sub
```

完成后跑全部 correctness test。

### Step 3：优化 MM2 → Vector

实现：

```text
MM2 CO2 → VECIN → Add
```

再次测试。

### Step 4：优化 Vector → MM2

这是风险最高的一步。

尝试：

```text
V_decay VECOUT → MM2
```

确认 Matmul LocalTensor 输入能力、TPosition 和同步关系。

再次测试。

### Step 5：H 状态本地化

尝试让：

```text
Hcur/Hnext
```

跨 chunk 保持 Local。

再次测试。

### Step 6：Buffer 生命周期优化

最后再考虑：

```text
buffer reuse
ping-pong
async Iterate
Cube / Vector overlap
```

不要在正确性尚未稳定时同时加入复杂流水。

---

## 十二、CV Fusion 不可实现时的回退原则

目标是**尽可能 Fusion**，但不能为了 Fusion 强行使用未被当前硬件/CANN 支持的路径。

如果某一条数据流确认无法直接融合，例如：

```text
VECOUT → MM2 B
```

当前 API/版本不支持，则只对该边回退：

```text
V_decay
↓
GM workspace
↓
MM2
```

不要因此退回成整个算子全部：

```text
Cube → GM → Vector → GM → Cube
```

允许形成混合方案，例如：

```text
MM1
↓
CO2→VECIN
↓
Vector
↓
GM
↓
MM2
↓
CO2→VECIN
↓
Vector
```

目标是**能省一段 GM 往返就省一段**。

---

## 十三、不要做的事情

不要：

1. 改变数学公式；
2. 改变 `h / v / final_state` 的保存时机；
3. 把不同 chunk 并行执行；
4. 把不同 sequence 的 H 混在一起；
5. 为了性能删除参考实现要求的 Cast；
6. 自己重新实现底层 Cube 微指令，除非 Matmul 高阶 API 明确无法完成；
7. 在没有确认 API 支持的情况下猜测 `TPosition`；
8. 一次同时重写 Host、Tiling、Kernel 和精度逻辑；
9. 为了 CV Fusion 引入比原实现更大的全局 workspace。

---

## 十四、最终需要给我的反馈

完成检查和每轮修改后，用下面格式汇报：

### 1. 修改前数据流

```text
MM1:
...

MM1 → Vector:
...

Vector → MM2:
...

MM2 → Vector:
...

H recurrence:
...
```

### 2. 修改后数据流

明确标记：

```text
CO2 → VECIN
VECOUT → Cube
GM fallback
```

分别出现在哪里。

### 3. 修改文件

列出：

```text
文件
函数
主要修改
```

### 4. Local Memory

列出主要 LocalTensor：

```text
名称
shape
dtype
TPosition
生命周期
是否复用
```

### 5. Workspace

告诉我：

```text
修改前大小
修改后大小
哪些 workspace 被删除
哪些仍必须保留
```

### 6. 正确性

至少检查：

```text
不同 K/V
不同 chunk 长度
最后一个不足64的chunk
多 sequence
多 HV/HK
initial_state有/无
FP16/BF16
NaN/Inf/overflow相关case
```

### 7. 性能判断

不要只给总耗时。

至少分析：

```text
MM1 Cube时间
MM2 Cube时间
Vector时间
GM搬运量
Cube等待Vector时间
Vector等待Cube时间
```

判断瓶颈究竟是：

```text
Cube
Vector
GM bandwidth
CV同步
小矩阵利用率
```

### 8. 仍未完成的 CV Fusion

如果某条链路仍然需要 GM，例如：

```text
V_decay → GM → MM2
```

必须明确写：

```text
为什么暂时不能Fusion
对应API/硬件限制是什么
下一步可以验证什么
```

不要隐藏 fallback。

---

最终目标不是“代码看起来用了 MIX”，而是让核心循环尽可能真正成为：

```text
H Local
   ↓
Cube MM1
   ↓
CO2→VECIN
   ↓
Vector Sub / Decay
   ↓
VECOUT→Cube
   ↓
Cube MM2
   ↓
CO2→VECIN
   ↓
Vector Add
   ↓
H Local
   ↓
下一 chunk
```

并在保持题目精度和输出语义完全正确的前提下，最大限度减少中间结果进入 GM。
