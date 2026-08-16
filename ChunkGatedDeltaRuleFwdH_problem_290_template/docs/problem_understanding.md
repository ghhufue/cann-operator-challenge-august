# ChunkGatedDeltaRuleFwdH 题意解读

## 1. 一句话说明

`ChunkGatedDeltaRuleFwdH` 不负责生成完整的注意力输出，也不负责生成 `w`、`u`、`g`。它接收上游已经准备好的中间张量，把每条序列按 64 个 token 分块，按 chunk 递推每个 value head 的隐藏状态，同时输出每个 token 修正后的 value。

可以把它理解为一个“分块执行、带遗忘门的矩阵记忆更新器”。

## 2. 名称拆解

- `Chunk`：每 64 个 token 为一个计算块，最后一个块可以不足 64。
- `Gated`：使用 `g` 控制旧状态和新信息的衰减。
- `Delta Rule`：通过 `u - w @ H` 只保留旧状态尚未包含的新信息。
- `FwdH`：沿序列向前递推隐藏状态 `H`。

这里的递推单位是 chunk，不是单个 token。同一序列、同一 value head 的 chunk 必须按顺序处理；chunk 内的最多 64 个 token 则通过矩阵乘并行处理。

## 3. 输入、输出和布局

以题目 3.1 的 PyTorch 参考实现为准，输入布局为：

| 张量 | 形状 | 含义 |
|---|---:|---|
| `k` | `[B,T,HK,K]` | key；`HK` 是 key head 数 |
| `w` | `[B,HV,T,K]` | 查询旧状态中已存在内容的系数 |
| `u` | `[B,HV,T,V]` | 当前 token 准备写入的新内容 |
| `g` | `[B,HV,T]` | chunk 内的累计门控值 |
| `initial_state` | `[N,HV,K,V]` | 每条实际序列的初始状态 |
| `cu_seqlens` | `[N+1]` | 打包变长序列的累计长度 |
| `chunk_indices` | `[NT,2]` | 每个全局 chunk 对应的 `(序列号, 序列内 chunk 号)` |

当前题目约束中：

- `B=1`；
- `N=token_batch`，范围为 1 到 4；
- `chunk_size=64`；
- `K=V`，范围为 32 到 256；
- `HV` 是 value/state head 数；
- `HK` 是 key head 数；
- 实现隐含要求 `HV % HK == 0`。

输出布局为：

| 张量 | 形状 | 实际语义 |
|---|---:|---|
| `h` | `[B,HV,NT,K,V]` | 每个 chunk 开始计算前的状态 |
| `v` | `[B,HV,T,V]` | 门控衰减前的 `u - w @ H` |
| `final_state` | `[N,HV,K,V]` | 每条序列最后一个 chunk 计算完成后的状态 |

题目的 Triton 注释使用过 `[B,T,HV,*]` 布局，但题目同时说明 Triton 仅供参考、以 3.1 为主，因此本项目按上表实现。

## 4. Value head 和 key head 的关系

每个 value head 都维护自己独立的状态：

\[
H_h \in \mathbb{R}^{K\times V}
\]

一个 token 会参与所有 value head 的计算，不存在把 token 动态路由给某一个 head 的过程。多个 value head 可以共用一个 key head，映射关系为：

\[
keyHead(h)=\left\lfloor\frac{h}{HV/HK}\right\rfloor
\]

例如 `HV=8、HK=2`：

- value head 0 到 3 使用 key head 0；
- value head 4 到 7 使用 key head 1。

共用 key 只表示它们使用相同的“写入地址方向”；这些 value head 的 `w`、`u`、`g` 和状态 `H` 仍彼此独立。

## 5. 单个 chunk 的计算

固定一条序列、一个 value head 和一个实际长度为 `L` 的 chunk，取出：

\[
K_c\in\mathbb{R}^{L\times K},\quad
W_c\in\mathbb{R}^{L\times K},\quad
U_c\in\mathbb{R}^{L\times V},\quad
g_c\in\mathbb{R}^{L}
\]

当前 chunk 开始前的状态为：

\[
H_c\in\mathbb{R}^{K\times V}
\]

### 5.1 计算需要写入的差量

\[
V_{new}=U_c-W_cH_c
\]

其中：

\[
[L,V]=[L,K]\times[K,V]
\]

`U` 表示准备写入的内容，`W @ H` 表示旧状态中已经能够解释的内容，二者之差才是需要新增或修正的信息。这就是 Delta Rule。

### 5.2 施加门控衰减

令当前 chunk 最后一个有效 token 的门控值为：

\[
g_{last}=g_c[L-1]
\]

新信息的逐 token 衰减为：

\[
\widetilde V_t=V_{new,t}\exp(g_{last}-g_t)
\]

旧状态的衰减为：

\[
\widetilde H_c=H_c\exp(g_{last})
\]

题目保证 chunk 内 `g` 严格递减且为负数，所以 `g_last - g_t <= 0`，新信息的衰减系数不大于 1；最后一个 token 的系数恒为 1。

### 5.3 更新状态

\[
H_{c+1}=\widetilde H_c+K_c^T\widetilde V
\]

也就是：

\[
H_{c+1}
=
\exp(g_{last})H_c
+
K_c^T\left[(U_c-W_cH_c)\odot\exp(g_{last}-g_c)\right]
\]

一个 chunk 的主要计算因此是两个矩阵乘：

1. `W @ H`：`[L,K] @ [K,V] -> [L,V]`；
2. `K.T @ V_decay`：`[K,L] @ [L,V] -> [K,V]`。

## 6. 三个输出的时间含义

假设一条序列有三个 chunk：

```text
H0 --chunk 0--> H1 --chunk 1--> H2 --chunk 2--> H3
```

则输出为：

```text
h = [H0, H1, H2]
final_state = H3
```

`h` 保存 chunk 的输入状态，而不是 chunk 计算后的状态。参考代码在进入每个 chunk 时先把当前状态放入 `S`，最后一个 chunk 之后的状态单独写入 `final_state`。

参考代码保存到输出 `v` 的是：

\[
V_{new}=U-WH
\]

门控衰减后的 `V_decay` 只在内部用于更新状态。题目表格中“输出 v 已经过 g 衰减”的描述与 PyTorch、Triton 的实际保存位置不一致，应以参考代码为准。

## 7. 变长序列与全局 chunk

例如：

```text
cu_seqlens = [0, 61, 64]
```

表示输入中打包了两条独立序列：

- 序列 0 占全局 token `[0,61)`，长度 61；
- 序列 1 占全局 token `[61,64)`，长度 3。

它们各自形成一个 chunk，而不是合并成一个长度 64 的 chunk：

```text
chunk_indices = [[0,0], [1,0]]
chunk_offsets = [0,1,2]
```

一般地，序列 `n` 的长度和 chunk 数为：

\[
L_n=cu\_seqlens[n+1]-cu\_seqlens[n]
\]

\[
C_n=\left\lceil L_n/64\right\rceil
\]

总 chunk 数为：

\[
NT=\sum_n C_n
\]

不同序列之间绝不能传递状态。每条序列从自己的 `initial_state[n]` 开始，最终写入自己的 `final_state[n]`。

## 8. 可以并行和不能并行的维度

同一序列、同一 value head 的 chunk 存在真实依赖：

```text
H0 -> chunk 0 -> H1 -> chunk 1 -> H2
```

因此不能把这些 chunk 当作独立任务直接并行。可以利用的并行维度包括：

- 不同序列；
- 不同 value head；
- 状态矩阵的不同 V 列分块；
- chunk 内 64 个 token；
- 两次矩阵乘内部的 K/V 元素和归约维度。

合理的设备任务粒度是 `(sequence, value_head, v_tile)`，每个任务内部顺序遍历该序列的 chunk。

## 9. 精度语义

参考实现并非全程 FP32，而是在多个位置显式舍入到 FP16/BF16：

1. `w @ H` 之后转 FP16；
2. `u` 先转 FP16，再参与减法；
3. `u - w @ H` 再转 FP16；
4. 门控系数转 FP16；
5. 门控后的 value 转 FP16，再转输入 dtype；
6. 衰减后的旧状态转 FP16；
7. `K.T @ V_decay` 之后转 FP16；
8. 两部分转 BF16 后相加。

因此“使用更高精度一次算完，最后转 BF16”不一定与参考结果一致。实现和测试都必须保留这些可观察的舍入边界。

`cast_to_float16` 还有特殊行为：有限值溢出时夹到 `[-65504,65504]`，原本就是 `NaN/Inf` 的值保留特殊值。矩阵乘结果中的 `NaN` 则按参考代码替换为 0。

## 10. 题面中需要规避的不一致

- 等长分支中 `N, NT, chunk_offsets` 的赋值数量和 `NT` 公式有笔误。
- 示例把二维 `chunk_indices` 命名成了 `chunk_offsets`。
- PyTorch `run()` 接收 `chunk_indices`，但实际使用重新计算的 `chunk_offsets`。
- `w` 的末维在表格中写成 `V`，矩阵语义应为 `K`；当前 `K=V` 掩盖了差异。
- Triton 注释中的 `w/u` 布局与 3.1 PyTorch 参考不同。
- 输出 `v` 的文字描述与参考代码的保存时机不同。
- 现有模板的 shape inference、tiling、kernel 和 UT 都是占位实现，不能作为算子语义依据。

## 11. 面向实现的等价伪代码

```python
for sequence in sequences:
    for value_head in range(HV):
        key_head = value_head // (HV // HK)
        H = initial_state[sequence, value_head]  # 不存在时为 0

        for chunk in sequence.chunks(size=64):
            h_output[value_head, chunk] = H

            K = k[chunk, key_head]
            W = w[value_head, chunk]
            U = u[value_head, chunk]
            G = g[value_head, chunk]

            Vnew = U - W @ H
            v_output[value_head, chunk] = Vnew

            g_last = G[-1]
            Vdecay = Vnew * exp(g_last - G)[:, None]
            Hdecay = H * exp(g_last)
            H = Hdecay + K.T @ Vdecay

        final_state[sequence, value_head] = H
```

实际 AscendC 实现还必须加入参考代码规定的 FP16/BF16 转换、饱和与特殊值处理。
