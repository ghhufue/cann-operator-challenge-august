# 双 Matmul 编译验证

这个目录用于验证 `ChunkGatedDeltaRuleFwdH` 正式 Kernel 采用的核内计算结构，不会被 `op_kernel/CMakeLists.txt` 收集，也不会进入最终算子包。

`chunk_gated_delta_rule_fwd_h.cpp` 只是 CANN 编译驱动要求的同名入口，实际探针实现在 `double_matmul_probe.cpp`。

固定验证形状：

```text
MM1: W[64,128]   @ H[128,64]       -> WH[64,64]
MM2: K^T[128,64] @ V_decay[64,64]  -> deltaH[128,64]
```

验证内容：

- 同一 MIX Kernel 注册两个高阶 `Matmul` 对象；
- `W/H` 从 GM 搬入 `VECCALC LocalTensor` 后作为 MM1 输入；正式实现由此给尾 chunk 的无效 W 行补零；
- MM1 同步结束后，Vector 侧读取其中间结果；
- Vector 侧产生的 `LocalTensor` 直接作为 MM2 输入；
- 从原始 K 布局整理出的 `LocalTensor` 作为 MM2 的转置 A 矩阵；
- 使用 `KERNEL_TYPE_MIX_AIC_1_2`，匹配 Ascend910B 的 Cube/Vector 混合执行方式。

执行：

```bash
cd code/validation/double_matmul_probe
bash compile_probe.sh
```

默认使用 `/home/workspace/hvm/Ascend/cann-9.1.0`。也可以通过 `ASCEND_HOME_PATH` 指向其他 CANN 安装目录。

这个验证只证明代码生成和接口组合可用，不能代替真实 NPU 上的数值正确性和性能测试。两个 Matmul 当前使用 GM 作为输出，因为 CANN 9.1 的 `IterateAll(LocalTensor)` 要求输出位于 L1 的 A1/B1 位置，不能把结果直接写到普通 Vector UB。正式设计需要在以下两条路径中选择：

1. 首版使用每核 workspace 作为 Cube/Vector 交接区，优先保证实现稳定；
2. 后续验证 L1 输出再搬到 UB 的路径，减少 GM 往返。

`build/` 是验证生成物，不应提交。
