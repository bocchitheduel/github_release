# Triton 注意力修改与 FA2 对照（2026-11-11）

## 实现

- 默认反向使用 KV 外循环：每个 CTA 独占一个 KV tile，dK/dV 在寄存器中累计；遍历 Q tile 时通过 FP32 `tl.atomic_add` 累加 dQ。一个 tile 的 QK、P、dO @ Vᵀ 同时供 dQ/dK/dV 使用。
- 逐行预处理计算 FP32 `delta = sum(O * dO)`，并同时初始化 FP32 dQ workspace。缓存命中后，反向只有预处理、主计算、dQ 转换三个 kernel；取消了单独的 PyTorch 清零 kernel。
- 全有效因果块与对角块使用两个独立循环及 constexpr 分支。全有效块不执行逐元素 causal 比较，边界 load/store 保留掩码。反向不再遍历所有 Q tile 再跳过无效部分。
- 前向使用 exp2 在线 softmax，在遍历 KV 之前预缩放 Q；LSE 仍保存自然对数。反向复用预缩放 Q，固定 K 的缩放放在 Q 循环外。log2(e) 在 FP32 score 上相乘，避免额外生成、舍入一份 BF16 Q。
- FP16/BF16 dot 使用原生低精度 Tensor Core，累加器为 FP32；dK/dV 直接作为 tl.dot 累加器更新。
- 默认路径保留原始 Q/K/V 的 batch/head/seq/feature stride，在 kernel 内解码 pid_hb；没有 Python QKV expand 或 contiguous。连续输入仅使用 view。
- 前向反转 pid_m，使用 make_block_ptr 和 advance(..., (0, block_n))；反向 Q 指针按负 block_m 步长 advance。
- KV 反向 autotune 有 24 个候选组合，tile 覆盖 16–128，warp/stage 组合包括 (4,1)、(4,3)、(8,2)，按用户 tile 上限剪枝。缓存 key 包含 dtype、QKV stride 和 dO stride，超出设备资源的配置由 autotune 跳过。
- 前向保存的张量仍只有 Q、K、V、O 和 LSE。默认路径没有完整 attention 矩阵，额外的 delta 与 dQ workspace 均为线性空间。

代码入口：[block_attention_triton.py](../block_attention_triton.py)。比较用的修改前源码保存在 [快照](attention_kv_baseline_20261111.py)。

## 正确性

主环境 Python 3.14 / PyTorch 2.14.0+cu126 / Triton 3.8.0：

```bash
.venv/bin/python -m pytest -q tests/test_block_attention_triton.py
```

结果：**7 passed，12 subtests passed**。覆盖 FP32/FP16/BF16 前后向、AMP 和保存张量、非连续多头 stride、非方形序列、部分 tile、KV 完全位于因果区域之外、旧 atomic/partial 模式，以及 Sliced-Q。新增 CUDA Graph 回归测试连续重放三次，验证输出和梯度与 eager 一致。

同时完成 attention 文件和四个 benchmark/profile 脚本的 py_compile；Sliced-Q benchmark 的直接调用参数和反向配置输出也已修复，并完成一次 L=128 冒烟运行。

## 性能环境与方法

- GPU：NVIDIA GeForce RTX 3070 Ti Laptop GPU。
- 隔离基准环境 `.venv-attention`：Python 3.10、PyTorch 2.6.0+cu126、Triton 3.8.0、官方 FlashAttention 2.8.3。
- FA2 使用 [官方 v2.8.3 release](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3) 的 cp310 / CUDA 12 / torch2.6 / cxx11abiTRUE wheel。
- BF16，B=1、H=8、D=64，causal=True、dropout=0。两边使用相同 Q/K/V 和上游梯度。
- 编译和 triton.autotune 在计时之外完成。前向每图 128 次调用，前反向每图 64–128 步；CUDA events 测量整图，取 10 次重放的中位数。被测图自身预热至少 150 ms。计时区间没有逐步 Python 提交。
- GPU 未锁频，初期曾观察到明显轮间波动；早期的 167 µs 不作为稳定基线。以下使用同一轮、同一计时方法比较快照、本次实现与 FA2，结果只代表此设备和这些形状。

### 缓存后的时间（µs）

| L | 修改前 前+反 | 当前 前向 | FA2 前向 | 当前 前+反 | FA2 前+反 | 相对修改前耗时下降 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 16.42 | 5.36 | 9.10 | 15.40 | 29.47 | 6.2% |
| 512 | 79.29 | 22.87 | 26.15 | 74.11 | 89.04 | 6.5% |
| 1024 | 241.21 | 62.93 | 79.74 | 237.80 | 244.63 | 1.4% |
| 2048 | 760.31 | 186.76 | 221.71 | 703.22 | 696.21 | 7.5% |

本轮 L=128/512 本地实现更快；L=1024/2048 与 FA2 接近，2048 本地略慢约 1%。几%的差异需要结合锁频和重复实验判断，不能从本轮推广到其他 batch/head/dimension 或 FP16 工作负载。

### 当前缓存配置与误差

| L | 前向 | 反向 | 输出最大绝对误差 | 梯度最大绝对误差 |
| ---: | --- | --- | ---: | ---: |
| 128 | BM=32,BN=128,warps=4,stages=2 | BM=64,BN=32,warps=4,stages=1 | 0 | 0.0004882812 |
| 512 | BM=32,BN=128,warps=4,stages=2 | BM=32,BN=32,warps=4,stages=3 | 0.001953125 | 0.0078125 |
| 1024 | BM=64,BN=64,warps=4,stages=2 | BM=64,BN=64,warps=4,stages=1 | 0.00390625 | 0.015625 |
| 2048 | BM=64,BN=64,warps=4,stages=2 | BM=64,BN=64,warps=4,stages=1 | 0.00390625 | 0.0078125 |

误差比较的是同值输入下的 FA2 输出和全部 Q/K/V 梯度，均通过 rtol=atol=0.03 的 BF16 检查。非方形序列使用本地 PyTorch 参考验证；FA2 性能对照仅使用方形序列。

### Kernel 拆分（profiler，µs）

每个实现另采样 5 次 eager 步骤。该采样与整图计时的调用方式和频率状态不同，不应将它直接相加或相减来解释整图时间。

| L | 本地前向 | FA2 前向 | 本地主反向 | FA2 主反向 | 本地预处理+转换 | FA2 预处理+转换 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 5.68 | 10.09 | 8.96 | 18.13 | 3.38 | 5.19 |
| 512 | 18.35 | 22.25 | 41.54 | 56.65 | 5.42 | 7.38 |
| 1024 | 54.48 | 69.44 | 139.28 | 149.43 | 10.27 | 12.70 |
| 2048 | 146.96 | 186.73 | 375.73 | 349.61 | 24.39 | 26.53 |

两边均为每步 4 个 GPU kernel：前向、预处理、主反向、转换。主要耗时集中在前向和主反向，当前预处理没有独立清零 kernel。主反向里还包含 Tensor Core dot、布局转换、概率重算与原子累加，这份 profile 不能进一步把差异归因到其中某一项。

FA2 kernel 名显示前向 Q/KV tile 为 128/128、4 warps；反向为 64/128、8 warps。本地的 tile 更小、CTA 更多。由 tile 和网格可推断，这会影响小序列的并行度及共享内存占用；尚未采集硬件计数器证明具体占用率或 atomic stall 的贡献。

## 复现与数据

```bash
.venv-attention/bin/python scripts/benchmark_attention_autotune.py \
  --lengths 128 512 1024 2048 --rep-ms 20 --tile-cap 128 \
  --baseline-file reports/attention_kv_baseline_20261111.py \
  --profile-kernels --output reports/attention_fa2_steady_20261111.json

.venv-attention/bin/python scripts/profile_attention_kv.py \
  --transposed --lengths 128 --tile-cap 128 --rep-ms 5 \
  --output reports/attention_kv_final_20261111.json
```

- [最终整图时间、配置、误差与完整 kernel 名](attention_fa2_steady_20261111.json)
- [单个 KV kernel 的寄存器、spill 和共享内存采样](attention_kv_final_20261111.json)
- 其他 attention_*_20261111.json 是过程排查数据，部分使用此前的计时方法或合并清零前的版本，不与最终表格混用。
