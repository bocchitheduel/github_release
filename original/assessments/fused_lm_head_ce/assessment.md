# Fused LM Head + Cross Entropy 试验

日期：2026-11-10。设备为 NVIDIA GeForce RTX 3070 Ti Laptop GPU，PyTorch 2.14.0+cu126，Liger-Kernel 0.8.5。测试使用实际 `train.bin`，主配置为 batch 128、sequence 256、d_model 512、4 层、词表 8,983，BF16 autocast 和 `torch.compile`。每个方案在独立进程运行，编译和预热后取 5 轮、每轮 10 步的平均值中位数。

## 结果

| 同一训练后端 | Native loss | Liger fused loss | fused 变化 |
| --- | ---: | ---: | ---: |
| 当前 block attention + 手写 AdamW | 297.73 ms/步，2,719.69 MiB 峰值 | 304.92 ms/步，2,720.36 MiB 峰值 | 慢 2.4%；峰值基本不变 |
| SDPA Flash + fused AdamW | 192.85 ms/步，1,829.86 MiB 峰值 | 201.07 ms/步，1,815.74 MiB 峰值 | 慢 4.3%；峰值少 14 MiB |

Liger 确实降低了前向阶段显存：当前 block 配置的前向峰值少约 126 MiB，SDPA 配置少约 311 MiB。整步峰值仍出现在反向阶段，因此这两种配置的总峰值几乎没有变化。相比本项目完整 BF16 logits 张量约 561 MiB 的理论大小，前向阶段节省量也会受同时存活的其他张量影响。

训练 loss 和裁剪前梯度范数在两个独立进程的首步结果相近：当前 block 方案 loss 相差约 1.2e-4，梯度范数相差约 0.035%。另用小张量直接对照时，loss 和输入梯度一致，LM Head 权重梯度相对 L2 差约 0.17%。单步 smoke 还通过正式训练循环完成了训练、验证和检查点保存。

## 接入与限制

训练通过 `--loss-backend liger_fused` 或 YAML 的 `loss_backend: liger_fused` 启用；默认仍是 `torch`。该路径把最终隐藏状态、`lm_head.W` 和展平目标交给 Liger，避免创建完整 logits；推理和普通模型调用仍返回原 logits。checkpoint 配置记录 loss backend，旧检查点可以按默认 `torch` 路径恢复，不能静默切换到 fused loss。

Liger 0.8.5 的 loss 调用不能完整交给当前 Inductor：它内部调用 `Tensor.item()`，捕获后会造成 graph break，并在本机环境触发 Inductor lowering 错误。实现把这一个自定义 autograd 调用留在 eager 路径，模型主干仍可 `torch.compile`。Liger forward 中的 `.item()` 会同步读取有效目标数量，是潜在耗时来源之一。

目前结论是：**功能可用，当前配置不适合默认启用。** 它降低前向临时显存，却没有降低整步峰值，且整步速度慢约 2–4%。只有当更大 batch/sequence 让前向 logits 成为峰值瓶颈时，才值得重新评估 chunk 大小和速度。

可复现命令：

```bash
.venv/bin/python -u assessments/fused_lm_head_ce/benchmark.py suite --attention-backend block --optimizer manual
.venv/bin/python -u assessments/fused_lm_head_ce/benchmark.py suite --attention-backend sdpa --optimizer fused
```

基准 JSONL 保存于本目录。此试验没有运行 10,000 步收敛对照。
