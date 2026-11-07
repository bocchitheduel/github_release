**结论：当前配置已经实测到约 1.54 倍训练吞吐量，单步耗时减少约 35%，峰值张量显存减少约 42%。第一阶段只需使用 PyTorch 自带的优化算子。**

本次仅新增评估脚本、结果和编译缓存。训练实现、数据和已有检查点未作修改，基准中的参数更新仅发生在内存中。

测试环境为 NVIDIA GeForce RTX 3070 Ti Laptop GPU，8 GiB 显存，PyTorch 2.14.0+cu126。模型约 2166 万参数；batch=128、sequence=256、4 层、d_model=512、d_ffn=1344、16 个注意力头、词表 8983。沿用 torch.compile、BF16 AMP 和 TF32。

| 累计替换内容 | 每步耗时，含数据读取 | 吞吐量 tokens/s | 相对吞吐量 | 峰值张量显存 |
| --- | ---: | ---: | ---: | ---: |
| 当前实现 | 236.4 ms | 138,630 | 1.00× | 6.57 GiB |
| 注意力换 SDPA / FlashAttention 后端 | 171.6 ms | 190,915 | 1.38× | 4.89 GiB |
| 再换 fused AdamW 和 foreach 梯度裁剪 | 157.9 ms | 207,483 | 1.50× | 4.89 GiB |
| 再换内置 cross_entropy，继续编译 | 153.7 ms | 213,186 | 1.54× | 3.79 GiB |

测量方法：各方案在独立进程中使用相同随机种子和真实 train.bin。先完成编译和 GPU 预热，再测量 5 组，每组 10 步，取每组平均耗时的中位数；组末同步 CUDA。对包含数据读取的测量，各方案先预读相同抽样序列，再恢复数据 RNG，以减少操作系统文件缓存偏差。梯度裁剪在两种实现中均保留非有限值检查。学习率固定为 1e-3；不包含验证、日志和检查点保存。显存为 torch.cuda.max_memory_allocated，不包含 CUDA 驱动及缓存分配器保留的全部显存。

按含数据读取的短基准外推，1 万步纯训练时间约从 39.4 分钟降至 25.6 分钟。完整训练任务仍需另计验证、保存、编译和长期运行中的温度、功耗变化。本次没有重跑 1 万步收敛实验，也未验证优化器替换后的旧检查点恢复。

**优先级最高的是注意力。** 当前 block_attention.py 的分块算法使用多个独立的 PyTorch 运算，并在块内将 Q/K/V 转成 FP32。它降低了注意力矩阵的存储需求，但仍有大量中间张量读写和 kernel 调用。SDPA 可直接调用融合的 FlashAttention 实现。[PyTorch SDPA 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

上一个会话已完成的注意力单项测量为：同样 B=128、H=16、L=256、D=32，前向加反向的当前编译版约 19.60 ms，SDPA 编译版约 3.13 ms，约 6.3 倍。其临时文件在会话中断后丢失，这组数字来自已返回的工具输出，未纳入本目录 summary.json。BF16 输出相对 L2 差异约 0.20%，三个输入梯度约 0.25%～0.29%；这仅是数值小测，不能代替完整训练验证。

**第二优先级是优化器、梯度裁剪和交叉熵。** 当前优化器逐参数启动多个运算；fused AdamW 和 foreach 裁剪能合并部分工作。内置交叉熵在本次测试中额外降低了约 1.10 GiB 峰值张量显存，速度收益相对小一些。PyTorch 官方也说明 foreach 和 fused 优化器通常快于逐张量循环。[PyTorch AdamW 文档](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html)

**进一步引入独立第三方库时，最值得评估 Liger 的 FusedLinearCrossEntropy。** 当前每批完整 logits 有 294,354,944 个元素，仅 BF16 存储就需要约 561 MiB。Liger 将词表投影与交叉熵分块融合，有机会继续降低显存。需要让训练损失直接接收最终隐藏状态和 lm_head 权重，并保留生成所需的 logits 接口。本环境未安装 Liger，因此不对额外加速倍数作承诺。[Liger 独立模型接入示例](https://linkedin.github.io/Liger-Kernel/Getting-Started/)

**文本生成还有独立的优化空间。** generate.py 当前逐 token 重新运行整个上下文，并计算所有位置的词表投影。应优先增加 KV cache、预分配缓存，并只对最新位置计算 lm_head。该改动需要适配 RoPE 偏移和窗口溢出规则，不能只替换一个注意力函数；本次未量化其速度收益。

后续已新增[推理专项评估](../inference/README.md)：SDPA 加末位置投影有明确实测收益，KV cache 在当前小模型、batch=1、eager 配置下的额外加速不稳定。推理改进顺序以技术报告第 8 节的新数据为准。

**数据准备目前优先级较低。** 已有记录显示，实际全量预处理使用 Julia，包含 I/O 和完整 ID 校验约 98.4 秒。不能把 Python BPE 换成 Rust 的潜在收益直接当成当前流水线收益。预热后本次 DataLoader 单项中位数约 1.38 ms，相对约 154 ms 的训练步骤占比很低。独立优化是将磁盘 token 存储从 int64 改为 uint16：在词表 8983 的条件下，训练文件可由约 4.06 GiB 降到 1.01 GiB，读取后仍转为 torch.long；需要同步更新元数据并检查恢复配置兼容性。

已有 10 月 15 日的训练日志约为 13.86 万 tokens/s，但其记录的 MH_selfattention.py 和 block_attention.py 哈希与当前实现不同。本次结论使用重新测量的当前代码。source_hashes.json 保存了被评估实现的哈希；summary.json 和四份 JSONL 保存整步训练测量。

复现命令：

```bash
.venv/bin/python -u assessments/external_libraries/benchmark.py suite
```

该命令会覆盖本目录同名测量结果，不写入模型检查点。短基准只覆盖当前无额外 attention mask 的训练路径；正式接入还需验证掩码、精度及检查点兼容性。
