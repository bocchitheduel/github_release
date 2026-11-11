# 推理模块专项评估

日期：2026-10-28。结论和表格已写入项目根目录 [TECHNICAL_REPORT.md](../../TECHNICAL_REPORT.md) 第 8 节。

本目录保留 2026-10-28 的推理基线、末位置投影/SDPA/KV cache 原型和 CUDA 基准脚本。之后正常推理源码已接入可切换的 SDPA 与末位置 logits，KV cache 仍只在评估原型中；历史结果不代表当前代码的完整请求性能。

## 环境与方法

- 已有 10,000 步检查点；BPE 词表 8,983；模型约 2166 万参数。
- RTX 3070 Ti Laptop GPU，PyTorch 2.14.0+cu126，batch=1，BF16，eager。
- TF32 matmul 保留推理进程的默认值 `False`；未套用训练入口的 TF32/compile 配置。
- 模型按正常 `load_model()` 路径在 inference_mode 之外加载；前向和生成使用 inference_mode。
- 受控前向从真实验证数据取固定 token 轨迹，不采样、不按 EOS 早停，各方案使用相同序列。首轮预热后，prefill 为 5 次中位数；decode 为 3 组各 16 步的组平均中位数，逐步同步 GPU。
- KV cache 仅接收一个样本：初始 prefill 后每步追加一个 token，超过 512 时重建缓存，以保持当前生成窗口语义。
- 基线真实请求直接调用当时的 generate()，temperature=0.8、top_p=0.9、seed=42，最多 128 个新 token；每条请求重复两次。
- 原型在同一进程中顺序比较、共享同一加载模型，未测试多进程服务并发。没有把短样本外推为请求 P95 或全服务吞吐。

## 主要发现

当前受控 decode 从 16-token 初始上下文的约 162 tokens/s，降至 256 时的约 39 tokens/s、512 时的约 19 tokens/s。SDPA 加末位置投影原型在 256 和 512 时分别约为 260、249 tokens/s。KV cache 在该小模型、batch=1、eager 配置中没有稳定额外加速。

FP32 缓存原型与 SDPA 全窗口的 logits 最大绝对差约 5.25e-6；BF16 相对 L2 差最大约 0.53%。均包含窗口边界检查。不同后端不保证相同的随机采样序列。

基线真实请求预热后的内部首 token 约 7～8 ms；英文长续写完成 89～128 个新 token，约需 0.8～1.2 秒。天空颜色问题仅输出换行后结束，中文指令得到英文故事片段。四个固定样例仅支持定性观察，不能形成统计性质量评分。当时的 Web UI 在整段结束后才返回文本。

## 当前源码更新（2026-11-07）

- 推理加载器默认选择 SDPA，`--attention-backend block` 可切回教学分块实现；直接构造的模型仍默认 `block`。
- `TransformerLM.forward_last_logits()` 避免对所有位置计算最终词表投影；生成器提供逐 token 流与 token 级计时，Web UI 可边生成边更新。
- Web UI 模型复用键现在包含规范化设备与注意力后端。
- 当前容器没有 CUDA，无法重跑 GPU profiler。CPU 小模型 profiler 记录 block/SDPA 分别 665/481 次算子调用；实测末位置前向中位数约 0.71/0.36 ms。简化 KV cache CPU 原型约 0.31 ms/token，全窗口重算约 0.35 ms/token，但算子调用由 481 增至 483。单层固定形状的 KV decode + `torch.compile` 小测由约 0.153 ms 降至 0.107 ms，首次编译约 10 秒。
- 历史 RTX 3070 Ti 测量显示 KV cache 相对 SDPA 末位路径没有稳定的额外收益，因此未将它接入默认生成路径。CPU 小模型结果不用于推断 GPU kernel 收益。

## 文件

| 文件 | 内容 |
| --- | --- |
| [benchmark.py](benchmark.py) | GPU 基准、真实请求和缓存数值验证原型 |
| [results.jsonl](results.jsonl) | 逐项原始结果与全部生成 token ID、文本 |
| [summary.json](summary.json) | 汇总后的延迟、吞吐、显存和数值差异 |
| [samples.md](samples.md) | 可阅读的 4 条完整生成样例 |
| [check_contracts.py](check_contracts.py) | 6 组 CPU 推理边界和采样检查 |
| [contracts.json](contracts.json) | 专项检查原始结果 |
| [unit_tests.log](unit_tests.log) | 4 项已有生成单测结果 |
| [source_hashes.json](source_hashes.json) | 2026-10-28 基线生成与模型核心源码的 SHA-256 |

## 复现

```bash
.venv/bin/python assessments/inference/check_contracts.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -m unittest discover -s tests -p test_generate.py -v
.venv/bin/python -u assessments/inference/benchmark.py
```

GPU 基准要求可用 CUDA，默认读取已有检查点和 `/path/to/tokenizer.toml`，支持用 `--checkpoint`、`--tokenizer` 指定路径。它会覆盖 results.jsonl、source_hashes.json，并自动更新 summary.json 和 samples.md；CPU 专项检查脚本会覆盖 contracts.json。技术报告中的静态表格需在重新基准后同步更新。
