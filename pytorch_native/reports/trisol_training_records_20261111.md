# Trisol TinyStories 训练记录核对

核对日期：2026-11-11（Asia/Shanghai）
查询范围：当前登录身份在 `yh-team` 可见的 Trisol 训练作业，以及本地 `cs336assignment1` 项目文件。

## 核对结论

Trisol 中存在一场完成 10,000 步的 TinyStories Muon+AdamW 训练，但该作业实际申请 **1 张 A100**。另有一场使用 **2 张 A100** 的 PyTorch DDP 权重续训，完成 64 步。当前身份可见的作业列表中未找到 10,000 步双 A100 训练；若该任务在其他团队或身份下运行，当前查询看不到它。

本地先前保留了双卡 64 步作业的工作流报告和训练脚本；没有找到 10,000 步云端作业的完整日志或检查点。本次从 Trisol 指标 API 导出该 10,000 步作业每 100 步的训练指标至 [CSV 快照](trisol_muon_a100_10k_20261109_metrics.csv)。

## 完成 10,000 步的 Muon+AdamW 训练

| 项目 | 记录 |
| --- | --- |
| 作业 | `cs336-tiny-muon-b512-s1024-10k-20261109`，ID `2119631749209141248`，状态 `succeeded` |
| 资源 | 1 × NVIDIA A100-SXM4-80GB，world size 1 |
| 初始化 | seed 42 随机初始化；输入模型 `kasamura-cs336-tinystories-random-init-20261109:2` |
| 输出模型 | `kasamura-cs336-tiny-b512-s1024-10k:1` |
| 数据 | TinyStories token 数据集；另挂载训练入口数据集 `cs336-train-b512-s1024-20261109` |
| batch 与长度 | 有效 batch 512，microbatch 128，梯度累积 4；序列长度 1,024 |
| 更新与 token 数 | 10,000 步；5,242,880,000 token |
| 优化器 | RMS 对齐 Muon + AdamW；Muon 参数 12,451,840，AdamW 参数 9,203,200 |
| 学习率 | Muon 峰值 0.002，AdamW 峰值 0.001；500 步 warmup 后余弦衰减；Muon weight decay 0，AdamW 0.01 |
| 结果 | 末步训练 loss 1.099443；step 10,000 进度指标 `train/loss` 1.090909；validation loss 1.116261 |
| 性能 | 训练耗时 11,169.788 秒；step 10,000 报告约 471,185 tokens/s |
| 资源监控 | 平均 GPU 利用率约 98.3%；峰值显存 39,902,511,104 bytes（约 37.2 GiB） |
| 模型与费用 | 新模型版本已注册；账单状态 `settled`，但最终金额字段为空 |

训练完成事件与进度指标的训练 loss 口径不同：`1.099443` 是最终事件中的末步 loss；`1.090909` 是 Trisol 在 step 10,000 记录的 `train/loss` 值。指标 API 共返回 100 个点，间隔 100 步。验证 loss 来自 `training_complete` 日志事件，指标 API 没有提供 `eval/loss` 点。

此结果不是本机 10,000 步对照实验的受控复现。云端作业使用随机初始化、有效 batch 512、序列长度 1,024，以及不同的数据打包脚本和验证实现；不能单凭较低 validation loss 推断模型或优化器更优。

## 双 A100 DDP 短续训

| 项目 | 记录 |
| --- | --- |
| 作业 | `cs336-tinystories-b256-s512-r2-20261107`，ID `2119016566531047424`，状态 `succeeded` |
| 资源 | 2 × NVIDIA A100-SXM4-80GB，world size 2 |
| 输入与步数 | AdamW 模型权重来自 step 10,000；完成 64 步至 step 10,064 |
| batch 与长度 | 每卡 batch 256，序列长度 512；每步合计 262,144 token |
| 新增 token | 16,777,216 |
| 结果 | 末步训练 loss 1.504097；单次抽样 validation loss 1.500642；训练计时 53.972 秒 |
| 资源监控 | 平均 GPU 利用率约 73%；峰值显存 41,256,222,720 bytes（约 38.4 GiB） |
| 输出 | 云端 `checkpoint-step-00010064`，opaque/custom 可续训格式；未下载或注册成可部署模型 |

该作业加载模型权重，但新建 AdamW 优化器，没有恢复输入 checkpoint 的优化器动量状态。Trisol 指标 API 对它返回 `no_metrics_reported_terminal`，因此训练和验证 loss 取自作业日志，不是指标时间序列。

## 任务列表检索

只读命令 `bohr trisol train list --team yh-team --scope all --all -o json` 在核对时返回 151 条可见作业。按作业资源中的 `gpu_count` 检查，14 条配置为 2 GPU；可识别的双卡作业包括 DDP 性能/通信测试、拓扑诊断和上述 64 步 TinyStories 续训。唯一在名称和配置中明确完成 10,000 步 TinyStories 训练的任务是前述单卡 A100 Muon+AdamW 作业。该列表只代表当前身份与团队权限范围内的可见记录。

主要云端核对命令：

```bash
bohr trisol train list --team yh-team --scope all --all -o json
bohr trisol train get 2119631749209141248 --team yh-team -o json
bohr trisol train logs 2119631749209141248 --team yh-team --history --tail 25
bohr trisol train metrics 2119631749209141248 --team yh-team \
  --key train/loss --key tokens_per_second --key muon_lr --key adamw_lr --key grad_norm -o json
bohr trisol train get 2119016566531047424 --team yh-team -o json
bohr trisol train logs 2119016566531047424 --team yh-team --history --tail 250
```
