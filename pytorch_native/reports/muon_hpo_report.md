# RMS-to-RMS Muon 超参数搜索报告

## 实验配置

- 模型：4 层 Transformer，d_model=512、d_ffn=1344、16 头、词表 8,983，参数量 21,655,040。
- Muon 负责 12,451,840 个参数的 28 个投影矩阵；AdamW 负责 9,203,200 个 embedding、norm 和输出头参数。
- Muon scale：`sqrt(m/n)`，数学上等价于 `adjust_lr_fn=spectral_unclamped` 的形状倍率；所有候选固定使用 RMS-to-RMS。
- 搜索空间：Muon LR (0.00025, 0.0005, 0.001, 0.002)，Muon weight decay (0.0, 0.1)；AdamW LR=0.001、weight decay=0.01 固定。
- 固定 momentum=0.95、Nesterov=True、NS steps=5、系数=(3.4445, -4.775, 2.0315)、eps=1e-7。
- Batch=128、sequence=256、accumulation=1、BF16 autocast、TF32、global gradient clipping=1.0；统一 10 步线性 warmup 后 constant。
- 数据：train 544,548,083 tokens，validation 5,499,380 tokens；每个候选以相同随机种子重放同一训练窗口顺序和验证窗口。
- 无更新数值预检耗时 0.461s，模型权重未改变；12 个代表矩阵的 NS 输出均有限。

## 实际预算

累计 optimizer updates：1000 / 1000（已提交 checkpoint 的 updates：999；数值/OOM 失败更新：0；恢复时保守计费的未提交 reservation：1）。
各阶段目标为 8×25、4×75、2×150、1×200；任一失败配置的未使用预算会在搜索结束前交给当前最优可用配置。
本次在 T08 第 50 步验证期间中断；该次更新已计入预算但未提交到 checkpoint。恢复后从第 49 步 checkpoint 重放，故累计 1 次恢复 reservation，最终 T07 在剩余预算内完成 449 步；999 步提交到 checkpoint，累计预算计数为 1,000。

## 搜索记录

| Trial | Muon LR | Muon WD | 累计步数 | Best val | Final val | Final train | Stability | 平均 tokens/s | PyTorch peak allocated MiB | Selection / elimination |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|
| T01 | 0.00025 | 0 | 25 | 5.563569 | 5.563569 | 7.510281 | stable | 13830 | 6848 | stage 1: validation loss at 25 steps; uncertainty margin 0.0159135; near-ties use max update ratio then lower Muon LR |
| T02 | 0.00025 | 0.1 | 25 | 5.564314 | 5.564314 | 7.510487 | stable | 83128 | 7011 | stage 1: validation loss at 25 steps; uncertainty margin 0.0159135; near-ties use max update ratio then lower Muon LR |
| T03 | 0.0005 | 0 | 25 | 5.361387 | 5.361387 | 7.429969 | stable | 85171 | 7176 | stage 1: validation loss at 25 steps; uncertainty margin 0.0159135; near-ties use max update ratio then lower Muon LR |
| T04 | 0.0005 | 0.1 | 25 | 5.362602 | 5.362602 | 7.430430 | stable | 53928 | 7343 | stage 1: validation loss at 25 steps; uncertainty margin 0.0159135; near-ties use max update ratio then lower Muon LR |
| T05 | 0.001 | 0 | 100 | 3.735399 | 3.735399 | 3.822047 | stable | 41062 | 8211 | stage 2: validation loss at 100 steps; uncertainty margin 0.0162547; near-ties use max update ratio then lower Muon LR |
| T06 | 0.001 | 0.1 | 100 | 3.736241 | 3.736241 | 3.822517 | stable | 46287 | 8377 | stage 2: validation loss at 100 steps; uncertainty margin 0.0162547; near-ties use max update ratio then lower Muon LR |
| T07 | 0.002 | 0 | 449 | 2.326899 | 2.326899 | 2.373850 | stable | 86938 | 8544 | stage 3: validation loss at 250 steps; uncertainty margin 0.0178449; near-ties use max update ratio then lower Muon LR |
| T08 | 0.002 | 0.1 | 250 | 2.868001 | 2.868001 | 2.924764 | stable | 90646 | 8008 | stage 3: validation loss at 250 steps; uncertainty margin 0.0178449; near-ties use max update ratio then lower Muon LR |

## 阶段筛选

- Stage 1（25 步）：保留 T07, T08, T05, T06；minimum fixed-step validation cross-entropy with uncertainty-aware tie handling。
- Stage 2（100 步）：保留 T07, T08；minimum fixed-step validation cross-entropy with uncertainty-aware tie handling。
- Stage 3（250 步）：保留 T07；minimum fixed-step validation cross-entropy with uncertainty-aware tie handling。

## 最优配置

推荐 trial `T07`：Muon LR=0.002、Muon WD=0；final validation loss=2.326899，累计 449 步。
固定参数：momentum=0.95、Nesterov=True、NS steps=5、RMS-to-RMS (`spectral_unclamped` shape scale)、AdamW LR=0.001、AdamW WD=0.01、10 步 warmup 后 constant、gradient clipping=1.0。
该配置只经过本次实际累计步数验证，不能解释为 10,000 步完整训练验证。

## 数值稳定性与诊断

- NaN/Inf/OOM 失败 trial 数：0；显存列保留 PyTorch allocator 计数，物理显存峰值不由该计数推断。
- `diagnostics.jsonl` 按评估 checkpoint 记录 blocks.0 和 blocks.3 的 attention Q/K/V/output 与 MLP up/down 矩阵的权重、梯度、momentum、NS 更新、参数实际更新、update/weight ratio，以及 power iteration 近似的谱范数和 RMS-to-RMS operator norm。
- 最终代表矩阵 raw NS sigma_max 估计为 1.112–1.177（理想半正交方向约 1）；实际参数更新的 RMS-to-RMS operator norm 为 0.00222–0.00235，Muon LR=0.002，更新/权重比为 0.000924–0.00231。
- PyTorch `max_memory_allocated` 记录的最大值为 8544 MiB，超过设备声明的 8192 MiB；因此该 allocator 计数不能解释为物理 VRAM 峰值。本次未持久化逐步 NVML 显存采样，表中原样保留该计数并标明来源。
- T07 在 stage 4 的验证 loss 从 300 步 2.6743 降至 449 步 2.3269，未观察到验证反弹；NS 最大奇异值通过 8 次 power iteration 估计，有限步 NS 不保证精确正交化。

## 不确定性与解释边界

- 每个 validation loss 为固定 20 个 batch 上按 token 数加权的 cross-entropy；报告标准误用于阶段内近似并列判断。短程训练下 weight decay 的差异仍可能很弱。
- 本次结果是该模型、固定数据抽样和当前随机种子的短程比较；差异不能直接外推到其他种子或完整预训练。
- 理论说明：RMS-to-RMS 倍率按矩阵逻辑形状 `sqrt(m/n)`；loss 排名、吞吐和稳定性均来自本次记录。未实际训练的长期效果属于推测。

## 下一轮建议

- 使用推荐配置从共同初始化做至少 2–3 个独立种子的 longer-run 对照，并继续固定评估集。
- 若 250/450 步验证仍呈单调改善，围绕推荐 LR 做更窄的局部搜索；将 Muon weight decay 结论延长到更长训练后再判断。

## 产物

- Manifest：`runs/muon_hpo_20261029_114312/manifest.json`
- Metrics：`runs/muon_hpo_20261029_114312/metrics.csv`
- Matrix diagnostics：`runs/muon_hpo_20261029_114312/diagnostics.jsonl`
- Plots：`plots/*.svg`
- Recommended config: `configs/muon_rms_optimal.yaml`
- Resume: `python scripts/tune_muon.py --max-total-steps 1000 --resume runs/muon_hpo_20261029_114312`
