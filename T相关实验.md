# 方案：SNR 弱特征驱动的 Timestep PolicyNet 实验方案

## 1. 实验定位

本方案用于替代原先的手工 `SNR -> T*` 动态时间步模块，但不完全丢弃原有 SNR 计算。核心思想是：

```text
T_snr 不再作为最终时间步
T_snr 仅作为 decoder-side 多源特征之一
最终时间步 T_pred 由轻量 TimestepPolicyNet 学习得到
```

原方案假设：

```text
T_best ~= f(SNR_compress)
```

已有实验观察表明：

- 真实最优时间步 `T_best / T_oracle` 与 `T_snr` 的相关性不高；
- `T_best / T_oracle` 与码率、bpp 或 λ 的相关性更低；
- 最优单步去噪时间步并不是简单码率或简单 SNR 的单变量函数；
- 真实时间步更可能由压缩不确定性、图像内容复杂度、latent 状态、Aux residual 和 UNet 修复需求共同决定。

因此，本方案将原来的手工映射问题改为：

> 从解码端可观测的多源状态中，学习每张图像的单步扩散去噪时间步。

## 2. 与 OSCAR 和原 SNR 方法的机制差异

### 2.1 OSCAR 的时间步机制

OSCAR 的核心机制可以概括为：

```text
bit-rate r -> cosine similarity calibration -> pseudo diffusion timestep
```

其关键假设是：同一 bit-rate 下的压缩 latent 可以对应较稳定的 pseudo diffusion timestep。

### 2.2 原 SNR 方法的问题

原 `DynamicTimestepModule` 的机制是：

```text
entropy scales -> compression SNR -> DDPM SNR schedule -> T_snr
```

它虽然没有直接使用 bit-rate，但仍然属于：

```text
compression quality proxy -> diffusion timestep
```

这与 OSCAR 的高层机制仍然接近，并且实验上已经发现 `T_snr` 对真实最优时间步的解释能力不足。

### 2.3 本方案的机制定位

本方案不再主张“提出压缩质量到扩散时间步的映射”，而是主张：

```text
oracle timestep analysis
-> 证明 bitrate/SNR 单变量映射不足
-> 将 T_snr 降级为弱物理启发特征
-> 使用 decoder-side 多因素特征学习 timestep policy
```

最终可主张的贡献边界是：

> 本方案发现压缩扩散解码中的最优单步去噪时间步并非码率或简单 SNR 的单调函数，并提出一种以 SNR-derived timestep 为弱特征、结合解码端多源状态的学习式 timestep policy，用于逐图预测单步扩散解码时间步。

## 3. 总体流程

整体实验改为六个阶段。关键变化是：**先尝试不依赖完整 Oracle-T 标签的端到端 PolicyNet 训练，再尝试 Oracle-T 约束的单独时间步预测网络**。同时，不直接在固定 `T=999` 训练得到的模型上生成 PolicyNet 监督标签，因为该模型的参数已经适配 `999`，posthoc 扫描很可能把最优时间步吸到 `999`。

```text
Stage 0: 固定 999 模型 posthoc sweep
         目的: 诊断事后换 T 是否有空间，不作为 PolicyNet 标签

Stage 1: T-augmentation 训练
         目的: 让模型具备多个时间步的工作能力，降低 999 训练偏置

Stage 2: 在 T-augmentation 模型上重新做 Oracle-T sweep
         目的: 获得更可信的 per-image / per-rate T_oracle_aug

Stage 3: 训练方式一：端到端 TimestepPolicyNet
         目的: 不依赖完整 Oracle-T 标签，直接通过重建损失学习时间步策略

Stage 4: 训练方式二：Oracle-T 约束的 TimestepPolicyNet
         目的: 用 Stage 2 的 T_oracle_aug 单独训练时间步预测网络

Stage 5: 两种 PolicyNet 路线的对照与微调
         目的: 比较端到端训练与 Oracle-T 约束训练的收益、稳定性和解释性
```

其中：

- Stage 0 的 `T_oracle_posthoc` 只用于诊断，不作为监督标签；
- Stage 1 通过随机或混合时间步训练，缓解模型只适配 `T=999` 的问题；
- Stage 2 的 `T_oracle_aug` 作为主要 oracle 分析对象，并服务于第二种训练方式；
- Stage 3 先尝试端到端训练，训练信号来自最终重建损失，不要求全训练集都有 Oracle-T；
- Stage 4 再尝试 Oracle-T 约束训练，用 `T_oracle_aug` 单独训练轻量预测器；
- Stage 5 将两种预测器接入原模型，验证实际重建收益。

## 4. 固定基础模型

为保证实验聚焦，以下部分保持不变：

- 256 通道 VAE latent 结构；
- LatentCodec 主干；
- 4-pass checkerboard entropy model；
- GaussianConditional 熵模型；
- λ-FiLM 连续码率控制；
- SD-Turbo / UNet 单步 latent 去噪；
- AuxDecoder 残差分支；
- 原主要重建损失、码率损失和感知损失设置。

被替换的模块是原 `DynamicTimestepModule` 的最终决策逻辑。

原流程：

```text
scales_all -> T_snr -> UNet timestep
```

新流程：

```text
scales_all -> T_snr
decoder-side features + T_snr -> TimestepPolicyNet -> T_pred
T_pred -> UNet timestep
```

## 5. Stage 0-2：可靠 Oracle-T 的获得流程

### 5.1 为什么不能直接用固定 999 模型找 Oracle-T

如果基础模型训练时始终固定 `T=999`，模型参数、UNet LoRA、AuxDecoder 和 latent codec 都会共同适配这个时间步。在这种情况下，直接做 posthoc sweep 得到的最优时间步：

```text
T_oracle_posthoc = argbest_T Quality(model_trained_with_999, T)
```

很可能只是说明：

```text
已经适配 999 的模型，在推理时继续使用 999 最稳。
```

这不能证明任务本身的最优时间步就是 `999`，也不能证明时间步自适应没有价值。因此，本方案将 oracle 搜索拆成两类：

```text
T_oracle_posthoc: 固定 999 模型上的事后扫描结果，只用于诊断
T_oracle_aug:     T-augmentation 模型上的扫描结果，用于特征分析和 PolicyNet 训练
```

### 5.2 Stage 0：固定 999 模型 posthoc sweep

Stage 0 的目标是回答：

1. 在当前固定 `999` 训练范式下，推理时临时改变 T 是否有收益；
2. 如果最优 T 大量集中在 `999`，这种集中是否来自训练偏置；
3. `SNR-T`、`Fixed-999` 和 `Fixed-best-global` 的初步差距有多大。

Stage 0 的结论只用于诊断：

```text
不使用 T_oracle_posthoc 训练 PolicyNet
不把 T_oracle_posthoc 当作真实任务规律
不据此否定多时间步策略
```

如果 Stage 0 显示 `T_oracle_posthoc` 几乎全是 `999`，合理解释是：

```text
模型已经过度适配 T=999，需要先做 T-augmentation 训练再重新寻找 oracle。
```

### 5.3 Stage 1：T-augmentation 训练

Stage 1 的目标是让模型在训练阶段见过多个时间步，使后续 oracle sweep 不再被固定 `999` 严重污染。

推荐先使用保守混合采样：

```text
50% batch: T = 999
50% batch: T ~ Uniform(800, 999)
```

也可以使用课程式采样：

```text
Phase 1: 80% T=999, 20% T~Uniform(900,999)
Phase 2: 50% T=999, 50% T~Uniform(850,999)
Phase 3: 50% T=999, 50% T~Uniform(800,999)
```

训练时不引入 PolicyNet，只使用随机或混合时间步：

```text
sample T_train
UNet(sample, T_train)
_batched_ddpm_step(pred, T_train, sample)
reconstruction loss + rate loss
```

建议优先微调：

- UNet LoRA；
- λ-FiLM；
- AuxDecoder；
- 必要时微调 LatentCodec 的少量后端模块。

VAE 主体和熵编码主干尽量保持稳定，避免把实验变量扩大。

Stage 1 的验证需要同时报告：

```text
Fixed-999
Fixed-best-global
Random-T validation
SNR-T validation
```

如果 T-augmentation 明显损伤 `Fixed-999` 性能，需要缩小随机 T 范围或提高 `T=999` 的采样比例。

### 5.4 Stage 2：在 T-augmentation 模型上重新做 Oracle-T sweep

Stage 2 才是正式 oracle 生成阶段。此时模型已经具备一定多时间步适应能力，扫描得到的：

```text
T_oracle_aug = argbest_T Quality(model_trained_with_T_aug, T)
```

更适合作为 per-image / per-rate 最优时间步的经验标签。

Stage 2 回答三个问题：

1. `T_oracle_aug` 与 `T_snr`、bpp、λ 的相关性到底有多低；
2. 哪些 decoder-side 变量更能解释 `T_oracle_aug`；
3. `T_oracle_aug` 相比 fixed timestep 和 SNR timestep 的上界收益有多大。

### 5.5 扫描对象

对验证集或训练子集中的样本，在多个 λ 或实际码率点下扫描时间步。

建议数据规模：

```text
图像数量: 500-2000 张
每张图 λ 数量: 3-6 个
候选 T 数量: 8-16 个
```

候选时间步可先取：

```text
T_candidates = [800, 825, 850, 875, 900, 925, 950, 975, 999]
```

如果后续发现最优值集中在某个区间，可加密：

```text
[850, 875, 900, 925, 940, 950, 960, 975, 990, 999]
```

注意：Stage 0 和 Stage 2 可以使用同一套候选时间步，便于比较固定 999 模型和 T-augmentation 模型的 oracle 分布差异。

### 5.6 Oracle 目标

可以定义多个 oracle：

```text
T_oracle_aug_psnr   = argmax_T PSNR(x_hat_T, x)
T_oracle_aug_msssim = argmax_T MS-SSIM(x_hat_T, x)
T_oracle_aug_lpips  = argmin_T LPIPS(x_hat_T, x)
T_oracle_aug_dists  = argmin_T DISTS(x_hat_T, x)
```

最终论文偏感知指标，主 oracle 使用 `LPIPS`
```

主实验中建议先选择一个主 oracle，避免目标含混。

### 5.7 需要记录的变量

每个样本、每个 λ、每个候选 T 记录：

```text
image_id
lambda
actual_bpp
T_candidate
T_snr
PSNR / MS-SSIM / LPIPS / DISTS
checkpoint_type  # fixed999 或 T_aug
```

同时记录 decoder-side 特征：

```text
scales_mean
scales_std
scales_p10 / p50 / p90
scales_spatial_std
likelihood_mean 或 nll_mean
bits_per_symbol_mean
y_hat_energy
y_hat_std
g_s_output_energy
res1_norm
sample_norm
```

如果计算成本允许，可记录 UNet 反应特征：

```text
fixed_T 下 model_pred_norm
fixed_T 下 x0_pred - sample 的 residual_norm
fixed_T 下 residual / sample 的相对能量
```

注意：这些特征必须在推理时可获得，不能依赖原图或未压缩 latent。

### 5.8 相关性分析

至少分析：

```text
corr(T_oracle_aug, T_snr)
corr(T_oracle_aug, log(lambda))
corr(T_oracle_aug, actual_bpp)
corr(T_oracle_aug, scales_mean)
corr(T_oracle_aug, scales_std)
corr(T_oracle_aug, nll_mean)
corr(T_oracle_aug, y_hat_energy)
corr(T_oracle_aug, res1_norm)
```

同时做多因素解释能力分析：

```text
Linear Regression R^2
Random Forest / XGBoost feature importance
Mutual information
per-lambda-bin correlation
per-bpp-bin correlation
```

同时必须比较：

```text
distribution(T_oracle_posthoc)
distribution(T_oracle_aug)
```

如果 `T_oracle_posthoc` 集中在 `999`，而 `T_oracle_aug` 出现更分散的最优时间步，说明固定 `999` 训练确实会污染 oracle 搜索。  
如果二者都集中在 `999`，则说明当前模型和任务设置下时间步自适应空间可能有限，应考虑转向 fixed T + uncertainty gate。

Stage 2 的预期结论不是找到一个单一强变量，而是证明：

```text
T_oracle_aug 不能由 bitrate 或 SNR 单独解释
多因素 decoder-side 特征比单变量 T_snr 更有解释力
```

## 6. decoder-side 特征设计

### 6.1 特征原则

输入特征必须满足：

- 解码端可获得；
- 不依赖原图；
- 不增加码流开销；
- 尽量来自已有中间量；
- 计算成本远低于 UNet。

### 6.2 推荐特征集合

基础特征：

```text
log_lambda
actual_bpp
T_snr
snr_compress
```

熵模型特征：

```text
scales_mean
scales_std
scales_min
scales_max
scales_p10
scales_p50
scales_p90
scales_spatial_std
nll_mean
nll_std
bits_per_symbol_mean
```

latent 内容特征：

```text
y_hat_mean_abs
y_hat_energy
y_hat_std
y_hat_channel_std_mean
y_hat_spatial_gradient_mean
```

解码器状态特征：

```text
sample_norm
sample_std
g_s_output_norm
res1_norm
res1_to_sample_ratio
```

可选 UNet 反应特征：

```text
model_pred_norm_at_fixed_T
x0_delta_norm_at_fixed_T
x0_delta_to_sample_ratio
```

其中 `T_snr` 不作为最终答案，只作为一个弱特征：

```text
features = concat(stats, T_snr)
T_pred = PolicyNet(features)
```

### 6.3 特征归一化

所有标量特征使用训练集统计量归一化：

```text
feature_norm = (feature - mean) / std
```

对于重尾变量建议使用：

```text
log(1 + x)
```

例如：

```text
log_bpp
log_scales_mean
log_nll
log_res1_norm
```

## 7. PolicyNet 的两种训练方式

本方案保留两条训练路线，实验顺序为：

```text
先尝试 Route 1: 端到端训练
再尝试 Route 2: Oracle-T 约束训练
```

原因是：端到端训练不需要为整个训练集离线扫描 `T_oracle_aug`，实现和数据准备更轻；Oracle-T 约束训练成本更高，但可解释性更强，也可以作为上界分析和后续增强路线。

### 7.1 网络输入输出

输入：

```text
feature vector [B, F]
```

输出：

```text
T_pred_float [B]
```

将输出限制到 `[t_min, t_max]`：

```python
u = MLP(features)              # [B, 1]
T_pred = t_min + (t_max - t_min) * sigmoid(u)
```

默认：

```text
t_min = 800
t_max = 999
```

### 7.2 推荐网络结构

最小版本：

```text
Linear(F, 128)
SiLU
LayerNorm
Linear(128, 64)
SiLU
Linear(64, 1)
Sigmoid
```

参数量很小，不改变主模型复杂度。

如果后续使用空间特征，可扩展为：

```text
scales map -> small CNN pooling
pooled vector + scalar stats -> MLP -> T_pred
```

第一阶段建议只用标量统计，便于解释和快速验证。

### 7.3 训练方式一：端到端训练

端到端训练不使用 `T_oracle_aug` 作为监督标签，而是让 PolicyNet 直接通过最终重建损失学习时间步：

```text
features + T_snr -> PolicyNet -> T_pred
T_pred -> UNet one-step denoising
x_hat -> reconstruction loss
```

主损失沿用原压缩重建目标：

```text
L = distortion_loss + rate_loss + perceptual_loss(optional) + regularization
```

其中 `rate_loss` 不直接依赖 `T_pred`，但保持完整训练目标，避免时间步策略只优化图像质量而破坏整体 RD 评估。

端到端训练的关键难点是：如果直接执行

```python
timestep = T_pred.round().long()
```

则梯度无法从重建损失传回 `T_pred`。因此需要选择一种可训练近似。

推荐优先尝试连续时间步近似：

```text
T_pred_float -> UNet timestep embedding
T_pred_float -> alphas_cumprod 连续插值
```

即：

- UNet 的 time embedding 使用 float timestep；
- `_batched_ddpm_step` 中的 `alpha_prod_t` 不再用整数索引，而是对 `alphas_cumprod` 做线性插值；
- 推理时再将 `T_pred` round 到最近整数，或继续使用连续插值版本。

如果连续时间步实现成本较高，可使用 straight-through estimator：

```text
forward:  T_hard = round(T_pred)
backward: pretend T_hard ~= T_pred
```

更稳定但更贵的版本是候选时间步 soft selection：

```text
PolicyNet -> logits over T_candidates
对多个候选 T 计算重建
用 softmax 权重加权输出或加权损失
```

该版本计算量随候选数量增加，只建议用于小规模验证，不作为主训练方案。

端到端训练建议分两步：

```text
Step 1: freeze VAE / LatentCodec / UNet，仅训练 PolicyNet
Step 2: unfreeze UNet LoRA、λ-FiLM 或 AuxDecoder 的少量参数做联合微调
```

端到端路线的优点：

- 不需要完整训练集的 Oracle-T；
- 可以直接优化最终重建质量；
- 如果 `T_oracle_aug` 标签噪声较大，端到端训练更灵活。

主要风险：

- 时间步离散化导致训练不稳定；
- PolicyNet 可能收敛到固定时间步；
- 如果主模型尚未通过 T-augmentation 适配多 T，端到端训练容易退回 `999`。

### 7.4 训练方式二：Oracle-T 约束训练

Oracle-T 约束训练使用 Stage 2 中由 T-augmentation 模型扫描得到的 `T_oracle_aug` 作为监督标签，单独训练时间步预测网络：

```text
features + T_snr -> PolicyNet -> T_pred
T_pred ~= T_oracle_aug
```

回归形式：

```text
L_T = SmoothL1(T_pred, T_oracle_aug)
```

如果 `T_oracle_aug` 来自离散候选集合，也可以使用分类形式：

```text
PolicyNet -> logits over T_candidates
L_cls = CrossEntropy(logits, oracle_index)
T_pred = weighted_sum(softmax(logits) * T_candidates)
```

分类版本的优点是训练稳定，适合候选时间步较少的情况。  
回归版本的优点是推理更连续，适合后续端到端微调。

注意：Oracle-T 约束训练只使用 `T_oracle_aug`，不使用 `T_oracle_posthoc`。Stage 0 的 `T_oracle_posthoc` 只用于诊断固定 `999` 训练偏置，不进入监督训练。

Oracle-T 约束路线的优点：

- 可解释性强；
- 便于评估 `T_pred` 与 `T_oracle_aug` 的误差；
- 可以清楚展示多源特征是否比 `T_snr` 更能预测最优时间步。

主要风险：

- 需要离线扫描 Oracle-T，成本较高；
- `T_oracle_aug` 受评价指标影响；
- 如果质量曲线很平，硬标签可能有噪声，需要 soft label 或 near-optimal set。

### 7.5 SNR 弱特征消融

必须设置以下消融：

```text
PolicyNet without T_snr
PolicyNet with T_snr
Only T_snr
Only bpp/lambda
```

预期结论：

```text
T_snr 单独效果有限
加入 T_snr 到多源特征后可能有小幅收益
最终性能主要来自多因素 decoder-side 特征
```

这样可以避免把贡献重新落回 `SNR -> T`。

## 8. Stage 5：接入主模型微调与路线对照

### 8.1 接入方式

原：

```python
T_star = dynamic_timestep(scales_all)
```

改为：

```python
T_snr = snr_timestep(scales_all)
features = build_decoder_features(...)
T_pred = timestep_policy(features, T_snr)
```

然后：

```python
x0_pred = _batched_ddpm_step(model_pred, T_pred.long(), sample)
```

### 8.2 两种路线的接入方式

端到端路线：

```text
PolicyNet 从训练一开始就在主模型 forward 内部产生 T_pred
训练目标来自最终重建损失
不需要 T_oracle_aug 标签
```

Oracle-T 约束路线：

```text
先单独训练 PolicyNet 拟合 T_oracle_aug
再将训练好的 PolicyNet 接入主模型
可选择只推理验证，也可进一步端到端微调
```

建议实验顺序：

```text
1. Route 1: End-to-end PolicyNet
2. Route 2: Oracle-supervised PolicyNet
3. Route 2 + small end-to-end finetune
```

这样可以先验证最便捷路线是否有效，再判断 Oracle-T 约束是否带来额外收益。

### 8.3 离散时间步处理

如果直接 `long()`，梯度不能通过 timestep 传回 PolicyNet。  
因此端到端路线有三种选择：

方案 A：连续 timestep 近似。

```text
UNet 使用 float timestep embedding
DDPM step 使用 alphas_cumprod 线性插值
```

```text
优点: 端到端、计算量不变
缺点: 需要确认 UNet time embedding 和 scheduler 插值实现稳定
```

方案 B：straight-through estimator。

```text
forward: T_hard = round(T_pred)
backward: 使用 T_pred 的近似梯度
```

```text
优点: 接近最终整数 timestep 推理
缺点: 梯度近似较粗，需要观察稳定性
```

方案 C：候选时间步 soft selection。

```text
对 K 个候选 T 分别跑 UNet 或 DDPM step
softmax 权重加权输出
```

```text
优点: 可端到端
缺点: 计算量变为 K 倍，不适合作为最终推理
```

建议路线：

```text
优先尝试方案 A
如果实现不稳定，再尝试方案 B
方案 C 仅用于小规模验证上界，不作为主方案
```

对于 Oracle-T 约束路线，如果只做监督预训练和推理评估，可以直接：

```text
timestep = round(T_pred)
```

此时不需要时间步对 PolicyNet 可导。只有当继续端到端微调时，才需要上面的连续近似或梯度近似。

### 8.4 微调策略

端到端路线第一步：冻结主模型，只训练 PolicyNet。

```text
freeze VAE / UNet / LatentCodec / AuxDecoder
train TimestepPolicyNet
```

端到端路线第二步：可选联合微调。

```text
unfreeze λ-FiLM 和少量 LoRA 参数
low learning rate
PolicyNet learning rate 稍高
```

建议学习率：

```text
PolicyNet lr = 1e-4
LoRA / FiLM lr = 1e-5
```

Oracle-T 约束路线：

```text
先离线训练 PolicyNet
再接入主模型评估
最后可选 low-lr end-to-end finetune
```

## 9. 对照实验

至少比较以下策略：

| 策略 | 描述 | 目的 |
|---|---|---|
| Fixed-999 | 固定 SD-Turbo 原始时间步 | 基础下界 |
| Fixed-best-global | 验证集最优固定时间步 | 排除固定 T 选择不当 |
| SNR-T | 原 `SNR -> T*` | 原方案基线 |
| Bitrate-T | 用 bpp 或 λ 拟合时间步 | 对比 OSCAR 式单变量思想 |
| PolicyNet-E2E no T_snr | 端到端训练，不输入 `T_snr` | 验证端到端多源特征本身 |
| PolicyNet-E2E with T_snr | 端到端训练，输入 `T_snr` | 第一优先路线，验证 SNR 弱特征是否有增益 |
| PolicyNet-Oracle no T_snr | Oracle-T 约束训练，不输入 `T_snr` | 验证监督路线中的多源特征 |
| PolicyNet-Oracle with T_snr | Oracle-T 约束训练，输入 `T_snr` | 第二路线，验证 Oracle 约束是否优于端到端 |
| Oracle-T-posthoc | 固定 999 模型上的离线扫描 | 诊断训练偏置，不作为方法上界 |
| Oracle-T-aug | T-augmentation 模型上的离线扫描 | 主要 oracle 上界 |

如果计算资源允许，增加：

| 策略 | 描述 | 目的 |
|---|---|---|
| PolicyNet-E2E only T_snr | 端到端，只输入 `T_snr` | 验证 SNR 特征单独不足 |
| PolicyNet-Oracle only T_snr | Oracle 约束，只输入 `T_snr` | 验证监督条件下 SNR 单特征上限 |
| PolicyNet only entropy stats | 只输入 scales / NLL | 验证熵模型特征贡献 |
| PolicyNet entropy + latent | 熵模型 + latent 统计 | 验证内容复杂度贡献 |
| PolicyNet all features | 全部特征 | 最终版本 |

## 10. 评价指标

### 10.1 图像质量指标

```text
PSNR
MS-SSIM
LPIPS
DISTS
FID 或 KID
CLIPIQA / MUSIQ 可选
```

### 10.2 码率指标

```text
bpp
RD curve
BD-rate
```

### 10.3 时间步预测指标

端到端路线即使不使用 Oracle-T 训练，也可以在 Stage 2 的 oracle 分析集上评估：

```text
MAE(T_pred, T_oracle_aug)
Top-1 accuracy over T_candidates
Top-2 accuracy over T_candidates
corr(T_pred, T_oracle_aug)
corr(T_pred, T_snr)
```

注意：时间步预测误差不是端到端路线的训练目标，只作为解释性指标。最终以重建质量和 RD/感知表现为准。

### 10.4 机制解释指标

```text
feature importance
per-lambda-bin mean(T_pred)
per-bpp-bin mean(T_pred)
per-content-complexity-bin mean(T_pred)
case study: T_snr 错但 PolicyNet 对的样本
```

## 11. 预期实验结论

理想结果应满足：

1. 固定 999 模型上的 `T_oracle_posthoc` 可能集中在 `999`，说明存在训练偏置；
2. T-augmentation 后的 `T_oracle_aug` 分布更能反映多时间步适配模型的真实偏好；
3. `T_oracle_aug` 与 `T_snr`、bpp、λ 的相关性较低；
4. 多源 decoder-side 特征能更好预测 `T_oracle_aug`；
5. `PolicyNet-E2E with T_snr` 优于 `SNR-T`，说明不依赖完整 Oracle-T 的端到端路线可行；
6. `PolicyNet-E2E with T_snr` 至少不弱于 `PolicyNet-E2E no T_snr`，说明 SNR 作为弱特征仍有价值；
7. `PolicyNet-Oracle with T_snr` 在时间步预测误差上优于端到端路线，说明 Oracle 约束能提供更明确的时间步监督；
8. 如果 `PolicyNet-Oracle` 的重建质量优于 `PolicyNet-E2E`，说明离线 Oracle-T 约束值得保留；
9. 如果 `PolicyNet-E2E` 已经接近或优于 `PolicyNet-Oracle`，说明可优先采用更简单的端到端训练；
10. 两种路线都应在不同 λ 区间和不同内容复杂度图像上报告稳定性。

如果出现：

```text
PolicyNet-E2E with T_snr ≈ PolicyNet-E2E no T_snr
```

也可以接受。此时结论应改为：

> SNR-derived timestep 对最终策略贡献有限，最优时间步主要由其他 decoder-side 状态决定。

这仍然能支持“码率/SNR 单变量映射不足”的核心判断。

## 12. 风险与备选方案

### 12.1 风险：固定 999 训练污染 oracle

如果模型一直固定 `T=999` 训练，posthoc sweep 得到的最优时间步很可能集中在 `999`。

处理方式：

- 不使用 `T_oracle_posthoc` 训练 PolicyNet；
- 先做 T-augmentation 训练，再重新生成 `T_oracle_aug`；
- 同时报告 `T_oracle_posthoc` 和 `T_oracle_aug` 的分布差异；
- 如果 T-augmentation 后 oracle 仍集中在 `999`，说明时间步自适应空间有限，应考虑转向 fixed T + uncertainty gate。

### 12.2 风险：T_oracle_aug 受指标影响

不同指标可能对应不同最优时间步。

处理方式：

- 主文只选择一个主优化目标；
- 附录报告不同 oracle 指标下的差异；
- 如果差异很大，可以训练不同质量偏好的 PolicyNet。

### 12.3 风险：端到端训练中的时间步不可导

端到端路线如果直接将 `T_pred` round / long 成整数，重建损失无法有效更新 PolicyNet。

处理方式：

- 优先使用连续 timestep embedding 和 `alphas_cumprod` 插值；
- 或使用 straight-through estimator；
- 小规模使用候选时间步 soft selection 验证上界；
- 监控 `T_pred` 是否坍缩到固定值，尤其是 `999`。

### 12.4 风险：PolicyNet 学到数据集偏置

处理方式：

- 使用跨数据集验证；
- 在 Kodak / DIV2K-val / CLIC 上分别报告；
- 做 content complexity bin 分析。

### 12.5 风险：T_pred 不稳定

处理方式：

- 输出限制在 `[800, 999]`；
- 对 `T_pred` 加平滑正则；
- 使用分类候选而不是连续回归；
- 对 batch 内异常预测做 clamp。

### 12.6 风险：收益低于固定最优 T

如果 `Fixed-best-global` 已经很强，说明时间步自适应空间有限。  
此时应把方法改为：

```text
固定 T + uncertainty gate
```

也就是转向局部修复分配路线。

## 13. 最小实现改动

### 13.1 新增模块

```python
class TimestepPolicyNet(nn.Module):
    def __init__(self, in_dim, hidden=128, t_min=800, t_max=999):
        super().__init__()
        self.t_min = t_min
        self.t_max = t_max
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, features):
        u = self.net(features).squeeze(-1)
        return self.t_min + (self.t_max - self.t_min) * torch.sigmoid(u)
```

### 13.2 保留原 SNR 计算

原 `DynamicTimestepModule` 可拆为：

```text
SNRTimestepFeature
```

只输出：

```text
T_snr
snr_compress
```

不再直接决定最终 timestep。

### 13.3 新增特征构建函数

```python
features = build_timestep_features(
    lmbda=lmbda,
    bpp=bpp,
    scales_all=scales_all,
    y_hat=y_hat,
    sample=sample,
    res1=res1,
    T_snr=T_snr,
    snr_compress=snr_compress,
)
```

### 13.4 替换时间步入口

```python
T_snr, snr_compress = snr_timestep(scales_all)
features = build_timestep_features(...)
T_pred = timestep_policy(features)
timestep = T_pred.round().long()
```

## 14. 日志与诊断

训练和验证阶段记录：

```text
lambda
bpp
policy_route  # E2E 或 Oracle-supervised
T_snr
T_pred
T_oracle_posthoc  # 仅 Stage 0 诊断集
T_oracle_aug      # 仅 Stage 2 oracle 分析集和监督训练集
mean feature values
PSNR / MS-SSIM / LPIPS / DISTS
```

诊断图：

```text
T_oracle_posthoc distribution
T_oracle_aug distribution
T_oracle_aug vs T_snr scatter
T_oracle_aug vs T_pred scatter
PolicyNet-E2E vs PolicyNet-Oracle quality comparison
T_pred distribution per lambda bin
T_pred distribution per bpp bin
PolicyNet gain over SNR-T per image
failure cases
```

## 15. 论文表述建议

这一路线不应写成：

```text
提出一种 SNR 到扩散时间步的动态映射。
```

应写成：

```text
我们首先发现，单步压缩扩散解码中的最优去噪时间步与码率或简单 SNR 并不呈稳定单调关系。
因此，我们将 SNR-derived timestep 仅作为弱物理启发特征，并结合熵模型不确定性、latent 内容统计和解码器状态，学习一个轻量级 decoder-side timestep policy。
```

贡献定位：

```text
从手工 compression-quality-to-timestep 映射
转向 decoder-side multi-factor timestep policy learning
```

这样能保留原 SNR 模块的工程价值，同时把方法机制与 OSCAR 的 bit-rate-to-timestep 标定拉开。
