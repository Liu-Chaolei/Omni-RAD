# StableCodec 可变码率 + 动态时间步方案设计报告（最终版）

> 基于 I2C（TPAMI 2024）与 MRIC（CVPR 2023）的 λ-FiLM 四注入点架构，
> 附加基于 SNR 的动态时间步 T\*

---

## 1. 背景与目标

原始 StableCodec 采用固定 λ 的两阶段训练策略（Stage I λ=0.5，Stage II 在离散 λ 集合上微调），每个码率点需要独立训练一个模型，部署成本高。同时，原始方案使用固定时间步 T=999 进行去噪，未能根据压缩质量自适应地调整去噪强度。

本方案的目标是：

- 单一模型覆盖连续码率范围（约 0.005–0.05 bpp）
- 解码端通过改变 λ 实现实时码率切换
- λ 以 2 字节（float16）存入码流，提供 65536 个有效码率点
- 对原始 StableCodec 架构的侵入最小，保持兼容性
- **根据压缩质量自适应调整去噪时间步 T\*：高压缩（高 λ）→ 量化更粗 → 更大 T\*（更强去噪）；低压缩（低 λ）→ 量化更精细 → 更小 T\*（保守还原）**

---

## 2. 参考方法分析

### 2.1 I2C（Invertible Continuous Codec，TPAMI 2024）

**控制目标**：码率（λ）

**关键机制**：
- IAT（Invertible Activation Transformation）模块：`e = (s ⊙ β) ⊕ γ`，逆变换 `ŝ = (ê ⊖ γ) ⊘ β`
- γ, β ∈ R^(C×H×W) 由量化级别 L 经卷积生成，空间自适应（逐像素）
- 张量化拉格朗日乘子：Λ = 0.0012 × e^(4.382·L)，Λ ∈ R^(C×H×W)，逐像素调制失真权重
- 训练损失：`L = R + Λ⊙D`，每次迭代逐像素随机采样，一次训练覆盖完整 RD 曲线

**本方案借鉴**：
- 张量化 Λ 训练策略（per-image 对数均匀采样）
- λ 以 float16 存入码流的比特流设计

### 2.2 MRIC（Multi-Realism Image Compression，CVPR 2023）

**控制目标**：真实感-失真权衡（β），解码端可调

**关键机制 FourierCond**：
- β → Fourier 特征（L=10 个频率对）→ 2 层 MLP（512 维）→ f(β) ∈ R^512
- 每个卷积层独立投影：`h' = h + W_i·f(β)`（纯加法，无缩放）
- 仅条件化解码器 G，不影响编码器

**本方案借鉴**：
- FourierCond 嵌入结构（完整移植至 PyTorch）
- 每注入点独立投影权重的设计

---

## 3. 核心设计：λ-FiLM 四注入点架构

### 3.1 设计选型对比

| 维度 | I2C（IAT） | MRIC（FourierCond） | 本方案（λ-FiLM） |
|---|---|---|---|
| 调制形式 | `h' = (h ⊙ β) ⊕ γ`，逐像素 | `h' = h + W_i·f(β)`，纯加法 | `h' = h ⊙ γ_i + β_i`，仿射 |
| 是否需要可逆性 | 是（严格数学约束） | 否 | 否 |
| 控制粒度 | 逐像素张量 Λ | 全局标量 β | 全局标量 λ（训练时逐图）|
| 参数开销 | 较大（逐层空间图） | 较小（共享嵌入+独立投影）| 较小（同 MRIC）|

FiLM 选型理由：相比 MRIC 纯加法多了 γ 缩放（表达力更强），同时不需要 I2C 的可逆性约束，天然适配 StableCodec 非可逆变换的架构。

### 3.2 λ 嵌入模块（LambdaFiLMEmbed）

```
λ (scalar/[B])
  │
  ▼
log-normalise → l = clip((log λ - log λ_min) / (log λ_max - log λ_min), 0, 1)
  │
  ▼
Fourier features → [l, sin(2^0·πl), cos(2^0·πl), ..., sin(2^9·πl), cos(2^9·πl)]
                   维度：1 + 2×10 = 21
  │
  ▼
2-layer MLP (Linear→ReLU→Linear→ReLU)  隐层/输出均为 512 维
  │
  ▼
f(λ) ∈ R^{B×512}     （所有注入点共享此嵌入）
```

超参数：`LAMBDA_MIN=0.1`（保守下界），`LAMBDA_MAX=128.0`（保守上界），`NUM_FREQS=10`，`FILM_DIM=512`

**注意**：`LAMBDA_MIN/LAMBDA_MAX` 为 FiLM 嵌入的 log-normalise 边界，应设为保守宽边界，避免 clamp 饱和。实际训练采样范围由 config 的 `lambda_min`/`lambda_max`/`lambda_sample_max` 控制。这些边界作为 buffer 注册在 state_dict 中，checkpoint 加载时自动恢复。

### 3.3 FiLM 调制层（FiLMLayer）

每个注入位置各自独立一对 `(gamma_proj, beta_proj)` 线性层：

```python
γ_i = Linear_γ_i(f(λ))          # [B, C]
β_i = Linear_β_i(f(λ))          # [B, C]
h'  = h ⊙ γ_i.view(B,C,1,1) + β_i.view(B,C,1,1)
```

**初始化策略**（确保训练起点与原始模型完全一致）：
- `gamma_proj`：weight=0，bias=1 → γ 初始恒为 1
- `beta_proj`：weight=0，bias=0 → β 初始恒为 0

### 3.4 四个注入点详解

#### 注入点 1 — AnalysisTransform（ga）

**位置**：ga 内每个 BasicBlock 之后

**信道维度变化**：

```
concat(pre1(ESD_latent), pre2(EAux_latent))  →  [B, 192, H/2, W/2]
BasicBlock(192)  →  FiLM(192, f(λ))                          ← 注入
Downsample(192→256)
BasicBlock(256)  →  FiLM(256, f(λ))                          ← 注入
Downsample(256→320)
BasicBlock(320)  →  FiLM(320, f(λ))                          ← 注入
```

**设计说明**：EAux（冻结）和 ESD 的输出在进入 ga 前以原始 concat 方式融合，λ 仅通过 FiLM 注入 ga 的特征变换层，不干预两路特征的融合权重。

#### 注入点 2 — SynthesisTransform（gs）

**位置**：gs 内每个 BasicBlock 之后

```
BasicBlock(320)  →  FiLM(320, f(λ))  →  Upsample(320→320)   ← 注入
BasicBlock(320)  →  FiLM(320, f(λ))  →  Upsample(320→320)   ← 注入
BasicBlock(320)  →  FiLM(320, f(λ))  →  Upsample(320→320)   ← 注入
```

**作用**：gs 产生送入 ϵSD 的噪声隐变量 lT。高 λ（高压缩）→ gs 生成更粗糙的 lT，让 UNet 去噪更激进；低 λ（低压缩）→ lT 更精细，UNet 保守还原。

#### 注入点 3 — ϵSD UNet LoRA 自适应缩放

**位置**：UNet 推理后、调度器步骤前

**实现**（不侵入 PEFT LoRA 内部）：

```python
# unet_lora_proj: Linear(512→64→1)，全零初始化
film_embed  = self.codec.film_embed(lmbda)                  # [B, 512]
delta_s     = self.unet_lora_proj(film_embed)               # [B, 1]
lora_scale  = (1.0 + torch.tanh(delta_s)).view(B, 1, 1, 1) # ∈ (0,2)，[B,1,1,1]
model_pred  = model_pred * lora_scale                       # per-image 缩放
```

**作用**：去噪强度随 λ 自适应，高 λ 时 `lora_scale → 2`（更强去噪/生成），低 λ 时 `lora_scale → 1`（接近原始输出）。

**注意**：`lora_scale` 必须展开为 `[B, 1, 1, 1]`（per-image）而非对 batch 取均值，以保持与注入点 1、2、4 的 per-image 粒度一致。

#### 注入点 4 — AuxDecoder（DAux）

**位置**：DAux 内每个 BasicBlock 之后

```
BasicBlock(320)  →  FiLM(320, f(λ))  →  Upsample(320→256)  ← 注入
BasicBlock(256)  →  FiLM(256, f(λ))  →  Upsample(256→256)  ← 注入
BasicBlock(256)  →  FiLM(256, f(λ))  →  Upsample(256→256)  ← 注入
```

**作用**：DAux 负责结构分配，告知 gs 哪些结构已由辅助路径还原。条件化 λ 后，高码率时 DAux 分配更丰富的结构信息，低码率时让位给 UNet 做更多生成补偿。

### 3.5 编解码器接口变更

```python
# 编码
output_dict = codec.compress(latent, latent2, lmbda)
# output_dict 包含：
#   "strings":     [y_strings, z_strings]
#   "shape":       z.size()[-2:]
#   "lmbda_val":   lmbda.cpu().half().tolist()   ← float16，2字节/图
#   "t_star_val":  T_star.detach().cpu().tolist() ← 新增：动态时间步

# 解码（自动从 lmbda_val 恢复 λ，从 t_star_val 恢复 T*）
x_hat, res, T_star = codec.decompress(strings, shape, lmbda)
```

---

## 4. 动态时间步 T\*（DynamicTimestepModule）

### 4.1 动机

原始 StableCodec 使用固定时间步 T=999 进行一步去噪。然而在可变码率设定下，不同 λ 产生不同质量的量化隐变量：

- **高 λ（极端压缩）**：量化更粗，信息损失大 → 需要更强的去噪以补偿
- **低 λ（高码率）**：量化更精细，信息保留好 → 过强的去噪反而破坏细节

因此，去噪时间步应与压缩质量自适应匹配。

### 4.2 SNR 分析方法

`DynamicTimestepModule` 利用熵模型的 scales（即 4-pass checkerboard 上下文模型输出的 σ）来推断压缩信噪比（SNR），进而映射到 DDPM 的 SNR 调度表，得到语义等价的时间步 T\*。

**核心思想**：量化噪声方差为 `σ²_quant = 1/12`（均匀量化），信号方差为 `scales_mean²`，两者之比即为压缩 SNR。通过 DDPM 的 `alphas_cumprod` 调度表（单调递减的 SNR 曲线），用 `searchsorted` 找到匹配时间步。

### 4.3 实现细节

```python
class DynamicTimestepModule(nn.Module):
    def __init__(self, alphas_cumprod, t_min=800, t_max=999):
        super().__init__()
        self.t_min = t_min
        self.t_max = t_max
        ac = alphas_cumprod.float()
        self.register_buffer("alphas_cumprod", ac)
        snr_schedule = ac / (1.0 - ac)          # DDPM SNR schedule
        self.register_buffer("snr_schedule", snr_schedule)

    def forward(self, scales_all):
        """
        Args:
            scales_all: [B, 320, H, W] — 四趟 checkerboard 的累积 scales
        Returns:
            T_star: [B] float, 每张图一个时间步，范围 [t_min, t_max]
        """
        sigma_quant = 1.0 / 12.0                          # 均匀量化噪声方差
        signal_var  = scales_all.mean(dim=[1, 2, 3]) ** 2  # [B]
        snr_compress = signal_var / sigma_quant            # 压缩 SNR

        # 在 DDPM SNR 调度表中找到语义等价时间步
        T_raw = torch.searchsorted(-self.snr_schedule, -snr_compress)

        # 线性映射到 [t_min, t_max] 范围
        T_star = self.t_min + (self.t_max - self.t_min) * (T_raw.float() / 999.0)
        T_star = T_star.clamp(self.t_min, self.t_max)
        return T_star
```

**关键参数**：
- `t_min=800`：下界，防止时间步过小导致去噪不足
- `t_max=999`：上界，与 SD-Turbo 的训练时间步一致
- `alphas_cumprod`：从 DDPMScheduler 获取，长度 1000

**预期行为**：
- 高 λ → bpp 低 → scales 小 → SNR 低 → T\* 大（更强去噪）
- 低 λ → bpp 高 → scales 大 → SNR 高 → T\* 小（保守还原）
- 期望 `corr(logλ, T*)` 为正相关

### 4.4 `_batched_ddpm_step`：支持 per-image 时间步的 DDPM 一步去噪

`DDPMScheduler.step()` 不支持 batched timesteps（标量 t 对整个 batch 统一），因此需要手动实现。

SD-Turbo 使用 `set_timesteps(1)`，即调度器从 t 直接跳到 0（`alpha_prod_t_prev = 1.0`）。此时公式简化为：

```python
def _batched_ddpm_step(self, model_pred, timesteps, sample):
    """
    Args:
        model_pred: [B, C, H, W]  UNet 噪声预测
        timesteps:  [B]           per-image 时间步（long）
        sample:     [B, C, H, W]  待去噪隐变量

    Returns:
        x0_pred:    [B, C, H, W]  直接预测的干净隐变量
    """
    ac = self.sched.alphas_cumprod                        # [1000]
    t = timesteps.clamp(0, len(ac) - 1)                   # [B]
    alpha_prod_t = ac[t].view(-1, 1, 1, 1)                # [B, 1, 1, 1]

    sqrt_alpha_t = alpha_prod_t.sqrt()
    sqrt_one_minus_alpha_t = (1.0 - alpha_prod_t).sqrt()

    # x₀ 直接预测：一步跳到 t=0
    x0_pred = (sample - sqrt_one_minus_alpha_t * model_pred) / sqrt_alpha_t
    return x0_pred
```

---

## 5. 改进的损失函数

### 5.1 从初始设计到最终方案的演变

**初始设计**（`trainv1_variable2.py`）：

```python
# TargetRateModule 内部：
rate_loss = (lmbda_t * total_bpp).mean()     # λ 加权 bpp

# 训练损失：
lambda_scale  = lmbda.mean() / lambda_ref
generator_loss = compression_loss * lambda_scale + rate_loss
# 等价于 ≈ mean(λ)/lambda_ref × D + mean(λ × bpp)
```

**问题**：`lambda_scale` 机制虽然解决了 D/R 梯度失衡问题，但引入了额外的超参数依赖（`lambda_ref`）并且在极端 λ 值处不够稳定。

**最终方案**（`trainv1_variable3_debug_step.py`）：

```python
# TargetRateModule 内部：
rate_loss = total_bpp.mean()                  # 不加权，直接取均值

# 训练损失：
per_image_dist = out_criterion["per_image_distortion"]  # [B]
inv_lambda_distortion = (per_image_dist / lmbda).mean()
generator_loss = inv_lambda_distortion + rate_loss
# 即：L = mean(D_i / λ_i) + mean(bpp_i)
```

### 5.2 最终损失公式分析

$$L = \frac{1}{B} \sum_{i=1}^{B} \frac{D_i}{\lambda_i} + \frac{1}{B} \sum_{i=1}^{B} \text{bpp}_i$$

**梯度行为**：
- **高 λ → D/λ 小 → bpp 项主导 → 模型倾向于降低码率**
- **低 λ → D/λ 大 → 失真项主导 → 模型倾向于提高重建质量**

**优势**：
1. 无需 `lambda_scale` 超参数，公式自然平衡
2. `D_i/λ_i` 是标准率失真理论中 `L = D + λR` 的等价形式（两侧同除 λ）
3. 不加权的 `mean(bpp)` 使码率项不随 λ 采样分布偏移

### 5.3 TargetRateModule

```python
class TargetRateModule(nn.Module):
    def forward(self, ..., lmbda: Tensor, ...):
        total_bpp  = latent_bpp + hyper_bpp
        rate_loss  = total_bpp.mean()                     # 不加权
        per_image_bpp = total_bpp.detach()                # [B]，用于诊断
        ...
```

`per_image_bpp` 字段保存逐图 bpp，供 `diag_buffer` 收集并用于关联分析。

---

## 6. 前向传播完整流程

```
Image x  [B, 3, H, W]
  │
  ├─► ELIC g_a (frozen) ─► aux_latent [B, 320, H/16, W/16]
  │
  └─► VAE encoder ─► lq_latent [B, 256, H/8, W/8]
         │
         ▼
   LatentCodec.g_a (注入点 1: FiLM(f(λ)))
     ─► y  [B, 320, H/32, W/32]
         │
   HyperAnalysis(y) → z → EntropyBottleneck
   HyperSynthesis(z_hat) → base [B, 320, H/32, W/32]
   4-pass checkerboard context model
     ─► y_hat [B, 320, H/32, W/32]
     ─► scales_all [B, 320, H/32, W/32]   ← 用于计算 T*
         │
   ┌─────┼──────────────────────────────────┐
   │     │                                  │
   │  DynamicTimestepModule(scales_all)     │
   │     ─► T_star [B]                      │
   │                                        │
   │  g_s (注入点 2: FiLM(f(λ)))            │
   │  [B, 320, H/8, W/8]                   │
   │     │                                  │
   │  UNet conv_in (320→320)                │
   │  UNet (+ LoRA) at timestep=T_star      │  ← 动态时间步
   │  model_pred * lora_scale               │  ← 注入点 3
   │  _batched_ddpm_step(model_pred, T*, .) │  ← per-image 时间步
   │     │                                  │
   │     │          aux (注入点 4: FiLM(f(λ)))
   │     │          [B, 256, H/8, W/8] ── res1 skip
   │     │                                  │
   │     └──────────── + res1 ──────────────┘
   │
   ▼
 VAE decoder ─► output_image [B, 3, H, W]

返回值：(output_image, RateLossOutput, T_star)
```

---

## 7. 变量码率训练策略

### 7.1 λ 采样

```python
def sample_lambda(B, lambda_min, lambda_max, device):
    log_min = math.log(lambda_min)
    log_max = math.log(lambda_max)
    return torch.exp(
        torch.empty(B, dtype=torch.float32, device=device).uniform_(log_min, log_max)
    )
```

对数均匀分布保证 RD 曲线各点覆盖密度均等。

**`lambda_sample_max`**：训练采样上界可独立于 FiLM 嵌入边界设置。当 FiLM 嵌入范围为 `[0.1, 128.0]` 时，实际训练可只采样 `[lambda_min, lambda_sample_max]`（例如 `[0.2, 32.0]`），使模型聚焦在实用码率区间，而嵌入空间仍保留外推能力。

```python
lambda_sample_max = float(config.get('lambda_sample_max', lambda_max))
lmbda = sample_lambda(B, lambda_min, lambda_sample_max, device)
```

### 7.2 验证策略

验证时使用固定的 `lambda_ref`，保证各 checkpoint 之间指标可比：

```python
lambda_ref = float(config.get('lambda_ref', 1.0))
lmbda_val = torch.full((B,), lambda_ref, dtype=torch.float32, device=device)
x_hat, RateLossOutput, T_star = model(d, pos_tag_prompt, H, W, lmbda=lmbda_val)
```

验证不使用 `t_override`，直接使用 codec 输出的自然 T\*。

---

## 8. 诊断系统（Debug Diagnostics）

### 8.1 `diag_buffer`

训练过程中逐图收集五个指标：

```python
diag_buffer = {"lambda": [], "bpp": [], "dist": [], "lambda_dist": [], "t_star": []}
```

每个训练步骤后追加当前 batch 的 detached per-image 值。

### 8.2 `_safe_corr`

安全的 Pearson 相关系数计算，处理样本不足或零方差的退化情况：

```python
def _safe_corr(x_arr, y_arr):
    x = np.asarray(x_arr, dtype=np.float64)
    y = np.asarray(y_arr, dtype=np.float64)
    if x.size < 4 or np.allclose(x.std(), 0.0) or np.allclose(y.std(), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])
```

### 8.3 `_log_diag_and_clear`

每 `diag_every` 步（默认 500）输出一次诊断报告并清空 buffer。报告包含：

**全局统计**：
- `corr(logλ, bpp)` — 期望负相关（λ 越大 bpp 越低）
- `corr(logλ, dist)` — 期望负相关（λ 越大失真越大但 dist 计算基于 D/λ）
- `corr(logλ, λ·dist)` — 期望正相关
- `corr(logλ, T*)` — 期望正相关（λ 越大 → scales 越小 → T\* 越大）
- `mean_bpp`、`range_bpp`、`mean_T*`

**per-λ-bin 分析**（默认 6 个 bin）：
- 对 `logλ` 等间隔分 bin，每 bin 报告 `mean_λ`、`mean_bpp`、`mean_dist`、`mean_λdist`、`mean_T*`
- 用于验证不同 λ 区间的模型行为是否符合预期

---

## 9. 关键 Bug 修复记录

### Bug 1：chunk view 原地操作报错

**现象**：
```
RuntimeError: Output 0 of SplitBackward0 is a view and is being modified inplace.
```

**原因**：`.chunk(2, 1)` 返回的是原张量的 view，对 view 执行 `*=` 或 `+=` 会破坏 autograd 计算图。

**修复**：

```python
# 错误（原地操作 view）
means_0, scales_0 = ...adapter_out[0](...).chunk(2, 1)
means_0 *= m0;  scales_0 *= m0
y_hat_0 += lrp

# 正确（非原地赋值）
means_0, scales_0 = ...adapter_out[0](...).chunk(2, 1)
means_0  = means_0  * m0;  scales_0 = scales_0 * m0
y_hat_0  = y_hat_0 + lrp
```

### Bug 2：注入点 3 的 λ 粒度不一致

**现象**：注入点 1、2、4 均为 per-image（`[B, 512]`），但注入点 3 对 batch 取了均值（`film_embed.mean(0)`），导致整个 batch 共用同一 `lora_scale`。

**修复**：

```python
# 错误
delta_s    = self.unet_lora_proj(film_embed.mean(0, keepdim=True))  # [1, 1]

# 正确
delta_s    = self.unet_lora_proj(film_embed)                        # [B, 1]
lora_scale = (1.0 + torch.tanh(delta_s)).view(B, 1, 1, 1)          # [B, 1, 1, 1]
```

### Bug 3：初始设计中 bpp 持续下降至 0

**原因**：Stage I 就对 `[0.5, 32]` 全范围采样 λ 且删掉了 `lambda_scale`，导致码率梯度远大于失真梯度。

**最终解决方案**：改用 `D/λ + bpp` 损失公式，彻底消除了梯度量级失衡问题，不再需要 `lambda_scale` 超参数。

---

## 10. Config 配置变更

在原有 `stage1.yaml` / `stage2.yaml` 基础上新增以下字段：

```yaml
# 可变码率范围
lambda_ref: 1.0           # 验证和 criterion 的参考 λ

model:
  lambda_min: 0.1          # FiLM 嵌入 & 训练 λ 下界
  lambda_max: 128.0        # FiLM 嵌入 λ 上界（保守宽边界）
  lambda_sample_max: 32.0  # 训练采样 λ 上界（可窄于 lambda_max）

# 诊断参数
diag_every: 500            # 诊断输出间隔步数
diag_num_bins: 6           # per-λ-bin 分析的 bin 数量
```

---

## 11. 新增参数量估算

| 模块 | 新增参数 | 说明 |
|---|---|---|
| `LambdaFiLMEmbed`（共享嵌入）| ~21×512×2 ≈ 21K | Fourier → MLP |
| `AnalysisTransform` FiLM × 3 | 2×(512×C)×3 ≈ 1.5M | C=192/256/320 各一层 |
| `SynthesisTransform` FiLM × 3 | 2×(512×320)×3 ≈ 1M | |
| `AuxDecoder` FiLM × 3 | 2×(512×320+512×256)×3 ≈ 1.7M | |
| `unet_lora_proj`（注入点 3） | 512×64 + 64×1 ≈ 33K | |
| `DynamicTimestepModule` | 0 | 纯分析模块，无可学习参数（仅 buffer） |
| **合计** | **≈ 4.3M** | 相对原模型参数量可忽略 |

---

## 12. Checkpoint 格式

```python
{
    "epoch":                epoch,
    "global_step":          global_step,
    "state_dict_codec":     codec.state_dict(),    # 含 FiLM 参数 + DynamicTimestepModule buffers
    "state_dict_aux_codec": aux_codec.state_dict(),# ELIC g_a 权重
    "state_dict_vae":       {...},                 # 仅 LoRA + mismatched keys
    "state_dict_unet":      {...},                 # LoRA + conv_in + mismatched keys
    "state_dict_lora_proj": unet_lora_proj.state_dict(),  # 注入点 3
    "optimizer":            optimizer.state_dict(),
    "aux_optimizer":        aux_optimizer.state_dict(),
    "lr_scheduler":         lr_scheduler.state_dict(),
    # Stage 2 额外：
    "state_dict_disc":      net_disc.state_dict(),
    "disc_optimizer":       disc_optimizer.state_dict(),
    "disc_lr_scheduler":    disc_lr_scheduler.state_dict(),
    # EMA：
    "ema_state_dict":       ema_net.state_dict(),
}
```

Shape-safe loading：mismatched keys 会被跳过（logged），支持从不同配置的 checkpoint 部分加载作为 warm-start。`DynamicTimestepModule` 作为 codec 子模块自动包含在 `state_dict_codec` 中。

---

## 13. 文件清单与修改摘要

| 文件 | 主要内容 |
|---|---|
| `latent_codec_variable2_step.py` | `LambdaFiLMEmbed`、`FiLMLayer`；FiLM-conditioned `AnalysisTransform`/`SynthesisTransform`/`AuxDecoder`；`DynamicTimestepModule`（SNR → T\*）；`TargetRateModule`（不加权 `mean(bpp)`）；4-pass checkerboard entropy coding；`compress` 输出 `lmbda_val` + `t_star_val` |
| `StableCodec_variable2_step.py` | 封装 SD-Turbo + LoRA；`unet_lora_proj`（注入点 3）；`_batched_ddpm_step`（per-image DDPM 一步）；`forward/compress/decompress` 使用动态 T\*；`set_train` 解冻 FiLM 和 `unet_lora_proj` |
| `trainv1_variable3_debug_step.py` | `sample_lambda` 对数均匀采样；`D/λ + bpp` 损失公式；`lambda_sample_max`；`diag_buffer` + `_safe_corr` + `_log_diag_and_clear` 诊断系统；T\* 日志；验证固定 `lambda_ref` |

---

## 14. 训练流程总结

```
Stage I（~100k iters）
  λ ~ LogUniform(lambda_min, lambda_sample_max)（per-image）
  目标：让模型学会基本率失真重建 + λ 条件化
  损失：L = mean(D_i / λ_i) + mean(bpp_i)
  T*：由 DynamicTimestepModule 根据 scales 自动确定

      ↓  加载 Stage I 最优 checkpoint

Stage II（~20k iters per λ-range）
  λ ~ LogUniform(lambda_min, lambda_sample_max)（per-image）
  目标：让模型学会跨码率连续调控
  损失：L = mean(D_i / λ_i) + mean(bpp_i) + GAN loss
  加入 GAN 判别器（DINOv2）

      ↓  推理

解码端选择 λ ∈ [lambda_min, lambda_max]（2字节写入码流）
四注入点读取同一 f(λ) → 解码图像码率连续可调
T* 由 DynamicTimestepModule 自动确定（无需额外信息传输）
```

---

## 15. 与初始设计的关键差异对照

| 维度 | 初始设计（sc-variable.md） | 最终设计（本文档） |
|---|---|---|
| 时间步 | 固定 T=999 | 动态 T\*∈[800,999]，SNR-based |
| DDPM step | `sched.step()`（标量 t） | `_batched_ddpm_step`（per-image t） |
| 损失函数 | `mean(λ)/λ_ref × D + mean(λ×bpp)` | `mean(D/λ) + mean(bpp)` |
| TargetRateModule | `rate_loss = mean(λ×bpp)` | `rate_loss = mean(bpp)` |
| lambda_scale | 需要，`mean(λ)/λ_ref` | 不需要 |
| 诊断系统 | 无 | diag_buffer + corr 分析 + per-bin |
| lambda_sample_max | 无，采样=嵌入范围 | 独立配置，可窄于嵌入范围 |
| Checkpoint | codec + vae + unet + lora_proj | 同左 + aux_codec + DynamicTimestep buffers |
| FiLM 嵌入边界 | `λ_min=0.5, λ_max=32.0` | `LAMBDA_MIN=0.1, LAMBDA_MAX=128.0`（保守宽边界） |
| per_image_bpp | 无 | 新增，用于诊断 |
