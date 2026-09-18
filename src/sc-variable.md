# StableCodec 可变码率方案设计报告

> 基于 I2C（TPAMI 2024）与 MRIC（CVPR 2023）的 λ-FiLM 四注入点架构

---

## 1. 背景与目标

原始 StableCodec 采用固定 λ 的两阶段训练策略（Stage I λ=0.5，Stage II 在离散 λ 集合上微调），每个码率点需要独立训练一个模型，部署成本高。本方案的目标是：

- 单一模型覆盖连续码率范围（约 0.005–0.05 bpp）
- 解码端通过改变 λ 实现实时码率切换
- λ 以 2 字节（float16）存入码流，提供 65536 个有效码率点
- 对原始 StableCodec 架构的侵入最小，保持兼容性

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

超参数：`λ_min=0.5`，`λ_max=32.0`，`NUM_FREQS=10`，`FILM_DIM=512`

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

**设计说明**：EAux（冻结）和 ESD 的输出在进入 ga 前以原始 concat 方式融合，λ 仅通过 FiLM 注入 ga 的特征变换层，不干预两路特征的融合权重。这是本方案对原始"注入点 1 包含加权融合"设计的关键修正——EAux 冻结且直接 concat，不适合引入 α(λ) 加权求和。

#### 注入点 2 — SynthesisTransform（gs）

**位置**：gs 内每个 BasicBlock 之后

```
BasicBlock(320)  →  FiLM(320, f(λ))  →  Upsample(320→320)   ← 注入
BasicBlock(320)  →  FiLM(320, f(λ))  →  Upsample(320→320)   ← 注入
BasicBlock(320)  →  FiLM(320, f(λ))  →  Upsample(320→320)   ← 注入
```

**作用**：gs 产生送入 ϵSD 的噪声隐变量 lT。高 λ（高压缩）→ gs 生成更粗糙的 lT，让 Unet 去噪更激进；低 λ（低压缩）→ lT 更精细，Unet 保守还原。

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

**作用**：DAux 负责结构分配，告知 gs 哪些结构已由辅助路径还原。条件化 λ 后，高码率时 DAux 分配更丰富的结构信息，低码率时让位给 Unet 做更多生成补偿。

### 3.5 编解码器接口变更

```python
# 编码
output_dict = codec.compress(latent, latent2, lmbda)
# output_dict 包含：
#   "strings":   [y_strings, z_strings]
#   "shape":     z.size()[-2:]
#   "lmbda_val": lmbda.cpu().half().tolist()   ← 新增：float16，2字节/图

# 解码（自动从 lmbda_val 恢复 λ）
x_hat, res = codec.decompress(strings, shape)           # lmbda 从 strings 读取
# 或显式指定
x_hat, res = codec.decompress(strings, shape, lmbda=0.5)
```

---

## 4. 变量码率训练策略

### 4.1 λ 采样

```python
def sample_lambda(B, lambda_min, lambda_max, device):
    """对数均匀采样，每张图独立采样一个 λ。"""
    log_min = math.log(lambda_min)
    log_max = math.log(lambda_max)
    return torch.exp(
        torch.empty(B, dtype=torch.float32, device=device).uniform_(log_min, log_max)
    )
```

对数均匀分布保证 RD 曲线各点覆盖密度均等（对应 I2C 的指数映射 Λ = θ·e^{τL}）。

### 4.2 损失函数

#### Stage I（固定 λ，模型先学基本重建）

```python
lmbda = torch.full((B,), lambda_ref, dtype=torch.float32, device=device)

# 失真项（由 criterion 计算，lambda_ref 标定系数）
out_criterion = criterion(x_hat, d)   # MSE + LPIPS + CLIP

# 码率项（TargetRateModule 内已含 λ 加权）
out_criterion['bpp_loss'] = RateLossOutput.rate_loss   # λ_ref × bpp

generator_loss = out_criterion["compression_loss"] + out_criterion['bpp_loss']
```

#### Stage II（可变 λ，全范围覆盖）

```python
lmbda = sample_lambda(B, lambda_min, lambda_max, device)   # [B] 对数均匀

out_criterion = criterion(x_hat, d)           # D（用 lambda_ref 标定）
out_criterion['bpp_loss'] = RateLossOutput.rate_loss  # mean(lmbda × bpp)

# lambda_scale 将 D 按当前批次 λ 均值等比例放大
# 目的：保持 D 和 R 的相对梯度量级在不同 λ 下稳定
# 等价于总损失 ≈ mean(λ)/lambda_ref × D + mean(λ × bpp)
lambda_scale  = lmbda.mean() / lambda_ref
generator_loss = out_criterion["compression_loss"] * lambda_scale \
               + out_criterion['bpp_loss']
```

`lambda_scale` 的作用：`criterion` 内的失真系数（`mse_coeff`、`lpips_coeff` 等）是用 `lambda_ref` 标定的，直接把这些系数乘到高 λ 场景会导致失真项相对码率项严重偏小，模型只会压码率。`lambda_scale` 把失真项等比例拉回正确量级，等价于对当前批次实施了 `L = λ·D + R` 的标准率失真目标。

#### Stage II λ 范围建议（渐进式扩大）

```
初始：[0.5, 4.0]   → 先在小范围稳定
中期：[0.5, 16.0]  → 扩大覆盖
后期：[0.5, 32.0]  → 全范围训练
```

### 4.3 TargetRateModule 修改

```python
class TargetRateModule(nn.Module):
    """无固定 λ，forward 时接收 per-image [B] 张量。"""
    def forward(self, ..., lmbda: Tensor, ...):
        total_bpp  = latent_bpp + hyper_bpp
        lmbda_t    = lmbda.to(total_bpp.device).expand_as(total_bpp)
        rate_loss  = (lmbda_t * total_bpp).mean()   # per-image 加权后取均值
        ...
```

### 4.4 验证策略

验证时使用固定的 `lambda_ref`（推荐与原始 StableCodec 基准点一致），保证各 checkpoint 之间指标可比：

```python
lmbda_val = torch.full((B,), lambda_ref, dtype=torch.float32, device=device)
x_hat, RateLossOutput = model(d, pos_tag_prompt, H, W, stage, lmbda=lmbda_val)
```

---

## 5. 关键 Bug 修复记录

### Bug 1：chunk view 原地操作报错

**现象**：
```
RuntimeError: Output 0 of SplitBackward0 is a view and is being modified inplace.
```

**原因**：`.chunk(2, 1)` 返回的是原张量的 view，对 view 执行 `*=` 或 `+=` 会破坏 autograd 计算图。

**位置**：`latent_codec_variable.py`，`forward` 中四个 quadtree 步骤。

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

`compress` / `decompress` 中的 `yh0 +=` 作用在 `unsequeeze_with_mask` 返回的 `torch.cat(...)` 新张量上，不是 view，无需修改。

### Bug 2：注入点 3 的 λ 粒度不一致

**现象**：注入点 1、2、4 均为 per-image（`[B, 512]`），但注入点 3 对 batch 取了均值（`film_embed.mean(0)`），导致整个 batch 共用同一 `lora_scale`。

**修复**：

```python
# 错误
delta_s    = self.unet_lora_proj(film_embed.mean(0, keepdim=True))  # [1, 1]
lora_scale = 1.0 + torch.tanh(delta_s)                              # 标量

# 正确
delta_s    = self.unet_lora_proj(film_embed)                        # [B, 1]
lora_scale = (1.0 + torch.tanh(delta_s)).view(B, 1, 1, 1)          # [B, 1, 1, 1]
model_pred = model_pred * lora_scale                                # per-image
```

### Bug 3：bpp 持续下降至 0 不动

**原因**：Stage I 就对 `[0.5, 32]` 全范围采样 λ，对数均匀分布几何均值约为 4，导致：

```
L = D + mean(λ) × bpp ≈ D + 4 × bpp
```

同时删掉了 `lambda_scale`，D 没有对应的 4 倍放大。码率梯度远大于失真梯度，模型不断把 bpp 压向 0。

**修复**：
1. Stage I 固定 `λ = lambda_ref`，不做宽范围采样
2. Stage II 恢复 `lambda_scale = mean(lmbda) / lambda_ref`，平衡 D 与 R 的梯度量级
3. Stage II λ 范围从窄到宽渐进式扩大

---

## 6. Config 配置变更

在原有 `stage1.yaml` / `stage2.yaml` 基础上新增以下字段：

```yaml
# 可变码率范围（全局）
lambda_ref: 1.0        # 验证和 criterion 的参考 λ（推荐与原始基准一致）

model:
  lambda_min: 0.5      # 训练 λ 下界
  lambda_max: 32.0     # 训练 λ 上界（Stage II 后期）
  # 注：不再使用原始的 'labda' 字段作为训练控制变量
  #     'labda' 可保留用于向后兼容的 checkpoint 加载
```

Stage I 专用：

```yaml
# stage1.yaml
lambda_min: 0.5
lambda_max: 0.5        # Stage I 固定 λ，等价于原始单点训练
lambda_ref: 0.5
```

Stage II 专用：

```yaml
# stage2.yaml
lambda_min: 0.5
lambda_max: 4.0        # 初始范围，训练稳定后可扩至 32
lambda_ref: 1.0
```

---

## 7. 新增参数量估算

| 模块 | 新增参数 | 说明 |
|---|---|---|
| `LambdaFiLMEmbed`（共享嵌入）| ~21×512×2 ≈ 21K | Fourier → MLP |
| `AnalysisTransform` FiLM × 3 | 2×(512×C)×3 ≈ 1.5M | C=192/256/320 各一层 |
| `SynthesisTransform` FiLM × 3 | 2×(512×320)×3 ≈ 1M | |
| `AuxDecoder` FiLM × 3 | 2×(512×320+512×256)×3 ≈ 1.7M | |
| `unet_lora_proj`（注入点 3） | 512×64 + 64×1 ≈ 33K | |
| **合计** | **≈ 4.3M** | 相对原模型参数量可忽略 |

---

## 8. 文件清单与修改摘要

| 文件 | 主要修改 |
|---|---|
| `latent_codec_variable.py` | 新增 `LambdaFiLMEmbed`、`FiLMLayer`；改写 `AnalysisTransform`、`SynthesisTransform`、`AuxDecoder` 插入 FiLM；`TargetRateModule` 改为接收 per-image `lmbda`；`compress` 输出 `lmbda_val`；修复 chunk view 原地操作 Bug |
| `StableCodec_variable.py` | 新增 `unet_lora_proj`（注入点 3）；`forward/compress/decompress` 接受 `lmbda` 参数；`set_train` 解冻 `film_embed` 和 `unet_lora_proj`；`save_model` 保存 `state_dict_lora_proj`；修复 per-image `lora_scale` Bug |
| `stablecodec-trainv1_variable.py` | 新增 `sample_lambda`；Stage I 固定 λ，Stage II 对数均匀采样；恢复 `lambda_scale` 平衡 D/R 梯度；验证固定 `lambda_ref`；checkpoint 保存 `state_dict_lora_proj` |

---

## 9. 训练流程总结

```
Stage I（~100k iters）
  λ = lambda_ref（固定）
  目标：让模型学会基本率失真重建
  损失：D + lambda_ref × bpp

      ↓  加载 Stage I 最优 checkpoint

Stage II（~20k iters per λ-range）
  λ ~ LogUniform(lambda_min, lambda_max)（per-image）
  目标：让模型学会跨码率连续调控
  损失：(mean(λ)/lambda_ref) × D + mean(λ × bpp)
  λ 范围渐进：[0.5, 4] → [0.5, 16] → [0.5, 32]
  加入 GAN 判别器（DINOv2）

      ↓  推理

解码端选择 λ ∈ [0.5, 32]（2字节写入码流）
四注入点读取同一 f(λ) → 解码图像码率连续可调
```