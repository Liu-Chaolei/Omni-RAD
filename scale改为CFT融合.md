# CFT 融合：以 UNet 为 prior、Aux 条件调制的过渡训练方案

> 目标：参考 CodeFormer 的 Controllable Feature Transformation（CFT）机制，在保留 `SNR → T*` 全局时间步和 entropy scales 局部不确定性信息的基础上，替代原先“scale/map 直接缩放 UNet 输出”的融合方式。Stage I 仍使用 `x0_pred + res1` 直接相加训稳主干；Stage II 以 `x0_pred` 作为 CFT prior 主体，让 `res1 / scales` 仅通过 `alpha,beta` 修正 UNet 输出，并用 `rho` 从直接相加平滑过渡到 CFT 融合。`sample` 不进入主 CFT predictor，仅作为消融项和诊断统计使用。

---

## 1. 修改动机

`scale逐空间通道.md` 中的核心融合方式是：

```python
delta_unet = x0_pred - sample
M = gate_module(scales_all, target_hw=sample.shape[-2:])
z_hat = sample + M * delta_unet + res1
```

该设计让 `scales_all` 产生逐空间或逐通道门控图 `M`，直接决定 UNet 修复残差的注入强度。它的问题是：

1. `M` 只能做强度缩放，不能改变 UNet 特征的局部形态；
2. `res1` 仍直接加到最终 latent，Aux 分支中的退化信息可能与 UNet 生成先验冲突；
3. 融合机制仍是“谁占多少比例”的门控逻辑，不是更细粒度的特征变换。

CodeFormer CFT 的关键思想是：

```text
prior feature 作为主体
input / condition feature 不直接 skip 到输出
condition feature 只负责预测 alpha,beta
再用可控系数 w 控制调制强度
```

因此，本方案将 Stage II 中原来的：

```text
M × UNet residual
```

改为：

```text
Aux / scales 条件 → alpha,beta → 仿射修正 UNet prior
```

---

## 2. 核心选择：prior 用 `x0_pred`，不是 `x0_pred + res1`

为了贴近 CodeFormer CFT 的语义，CFT 的主特征必须是相对“高质量先验”的特征。在本方案中对应：

```python
z_prior = x0_pred
```

`res1` 不再直接作为 CFT 主体的一部分，而是作为条件特征参与预测 `alpha,beta`：

```python
alpha, beta = P_theta(concat(x0_pred, res1, scale_feat), f_lambda)
```

这样 CFT 的机制才是：

```text
用 Aux / entropy 条件去修正 UNet prior
```

而不是：

```text
用条件再修正一个已经把 UNet 和 Aux 混合过的 base
```

但为了避免 Stage II 一开始突然丢掉 Stage I 的 `res1` 融合能力，本方案引入过渡系数 `rho`：

```python
z_direct = x0_pred + res1
z_cft = x0_pred + w * (alpha * x0_pred + beta)
z_hat = (1 - rho) * z_direct + rho * z_cft
```

其中：

```text
rho: 0 → 1
```

训练刚开始：

```python
rho = 0
z_hat = x0_pred + res1
```

完全等价于 Stage I。

训练后期：

```python
rho = 1
z_hat = x0_pred + w * (alpha * x0_pred + beta)
```

完全切换到 CFT 融合。

---

## 3. 保留与删除

## 3.1 保留

以下部分保持不变：

1. 256 通道 VAE latent；
2. `λ-FiLM` 条件化；
3. 4-pass checkerboard entropy model；
4. `DynamicTimestepModule` 的 `scales → SNR → T*`；
5. UNet 在 per-image `T*` 上执行一步去噪；
6. AuxDecoder 输出 256 通道特征 `res1`；
7. `scales_all` 作为解码端可重算的不确定性信息。

## 3.2 删除或降级

删除原方案中的直接门控融合：

```python
z_hat = sample + M * (x0_pred - sample) + res1
```

不再让 `M` 直接缩放：

- `model_pred`
- `x0_pred`
- `x0_pred - sample`
- `res1`

若保留由 `scales_all` 得到的局部不确定性特征，它只作为 CFT predictor 的输入条件，而不是直接乘在 UNet 输出上。

---

## 4. CFT 融合核心设计

## 4.1 特征角色定义

参考 CodeFormer，先明确主从关系：

```text
主特征 F_prior：UNet 一步去噪后的 prior latent
条件特征 F_cond ：Aux / entropy scales 提供的保真和不确定性信息
```

在本方案中：

```python
F_prior = x0_pred                 # [B,256,H/8,W/8]
F_aux   = res1                    # [B,256,H/8,W/8]
F_scale = scale_feat(scales_all)  # [B,Cs,H/8,W/8]
```

其中：

- `x0_pred` 是 UNet 在 `T*` 上一步去噪得到的 latent，作为 CFT 主体；
- `res1` 是 AuxDecoder 输出的结构/保真条件；
- `scale_feat` 是由 `scales_all` 提取的局部不确定性特征；
- `f(λ)` 作为全局码率条件调制 CFT predictor。

---

## 4.2 CFT 融合公式

Stage II 的 CFT 输出为：

$$
z_{cft} = z_{prior} + w \cdot (\alpha \odot z_{prior} + \beta)
$$

其中：

$$
\alpha, \beta = P_{\theta}(\text{concat}(z_{prior}, z_{aux}, \phi_s(scales\_all)), f(\lambda))
$$

为了保持 Stage II 初始稳定，引入过渡融合：

$$
\hat{z} = (1-\rho)(x0\_pred + res1) + \rho z_{cft}
$$

对应代码形式：

```python
z_prior = x0_pred
z_direct = x0_pred + res1

cond = torch.cat([z_prior, res1, scale_feat], dim=1)
alpha, beta = cft_predictor(cond, film_embed).chunk(2, dim=1)

z_cft = z_prior + w * (alpha * z_prior + beta)
z_hat = (1.0 - rho) * z_direct + rho * z_cft
```

与旧方案相比：

```python
# 旧：直接缩放 UNet residual
z_hat = sample + M * (x0_pred - sample) + res1

# 新：CFT 条件调制 UNet prior，并用 rho 平滑过渡
z_prior = x0_pred
z_direct = x0_pred + res1
z_cft = z_prior + w * (alpha * z_prior + beta)
z_hat = (1 - rho) * z_direct + rho * z_cft
```

关键变化是：

- CFT 主体是 `x0_pred`，保持“Aux 修正 UNet prior”的语义；
- `res1 / scales` 不直接相加或门控输出；
- `res1 / scales` 只参与预测 `alpha,beta`；
- `sample` 不进入主 CFT predictor，避免压缩退化信息泄漏；
- `rho` 负责训练过渡稳定性；
- `w` 控制 CFT 信息流强弱。

---

## 4.3 `w` 的含义

`w ∈ [0,1]` 是 CFT 注入强度，参考 CodeFormer 的 quality-fidelity knob。

在本方案中：

```text
w 小：更依赖 UNet prior，感知质量更强，保真约束更弱
w 大：更多使用 Aux/scales 条件修正，结构和像素保真更强
```

推荐两种实现。

### 方案 A：固定全局 `w`

```text
w = 0.5
```

训练时可设：

```text
w_train = 1.0
```

推理时扫描：

```text
w ∈ {0, 0.25, 0.5, 0.75, 1.0}
```

这是第一版实验，先验证 CFT 本身是否有效。

### 方案 B：由 decoder-side 状态自适应预测 `w`

```python
w = WPolicyNet(global_stats)
```

`w` 不建议只由 `λ` 或单一 SNR 决定。更合理的是让它根据当前样本的压缩可靠性、熵模型不确定性、Aux 分支强度以及 UNet/Aux 一致性共同决定。

推荐输入：

```text
[
  logλ,
  bpp,
  T*_norm,
  snr_compress,
  scales_mean,
  scales_std,
  scales_p90,
  res1_norm / (z_direct_norm + eps),
  cosine(delta_unet, res1)
]
```

其中：

```python
delta_unet = x0_pred - sample
z_direct = x0_pred + res1
```

直觉如下：

- 高压缩 / 高 λ / 大 `T*`：减小 `w`，避免退化 codec 信息过强污染 UNet；
- 低压缩 / 低 λ / 小 `T*`：增大 `w`，更多保留输入结构和细节；
- `scales` 整体大或空间波动大：减小 `w`；
- `res1` 相对 `z_direct` 过大：减小 `w`，避免 Aux 过强主导；
- `delta_unet` 与 `res1` 方向一致：增大 `w`，说明 UNet 与 Aux 协同；
- `delta_unet` 与 `res1` 方向冲突：减小 `w`，避免放大冲突修正。

首轮实验建议先用方案 A；第二版实验再启用方案 B。

---

## 5. 模块结构

## 5.1 `ScaleFeatureAdapter`

将 `scales_all` 从 entropy latent 尺度映射到 VAE latent 尺度：

```text
scales_all [B,320,H/32,W/32]
  → log(1 + scales_all)
  → 1x1 Conv(320→64)
  → Upsample ×4
  → 3x3 Conv(64→64)
  → scale_feat [B,64,H/8,W/8]
```

这里的 `scale_feat` 不再输出门控图 `M`，而是作为 CFT predictor 的条件特征。

## 5.2 `CFTFusionModule`

最小版本：

```python
class CFTFusionModule(nn.Module):
    def __init__(self, latent_ch=256, scale_ch=64, film_dim=512):
        super().__init__()
        in_ch = latent_ch * 2 + scale_ch  # x0_pred, res1, scale_feat
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.film_to_affine = nn.Linear(film_dim, 512)
        self.out = nn.Conv2d(256, latent_ch * 2, 3, padding=1)

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z_prior, z_aux, scale_feat, film_embed, w, rho):
        cond = torch.cat([z_prior, z_aux, scale_feat], dim=1)
        h = self.body(cond)

        gamma, beta_film = self.film_to_affine(film_embed).chunk(2, dim=1)
        gamma = gamma.view(-1, 256, 1, 1)
        beta_film = beta_film.view(-1, 256, 1, 1)
        h = h * (1.0 + gamma) + beta_film

        alpha, beta = self.out(h).chunk(2, dim=1)
        z_cft = z_prior + w.view(-1, 1, 1, 1) * (alpha * z_prior + beta)
        z_direct = z_prior + z_aux
        z_hat = (1.0 - rho) * z_direct + rho * z_cft
        return z_hat, z_cft, alpha, beta
```

设计要点：

1. `out` 零初始化，使刚启用 CFT 时 `alpha≈0,beta≈0`；
2. 当 `rho=0` 时，输出完全等价于 `x0_pred + res1`；
3. 随着 `rho→1`，输出逐步切换为 CFT；
4. `res1` 不再直接作为 CFT 主体，而是条件特征；
5. `scale_feat` 只作为条件，不直接控制乘法门。

## 5.3 `rho` 过渡调度

`rho` 用于从直接相加平滑过渡到 CFT：

```text
rho ∈ [0,1]
```

推荐 warm-up：

```python
rho = min(1.0, max(0.0, (global_step - rho_start) / rho_warmup_steps))
```

建议：

```text
rho_start = Stage II 开始
rho_warmup_steps = Stage II 前 10%～30% steps
```

也可以使用 sigmoid 调度：

```python
rho = sigmoid(k * (progress - 0.5))
```

第一版建议线性调度，便于诊断。

## 5.4 `WPolicyNet`：自适应 CFT 强度

`WPolicyNet` 是第二版实验使用的轻量全局策略网络，用于为每张图预测 CFT 注入强度：

```text
w_i ∈ [w_min, w_max]
```

推荐范围：

```text
w_min = 0.1
w_max = 0.8
```

不建议一开始允许 `w=0` 或 `w=1`，避免完全关闭或完全放大 CFT。

输入为 decoder-side 可观测全局统计量：

```python
global_stats = [
    log_lambda,
    bpp,
    T_star_norm,
    snr_compress,
    scales_mean,
    scales_std,
    scales_p90,
    res1_norm / (z_direct_norm + eps),
    cosine(delta_unet, res1),
]
```

输出：

```python
w_raw = mlp(global_stats)
w = w_min + (w_max - w_min) * torch.sigmoid(w_raw)
```

伪代码：

```python
class WPolicyNet(nn.Module):
    def __init__(self, in_dim=9, hidden=64, w_min=0.1, w_max=0.8):
        super().__init__()
        self.w_min = w_min
        self.w_max = w_max
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_stats):
        w_raw = self.net(global_stats)
        return self.w_min + (self.w_max - self.w_min) * torch.sigmoid(w_raw)
```

---

## 6. 完整数据流

```text
Image x
  │
  ├─ VAE encoder → lq_latent [B,256,H/8,W/8]
  │
  ├─ ELIC aux encoder → aux_latent [B,320,H/16,W/16]
  │
  ▼
LatentCodec.g_a / hyperprior / 4-pass checkerboard
  ├─ y_hat
  └─ scales_all [B,320,H/32,W/32]
        │
        ├─ DynamicTimestepModule → T* [B]
        └─ ScaleFeatureAdapter → scale_feat [B,64,H/8,W/8]

SynthesisTransform.g_s → lq_latent_hat [B,320,H/8,W/8]
  │
  ├─ sample = lq_latent_hat[:, :256]
  ├─ UNet(sample, T*) → model_pred
  ├─ _batched_ddpm_step(model_pred, T*, sample) → x0_pred
  │
  └─ AuxDecoder → res1 [B,256,H/8,W/8]

Stage I:
  z_hat = x0_pred + res1

Stage II:
  z_prior = x0_pred
  z_direct = x0_pred + res1
  alpha,beta = CFTPredictor(x0_pred, res1, scale_feat, f(λ))
  z_cft = x0_pred + w * (alpha * x0_pred + beta)
  z_hat = (1-rho) * z_direct + rho * z_cft

VAE decoder → reconstructed image
```

---

## 7. 与 CodeFormer CFT 的对应关系

| CodeFormer CFT | 本方案 CFT |
|---|---|
| HQ decoder feature `F_d` | UNet prior latent `x0_pred` |
| LQ encoder feature `F_e` | `res1 + scale_feat` |
| `α, β = Pθ(concat(F_d, F_e))` | `α, β = Pθ(concat(x0_pred, res1, scale_feat), f(λ))` |
| `F_hat_d = F_d + w(α⊙F_d + β)` | `z_cft = x0_pred + w(α⊙x0_pred + β)` |
| 直接输出 CFT 结果 | 本方案用 `rho` 从 `x0_pred+res1` 过渡到 `z_cft` |
| `w` 控制 quality/fidelity | `w` 控制 prior/Aux-entropy 条件的平衡 |

机制上的关键继承是：

> 条件特征不直接 skip 到输出，而是预测对 prior 特征的可控仿射残差。

---

## 8. 训练策略

## 8.1 Stage I：基础融合预训练

目标：

- 先让 UNet 分支和 Aux 分支学会稳定协作；
- 让 `x0_pred + res1` 成为可靠的基础融合结果；
- 不在这一阶段引入 CFT，避免训练初期额外耦合。

设置：

```text
T* = DynamicTimestepModule(scales_all)
z_hat = x0_pred + res1
loss = mean(D_i / λ_i) + mean(bpp_i)
```

训练完成后保存 Stage I 最优 checkpoint。

## 8.2 Stage II：CFT 过渡融合

目标：

- 从 Stage I checkpoint 出发；
- 让 Aux/codec/scales 通过 `alpha,beta` 修正 UNet prior；
- 用 `rho` 保持训练初期连续，不突然丢掉 `res1`。

设置：

```text
z_prior = x0_pred
z_direct = x0_pred + res1
w_train = 1.0
rho: 0 → 1
z_cft = x0_pred + w_train * (alpha * x0_pred + beta)
z_hat = (1-rho) * z_direct + rho * z_cft
loss = mean(D_i / λ_i) + mean(bpp_i) + perceptual/GAN terms
```

Stage II 初始化与训练细节：

- 从 Stage I 最优 checkpoint 加载；
- `CFTFusionModule.out` 零初始化；
- `rho=0` 时输出等价于 Stage I；
- `rho` 在 Stage II 前 10%～30% steps 从 0 增加到 1；
- 第一版 Stage II 使用固定 `w_train=1.0` 训练，推理扫描 `w`；
- 第二版 Stage II 可在固定 `w` 方案稳定后启用 `WPolicyNet`。

推理时：

```text
rho = 1
w ∈ {0, 0.25, 0.5, 0.75, 1.0}
默认 w = 0.5
```

若启用 `WPolicyNet`：

```text
rho = 1
w = WPolicyNet(global_stats)
```

---

## 9. 正则化与稳定性

## 9.1 仿射幅度正则

防止 `alpha,beta` 过大：

$$
L_{affine} = \lambda_{\alpha}\|\alpha\|_1 + \lambda_{\beta}\|\beta\|_1
$$

首轮可以不加，若出现大幅色偏或 latent 爆炸再启用。

## 9.2 beta 残差幅度限制

可选：

```python
alpha = 0.1 * torch.tanh(alpha_raw)
beta = 0.1 * torch.tanh(beta_raw)
```

这能保证 CFT 是“轻微调制”，符合 CodeFormer 的设计精神。

## 9.3 rho 调度稳定性

如果 Stage II 前期性能抖动，优先调整：

```text
延长 rho_warmup_steps
降低 CFTFusionModule 学习率
缩小 alpha,beta 输出幅度
```

不要一开始就让 `rho=1`。

## 9.4 w 训练策略

第一版推荐：

```text
训练 w=1
推理 w 可调
```

第二版自适应 `w` 训练：

```text
Stage II-a: 固定 w=1 训练 CFTFusionModule
Stage II-b: 加入 WPolicyNet，CFTFusionModule 可继续微调或短暂冻结
```

推荐先冻结 CFTFusionModule 的最后输出层若干步，只训练 `WPolicyNet`，让它先学会选择强度；随后再联合微调。

可选正则：

```text
L_w_center = (mean(w) - w_ref)^2
L_w_smooth = Var(w) 的轻量约束，防止 w 过度抖动
```

首轮可设：

```text
w_ref = 0.5
```

若自适应 `w` 很快塌缩到边界，则收窄范围到：

```text
w ∈ [0.2, 0.7]
```

---

## 10. 诊断系统

原方案记录 `gate_mean/gate_std`。CFT 版本改为记录仿射调制和过渡统计：

```python
diag_buffer = {
    "lambda": [],
    "bpp": [],
    "dist": [],
    "t_star": [],
    "rho": [],
    "w": [],
    "w_pred_mean": [],
    "w_pred_std": [],
    "alpha_mean": [],
    "alpha_abs": [],
    "beta_abs": [],
    "cft_delta_norm": [],
    "cft_to_prior_ratio": [],
    "cft_to_direct_ratio": [],
    "cos_delta_res1": [],
    "res1_to_direct_ratio": [],
}
```

定义：

```python
cft_delta = w * (alpha * x0_pred + beta)
cft_to_prior_ratio = ||cft_delta|| / (||x0_pred|| + eps)
cft_to_direct_ratio = ||cft_delta|| / (||x0_pred + res1|| + eps)
cos_delta_res1 = cosine(x0_pred - sample, res1)
```

期望趋势：

1. `rho=0` 时结果应接近 Stage I；
2. `rho→1` 时性能不应突然崩塌；
3. `cft_delta_norm` 不应无界增大；
4. `alpha/beta` 热力图应与结构复杂区域、压缩不确定区域有一定对应；
5. `w=0` 到 `w=1` 应产生连续、平滑的质量/保真变化。

若启用 `WPolicyNet`，额外检查：

1. `corr(w_pred, bpp)`：通常期望正相关；
2. `corr(w_pred, T*)`：通常期望负相关；
3. `corr(w_pred, scales_mean/scales_p90)`：通常期望负相关；
4. `corr(w_pred, cosine(delta_unet, res1))`：通常期望正相关；
5. `w_pred` 是否塌缩到 `w_min` 或 `w_max`。

---

## 11. 消融实验

## 11.1 主实验

### A0：原动态 T* + 直接加法

```python
z_hat = x0_pred + res1
```

### A1：原动态 T* + scale/map 门控

```python
z_hat = sample + M * (x0_pred - sample) + res1
```

用于对比旧路径 3。

### A2：动态 T* + CFT + rho 过渡

```python
z_direct = x0_pred + res1
z_cft = x0_pred + w * (alpha * x0_pred + beta)
z_hat = (1-rho) * z_direct + rho * z_cft
```

主方案。

### A3：CFT without scales

```python
alpha,beta = Pθ(concat(x0_pred, res1))
```

检验 `scales_all` 是否真的提供额外局部不确定性信息。

### A4：CFT without aux

```python
alpha,beta = Pθ(concat(x0_pred, scale_feat))
```

检验 AuxDecoder `res1` 的价值。

### A5：CFT with sample

```python
alpha,beta = Pθ(concat(x0_pred, sample, res1, scale_feat))
```

检验把 codec latent 当前状态纳入 CFT predictor 是否有益，或是否引入压缩退化泄漏。

### A6：CFT + adaptive `w`

```python
w = WPolicyNet(global_stats)
z_cft = x0_pred + w * (alpha * x0_pred + beta)
z_hat = z_cft  # inference, rho=1
```

用于检验逐图自适应强度是否优于固定 `w`。

### A7：CFT without rho warm-up

```python
rho = 1 from Stage II start
```

用于验证 `rho` 过渡是否必要。

---

## 11.2 w 曲线实验

固定同一 checkpoint，推理时设置：

```text
rho = 1
w = 0, 0.25, 0.5, 0.75, 1.0
```

记录：

- PSNR / MS-SSIM；
- LPIPS / DISTS；
- bpp；
- 视觉伪影；
- 结构一致性。

预期：

```text
w 小：感知质量更强
w 大：保真和结构一致性更强
```

若曲线不连续或指标无规律，说明 CFT 未学到可控信息流。

## 11.3 固定 w 与自适应 w 对照

建议比较：

### B1：固定 `w=0.5`
### B2：固定 `w=1.0`
### B3：推理扫描最优 `w`
### B4：`WPolicyNet` 自适应 `w`

判断标准：

- `WPolicyNet` 是否优于常数 `w=0.5`；
- `WPolicyNet` 是否接近“推理扫描最优 `w`”；
- `WPolicyNet` 输出是否具有可解释相关性，而不是随机波动。

---

## 12. 关键风险

### 风险 1：CFT 学不会替代直接 res1

如果 `rho→1` 后性能下降明显，说明 CFT 没有学会通过 `alpha,beta` 吸收 Aux 条件。

应对：

- 延长 `rho` warm-up；
- 降低 CFT 学习率；
- 增强 `res1` 输入路径；
- 加入 A7 对照验证是不是过渡问题。

### 风险 2：CFT 退化为直接加法

如果 `beta` 学成近似 `res1`，则 CFT 可能只是绕路实现加法。

应对：

- 限制 `beta` 幅度；
- 记录 `corr(beta, res1)`；
- 做 A4/A5 消融。

### 风险 3：CFT 覆盖 UNet prior

若 `alpha,beta` 过大，输出会被 Aux/scales 条件主导，丢失扩散先验质量。

应对：

- `out` 零初始化；
- 对 `alpha,beta` 加 `tanh` 限幅；
- 监控 `cft_to_prior_ratio`。

### 风险 4：w 不可控

如果不同 `w` 输出变化不连续，说明训练时没有形成稳定信息流。

应对：

- 训练 `w=1`，推理调节；
- 必要时训练时随机采样 `w`，并加入同一图不同 `w` 的平滑一致性约束。

### 风险 5：自适应 w 学到伪相关

如果 `WPolicyNet` 只依赖某个简单变量，例如 λ 或 `T*`，则可能没有真正利用多源状态。

应对：

- 做 only-λ、only-SNR、only-entropy-stats 的输入消融；
- 检查 `WPolicyNet` 对 `cos(delta_unet, res1)`、`res1_to_direct_ratio` 的额外贡献；
- 与固定 `w`、推理扫描最优 `w` 对比。

### 风险 6：CFT 参数量过大

拼接 `[x0_pred, res1, scale_feat]` 后通道较多；若做 A5 with sample，通道数会进一步增加。

应对：

- 对 `x0_pred/res1` 先各自用 `1x1 Conv(256→64)` 压缩；
- 再 concat 为 `64×2+64=192` 通道；
- 首轮可先用压缩版本降低显存。

---

## 13. 推荐实验顺序

1. Stage I 先训练原始直接相加版本：`z_hat = x0_pred + res1`；
2. 保存 Stage I 最优 checkpoint；
3. 实现 `ScaleFeatureAdapter`，只产生 `scale_feat`，不输出门控图；
4. 实现单尺度 `CFTFusionModule`，输入 `x0_pred/res1/scale_feat`，`out` 零初始化；
5. Stage II 从 Stage I checkpoint 加载，启用 CFT，设置 `rho: 0→1`；
6. 跑 A0 与 A2，对比 CFT + rho 是否稳定提升；
7. 跑 A7，验证没有 rho warm-up 是否会掉性能；
8. 扫描 `w∈{0,0.25,0.5,0.75,1}`，确认 quality/fidelity 曲线；
9. 跑 A3/A4/A5，判断 scales、aux 的贡献，以及 sample 是否值得加入；
10. 若固定 `w` 的 CFT 有效，再加入 A6：`WPolicyNet` 自适应 `w`；
11. 做 B1-B4 对照，判断自适应 `w` 是否优于常数 `w`；
12. 若两者都有效，再考虑多尺度 CFT；
13. 若 CFT 无效，回退到旧的 scale/map 门控，但保留 CFT 结果作为负实验。

---

## 14. 最终贡献表述边界

建议不要把本方案表述为“提出新的 CFT”，而应表述为：

> 受 CodeFormer CFT 启发，本方案在 Stage I 先使用直接相加建立稳定的 `x0_pred + res1` 基础模型，并在 Stage II 将原有门控缩放替换为基于 entropy uncertainty 条件的 CFT 式仿射残差调制。与直接把 Aux 残差混入主特征不同，本方案以 UNet 一步去噪结果 `x0_pred` 作为 prior 主体，使 Aux/scales 条件仅通过 `alpha,beta` 对其进行轻量修正，并通过 `rho` 实现从直接相加到 CFT 融合的平滑过渡，通过 `w` 提供质量-保真连续调节。

这一区分很重要：

- CodeFormer 的 CFT 用于人脸 restoration 的 codebook decoder；
- 本方案的 CFT-style fusion 用于压缩扩散解码中的 UNet latent 与 Aux/entropy 条件融合；
- 继承的是“可控仿射调制”机制，不主张 CFT 概念本身原创。
