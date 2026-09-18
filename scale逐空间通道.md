# 路径 3：SNR 全局时间步 + 局部不确定性修复分配实验方案

> 目标：引入“局部不确定性驱动的修复分配”机制，使系统同时具备图像级去噪强度调节和空间/通道级修复强度调节。

---

## 1. 研究目标

当前总方案中，`DynamicTimestepModule` 的基本逻辑是：

```text
scales_all → 压缩 SNR → DDPM SNR schedule → T* → 一步去噪
```

这一路径把压缩 latent 的整体退化程度映射为扩散过程中的某个“伪时间步”，再利用该时间步完成单步去噪。该分支仍然保留，作为全局去噪工作点估计。

本路径的新增机制是增加一条局部控制分支：

```text
scales_all
  ├─ 全局分支：压缩 SNR → DDPM SNR schedule → T*
  └─ 局部分支：局部不确定性图 U → 修复门控图 M → 局部控制 UNet 残差注入强度
```

研究目标变为：

1. 利用熵模型已有的 `scales` 构造空间或通道级局部不确定性估计；
2. 在保留全局 `T*` 一步扩散去噪框架下，自适应分配不同区域的生成式修复强度；
3. 让可靠区域尽量保真保留，不可靠区域更多依赖扩散先验补偿；
4. 将原本“单一全局时间步控制”扩展为“全局时间步 + 局部门控”的双层控制。

---

## 3. 方法设计

## 3.1 整体流程

保持现有大部分主干不变：

```text
Image
  → VAE encoder (256ch latent)
  → AnalysisTransform + Hyperprior + 4-pass entropy model
  → y_hat, scales_all
  → DynamicTimestepModule(scales_all) 得到全局 T*
  → SynthesisTransform 得到 lq_latent_hat
  → UNet 在 T* 上一步去噪
  → UncertaintyGateModule(scales_all) 生成修复门控图 M
  → 用 M 调制 UNet 残差
  → 与 AuxDecoder 残差 res1 融合
  → VAE decoder 重建图像
```

核心新增点只有一个：

- 保留 `DynamicTimestepModule`
- 在其旁路新增 `UncertaintyGateModule`
- 将 UNet 一步去噪结果从直接相加改为门控残差融合

---

## 3.2 全局时间步策略：保留 SNR → T*

本路径保留原方案的 `DynamicTimestepModule`：

```text
scales_all → scales_mean² / (1/12) → snr_compress → DDPM SNR schedule → T*
```

`T*` 仍然是 per-image 的全局时间步，范围保持：

```text
T* ∈ [800, 999]
```

它负责回答图像级问题：

> 当前压缩 latent 整体应在多强的扩散噪声工作点上被恢复？

新增的 `UncertaintyGateModule` 则负责回答局部问题：

> 在已经确定全局 `T*` 后，哪些位置应该更多接受 UNet 的修复残差？

因此最终机制是双层控制：

```text
T*：图像级去噪强度
M ：空间/通道级修复分配
```

这样可以最大限度保留现有实现，同时把新增贡献放在 `T*` 之后的局部残差调制上。

---

## 3.3 `UncertaintyGateModule`

### 3.3.1 输入输出定义

输入：

```text
scales_all: [B, 320, H/32, W/32]
```

输出：

```text
M: [B, 1, H/8, W/8]      空间门控图
或
M: [B, 256, H/8, W/8]    通道-空间门控图
```

门控值范围：

```text
M ∈ [0, 1]
```

其中：

- `M = 0` 表示几乎完全保留 codec latent；
- `M = 1` 表示完全接受 UNet 预测的修复残差；
- 中间值表示部分修复。

### 3.3.2 最小实现版本：空间门控图

先做一个尽量稳的最小版本：

1. 对 `scales_all` 在通道维求平均：

$$
U = \text{mean}_{c}(scales\_all) \in \mathbb{R}^{B \times 1 \times H/32 \times W/32}
$$

2. 归一化到每张图内部的稳定范围：

$$
\tilde{U} = \frac{U - \mu(U)}{\sigma(U) + \epsilon}
$$

或采用分位数归一化：

$$
\tilde{U} = \text{clip}\left(\frac{U - q_{0.1}}{q_{0.9} - q_{0.1} + \epsilon}, 0, 1\right)
$$

3. 上采样到 VAE latent 分辨率：

```text
[B, 1, H/32, W/32] → [B, 1, H/8, W/8]
```

4. 用 `sigmoid` 或裁剪得到门控图：

$$
M = \sigma(\alpha \tilde{U} + \beta)
$$

其中 `α, β` 可以先固定，也可以设为可学习标量。

### 3.3.3 增强版本：通道-空间门控图

如果空间门控验证有效，再进一步做通道细化：

```text
scales_all [B, 320, H/32, W/32]
  → 1x1 Conv(320→256)
  → Upsample ×4
  → 3x3 Conv(256→256)
  → Sigmoid
  → M [B, 256, H/8, W/8]
```

这样每个 latent 通道都能有不同的修复强度，更适合 256 通道 VAE 设定。

---

## 3.4 残差融合机制

### 3.4.1 基础融合式

在 `T*` 上完成一步去噪后，先得到：

```python
delta_unet = x0_pred - sample
```

然后用门控图调制 UNet 修复强度：

```python
z_hat = sample + M * delta_unet + res1
```

其中：

- `sample`：压缩 latent（送入一步去噪前的特征）
- `x0_pred`：UNet 一步去噪后预测的干净 latent
- `delta_unet`：UNet 对 latent 的修复建议
- `res1`：AuxDecoder 结构残差

这个公式的含义是：

- 低不确定区域：`M` 小，保留 `sample` 为主；
- 高不确定区域：`M` 大，更相信 UNet 的修复；
- `res1` 继续承担结构补偿作用。

### 3.4.2 更稳的双门控版本

若后期发现 `res1` 与 `delta_unet` 在某些区域存在冲突，可以改为双门控：

```python
z_hat = sample + M_unet * delta_unet + M_aux * res1
```

其中：

- `M_unet` 控制生成式修复；
- `M_aux` 控制结构残差注入。

但首轮实验建议先只做 `M * delta_unet`，减少变量数。

---

## 3.5 模块伪代码

```python
class UncertaintyGateModule(nn.Module):
    def __init__(self, out_channels=1, mode="spatial", learnable_affine=True):
        super().__init__()
        self.mode = mode
        if mode == "channel_spatial":
            self.proj = nn.Conv2d(320, 256, kernel_size=1)
            self.refine = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        else:
            self.proj = None
            self.refine = None

        if learnable_affine:
            self.alpha = nn.Parameter(torch.tensor(1.0))
            self.beta = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("alpha", torch.tensor(1.0))
            self.register_buffer("beta", torch.tensor(0.0))

    def _normalize_map(self, u, eps=1e-6):
        mean = u.mean(dim=[2, 3], keepdim=True)
        std = u.std(dim=[2, 3], keepdim=True)
        return (u - mean) / (std + eps)

    def forward(self, scales_all, target_hw):
        if self.mode == "spatial":
            u = scales_all.mean(dim=1, keepdim=True)              # [B,1,H/32,W/32]
            u = self._normalize_map(u)
            u = F.interpolate(u, size=target_hw, mode="bilinear", align_corners=False)
            m = torch.sigmoid(self.alpha * u + self.beta)         # [B,1,H/8,W/8]
            return m

        feat = self.proj(scales_all)                              # [B,256,H/32,W/32]
        feat = F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=False)
        feat = self.refine(feat)
        feat = self._normalize_map(feat)
        m = torch.sigmoid(self.alpha * feat + self.beta)          # [B,256,H/8,W/8]
        return m
```

在主干中的使用：

```python
sample = lq_latent_hat[:, :256]
T_star = dynamic_timestep_module(scales_all)
model_pred = unet_out
x0_pred = batched_ddpm_step(model_pred, T_star.long(), sample)
M = gate_module(scales_all, target_hw=sample.shape[-2:])
delta = x0_pred - sample
z_hat = sample + M * delta + res1
```

---

## 4. 与当前总方案的关系

## 4.1 保留的部分

以下设计全部保留：

1. 256 通道 VAE 扩展；
2. `λ-FiLM` 四注入点；
3. ELIC 风格的 `g_a / g_s / hyperprior / 4-pass checkerboard` 熵模型；
4. `mean(D_i / λ_i) + mean(bpp_i)` 的训练目标；
5. 单步 SD-Turbo 去噪；
6. `DynamicTimestepModule` 的 `SNR → T*` 全局时间步估计；
7. `AuxDecoder` 输出 256 通道 `res1`。

## 4.2 修改的部分

只改这几项：

1. 新增 `UncertaintyGateModule`，与 `DynamicTimestepModule` 并行读取 `scales_all`；
2. 保留 `T*`，UNet 仍在 per-image `T*` 上一步去噪；
3. 将去噪后的 latent 融合方式改为门控残差融合；
4. 诊断系统保留 `t_star`，并增加 `gate_mean / gate_std / gate_sparsity`。

---

## 5. 训练方案

## 5.1 训练阶段设计

建议仍采用两阶段训练，但阶段目标略调整。

### Stage I：稳定重建阶段

目标：

- 让 256ch VAE、Latent Codec、FiLM 条件化先收敛；
- 让动态 `T*` 一步去噪与 `res1` skip 形成稳定协作；
- 让门控模块先学会基本的“该不该修复”。

设置：

```text
T* = DynamicTimestepModule(scales_all)
λ ~ LogUniform(lambda_min, lambda_sample_max)
loss = mean(D_i / λ_i) + mean(bpp_i)
```

训练策略：

- 起始时将 `alpha=0, beta=0` 或直接让 `M≈0.5`，避免训练初期门控过饱和；
- 前若干步可冻结 `gate_module`，待主干稳定后再解冻；
- 或给门控模块更小学习率。

### Stage II：感知增强阶段

目标：

- 让门控模块学习更细粒度的纹理修复分配；
- 在 GAN/perceptual loss 下平衡结构保真与真实感。

设置：

```text
T* = DynamicTimestepModule(scales_all)
λ ~ LogUniform(lambda_min, lambda_sample_max)
loss = mean(D_i / λ_i) + mean(bpp_i) + GAN/perceptual terms
```

此时门控图应自然呈现：

- 平滑区域抑制 UNet 大幅修改；
- 纹理、边缘、复杂结构区域允许更强修复。

---

## 5.2 正则化建议

为了防止门控图退化到全 0 或全 1，可加入轻量正则。

### 方案 A：均值约束

控制平均修复强度：

$$
L_{gate-mean} = (\text{mean}(M) - \rho)^2
$$

其中 `ρ` 可设为：

- 固定常数，如 `0.4`；
- 或随 λ 变化，使高压缩时允许更大的平均门控值。

### 方案 B：总变差正则

鼓励门控图空间平滑：

$$
L_{TV}(M)
$$

避免门控图呈现棋盘噪声或高频抖动。

### 方案 C：熵引导单调性

鼓励高 `scales` 区域的 `M` 更大：

$$
L_{rank}
$$

可用简单排序损失或相关性约束实现。首轮实验可先不加。

建议首轮仅使用：

```text
L_total = L_rd + λ_tv * L_TV(M)
```

尽量少引入新超参数。

---

## 6. 诊断系统

原来的 `diag_buffer` 以 `t_star` 为核心，这里改为门控统计。
原来的 `t_star` 不删除，而是作为全局时间步诊断继续保留；新增门控统计用于分析局部修复分配。

## 6.1 新的统计项

```python
diag_buffer = {
    "lambda": [],
    "bpp": [],
    "dist": [],
    "lambda_dist": [],
    "t_star": [],
    "gate_mean": [],
    "gate_std": [],
    "gate_high_ratio": [],
}
```

其中：

- `gate_mean`：每张图的平均修复强度；
- `gate_std`：门控图离散程度；
- `gate_high_ratio`：`M > 0.7` 的像素比例。

## 6.2 期望相关性

建议记录以下相关性：

1. `corr(logλ, gate_mean)`：期望正相关；
2. `corr(bpp, gate_mean)`：期望负相关；
3. `corr(dist, gate_mean)`：通常正相关；
4. `corr(logλ, gate_high_ratio)`：期望正相关；
5. `corr(t_star, gate_mean)`：用于观察全局去噪强度和局部修复强度是否协同。

## 6.3 可视化检查

训练和验证时定期保存：

1. 输入图像；
2. 重建图像；
3. `M` 的热力图；
4. `|x0_pred - sample|` 的残差幅度图。

重点看：

- 门控高值区是否集中在纹理、边缘、复杂语义区域；
- 平坦区域是否被抑制；
- 门控是否出现无意义满图高亮。

---

## 7. 消融实验设计

## 7.1 主消融

### A0：原始动态时间步 baseline

```text
T* = DynamicTimestepModule(scales_all)
无门控
z_hat = x0_pred + res1
```

作用：作为保留原 `SNR → T*` 机制但不加局部门控的直接对照基线。

### A1：空间门控

```text
T* = DynamicTimestepModule(scales_all)
M: [B,1,H/8,W/8]
z_hat = sample + M * (x0_pred - sample) + res1
```

这是主实验版本。

### A2：通道-空间门控

```text
T* = DynamicTimestepModule(scales_all)
M: [B,256,H/8,W/8]
z_hat = sample + M * (x0_pred - sample) + res1
```

作用：验证 256 通道 latent 下更细粒度控制是否有效。

### A3：双门控

```text
z_hat = sample + M_unet * delta_unet + M_aux * res1
```

作用：验证结构残差与生成残差是否值得分开调制。

---

## 7.2 替代对照

### B1：全局标量门控

```text
M_scalar ∈ [0,1]
每张图只用一个门控值
```

如果这个版本明显弱于空间门控，说明局部机制确实有价值。

### B2：随机门控

用随机图替代 `M`，检查性能是否显著下降。若下降明显，说明 `scales` 提供了有效信息，而不是任何噪声图都可以。

### B3：仅用 λ 预测门控

不用 `scales_all`，而是由 `f(λ)` 直接生成一张门控图。若性能不如 `scales` 驱动版本，说明真实图像内容相关的不确定性是有用的。

---

## 7.3 时间步与门控交互对照

为了确认收益来自“全局 `T*` + 局部 `M`”的协同，而不是单一分支，建议做以下对照：

### C1：固定 `t=999` + 空间门控

检验门控在没有动态 `T*` 时是否仍有效。

### C2：动态 `T*` + 无门控

对应原方案，用于衡量新增门控的边际收益。

### C3：动态 `T*` + 空间门控

主实验版本，用于验证全局和局部控制的协同。

### C4：动态 `T*` + 随机门控

检验 `scales` 驱动的局部门控是否优于无意义空间图。

---

## 8. 评估指标

建议保留当前方案已有评价体系：

1. `PSNR`
2. `MS-SSIM`
3. `LPIPS`
4. `DISTS`
5. `FID` 或可替代感知指标
6. `CLIPIQA / MUSIQ`（若你当前实验链路已有）

额外增加两个局部机制分析指标：

1. `gate-structure overlap`：门控高值区域与 Sobel 边缘区域的重叠程度；
2. `gate-residual corr`：门控值与 `|x0_pred - sample|` 的相关性。

如果这两项呈合理趋势，会很有助于支撑机制解释。

---

## 9. 预期结果

若路径 3 成立，预期应观察到：

1. 相比无门控 baseline，感知质量指标提升更明显；
2. 高频纹理和边缘区域细节更好；
3. 平坦区域的过度 hallucination 减少；
4. `M` 热力图与复杂区域、边缘区域有较好一致性；
5. 高 λ / 低 bpp 时 `gate_mean` 增大，说明模型学会在高压缩场景中更积极调用扩散先验。

---

## 10. 风险与失败模式

### 风险 1：门控塌缩

表现：

- `M` 接近全 0：模型退化为几乎不用 UNet 修复；
- `M` 接近全 1：模型退化为完全信任 UNet，失去局部分配意义。

应对：

- 加入轻量均值约束；
- 初始时设置 `M≈0.5`；
- 给 `gate_module` 更小学习率。

### 风险 2：`scales` 与“需要修复”不完全一致

`scales` 反映的是熵模型不确定性，不一定总是等于视觉退化程度。

应对：

- 从最小空间门控开始验证；
- 若效果有限，再加入通道投影和小型 refine conv；
- 必要时融合 `|y_hat - means| / scales` 这类标准化残差作为补充输入。

### 风险 3：与 `res1` 冲突

`delta_unet` 和 `res1` 可能在某些区域重复修复或相互抵消。

应对：

- 先观察 `res1` 与 `delta_unet` 的幅度分布；
- 若冲突明显，再启用双门控版本。

### 风险 4：收益不足以覆盖复杂度

如果空间门控只带来极小收益，则说明这条机制可能解释性强但收益有限。

应对：

- 先做最小版本，快速判断；
- 若收益很小，可将其降级为分析工具或辅助模块，而非主创新。

---

## 12. 推荐实验顺序

建议按最小成本逐步推进：

1. `A0`: 保留动态 `T*`，不加门控，复现实验基线；
2. `A1`: 动态 `T*` + 空间门控 `M [B,1,H/8,W/8]`；
3. 做 `T*`、门控热力图和相关性诊断；
4. 做 `C1`: 固定 `t=999` + 空间门控，判断门控和 `T*` 是否互补；
5. 若有效，再做 `A2` 通道-空间门控；
6. 若发现 `res1` 冲突明显，再做 `A3` 双门控；
7. 最后再考虑加入门控正则和更复杂输入。

这样可以最快判断：

> 在保留 `DynamicTimestepModule` 的条件下，“局部不确定性修复分配”是否能作为额外机制带来稳定收益。
