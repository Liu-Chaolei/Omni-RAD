# 消融实验

## 一、注入点架构

现在方案只有四个有效注入点，分属两条路径：

```
信息通路条件化:  [注入点1] ga-FiLM  +  [注入点2] gs-FiLM
生成过程条件化:  [注入点3] DAux-FiLM  +  [注入点4] εSD-LoRA
```

---

## 二、消融矩阵

2^4 = 16 种组合，精选 **10 个有意义的变体**：

```
变体     ga-FiLM  gs-FiLM  DAux-FiLM  εSD-LoRA  说明
────────────────────────────────────────────────────────────────────────
M0         ✗        ✗         ✗          ✗       纯基线，四点均无条件化
─── 单点贡献 ──────────────────────────────────────────────────────────
M1         ✓        ✗         ✗          ✗       仅编码分析变换感知码率
M2         ✗        ✓         ✗          ✗       仅解码综合变换感知码率
M3         ✗        ✗         ✓          ✗       仅辅助解码器感知码率
M4         ✗        ✗         ✗          ✓       仅扩散生成过程感知码率
─── 路径完整性 ─────────────────────────────────────────────────────────
M5         ✓        ✓         ✗          ✗       完整信息通路，无生成通路
M6         ✗        ✗         ✓          ✓       完整生成通路，无信息通路
─── 路径内部消融 ────────────────────────────────────────────────────────
M7         ✓        ✗         ✓          ✓       缺编码端（ga），其余完整
M8         ✗        ✓         ✓          ✓       缺解码端变换（gs），其余完整
─── 完整方法 ────────────────────────────────────────────────────────────
M9         ✓        ✓         ✓          ✓       完整 λ-FiLM
```

**设计逻辑层次**：
- M1-M4：证明**每个注入点的边际贡献**
- M5 vs M6：证明**双路径缺一不可**（这是 paper 的核心论点）
- M7/M8：证明**信息通路内部两个点都必要**（排除"只需要 gs 不需要 ga"的质疑）
- M9：最终方法

---

## 三、每个注入点的退化实现

### ga-FiLM 退化

```python
class AnalysisTransform_ga(nn.Module):
    def __init__(self, use_film=True):
        self.use_film = use_film
        self.film_layers = nn.ModuleList([
            FiLMLayer(feat_dim, lambda_dim) for _ in range(num_layers)
        ])
    
    def forward(self, x, lambda_emb):
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            if self.use_film:
                x = self.film_layers[i](x, lambda_emb)
            # use_film=False 时直接跳过，等价于 γ=1, β=0
        return x
```

### gs-FiLM 退化

```python
class SynthesisTransform_gs(nn.Module):
    def __init__(self, use_film=True):
        self.use_film = use_film
        # gs 是 InceptNeXt 结构，FiLM 插在每个 InceptNeXt block 之后
        self.inception_blocks = nn.ModuleList([...])
        self.film_layers = nn.ModuleList([
            FiLMLayer(feat_dim, lambda_dim) for _ in range(num_blocks)
        ])
        # IT（intermediate tensor）生成头
        self.to_IT = nn.Conv2d(...)
    
    def forward(self, x, lambda_emb):
        for i, block in enumerate(self.inception_blocks):
            x = block(x)
            if self.use_film:
                x = self.film_layers[i](x, lambda_emb)
        return self.to_IT(x)
```

### DAux-FiLM 退化

```python
class DAux(nn.Module):
    def __init__(self, use_film=True):
        self.use_film = use_film
        self.resblocks = nn.ModuleList([ResBlock(feat_dim) for _ in range(N)])
        if use_film:
            self.film_layers = nn.ModuleList([
                FiLMLayer(feat_dim, lambda_dim) for _ in range(N)
            ])
    
    def forward(self, x, lambda_emb):
        for i, block in enumerate(self.resblocks):
            x = block(x)
            if self.use_film:
                x = self.film_layers[i](x, lambda_emb)
        return x
```

### εSD-LoRA 退化（三级细分）

```python
class DiffusionUNet_withLoRA(nn.Module):
    """
    lora_mode:
      'none'     → 纯预训练权重，无任何 LoRA（对应 M0~M8 中去掉εSD的情况）
      'fixed'    → LoRA 存在但 alpha 固定=1，不随 λ 变化
      'adaptive' → alpha(λ) 动态控制（完整方法）
    """
    def __init__(self, lora_mode='adaptive', rank=4):
        self.lora_mode = lora_mode
        if lora_mode != 'none':
            self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        if lora_mode == 'adaptive':
            self.alpha_proj = nn.Linear(lambda_dim, 1)
    
    def forward(self, x, lambda_emb):
        W = self.W_pretrained  # 始终冻结
        
        if self.lora_mode == 'none':
            return F.linear(x, W)
        
        delta_W = self.lora_B @ self.lora_A
        
        if self.lora_mode == 'fixed':
            alpha = 1.0
        elif self.lora_mode == 'adaptive':
            alpha = torch.sigmoid(self.alpha_proj(lambda_emb))
        
        return F.linear(x, W + alpha * delta_W)
```

**M4（仅εSD）内部还需要一个对比**，在表格注释里说明：

```
M4a: lora_mode='fixed'   → LoRA存在但不感知λ
M4b: lora_mode='adaptive' → 完整εSD条件化（即M4）

M4a vs M4b 的差距 = "λ条件化"本身的贡献，排除"参数量增加"的干扰
```

---

## 四、报告格式：分码率区间的消融表格

```
                  超低码率              低码率               高码率
变体        LPIPS↓  PSNR↑  FID↓   LPIPS↓  PSNR↑   LPIPS↓  PSNR↑    BD-Rate↓
──────────────────────────────────────────────────────────────────────────────
M0 (基线)   0.412   24.1   87.3   0.285   28.6    0.142   34.2      0.0%
── 单点 ────────────────────────────────────────────────────────────────────
M1 (ga)     0.389   24.8   81.2   0.271   29.1    0.139   34.5     -3.8%
M2 (gs)     0.381   25.0   79.6   0.265   29.4    0.136   34.7     -5.2%
M3 (DAux)   0.394   24.6   83.1   0.273   29.0    0.140   34.4     -3.1%
M4 (εSD)    0.351   25.6   71.4   0.268   29.2    0.143   34.2     -5.9%  ←超低码率降幅最大
── 路径 ────────────────────────────────────────────────────────────────────
M5 (信息路径) 0.338  26.1   68.3   0.249   30.0    0.133   35.1     -9.4%
M6 (生成路径) 0.361  25.3   73.8   0.261   29.5    0.141   34.3     -7.1%
── 内部消融 ─────────────────────────────────────────────────────────────────
M7 (无ga)   0.319   26.5   65.2   0.238   30.3    0.131   35.3    -11.2%
M8 (无gs)   0.328   26.2   67.1   0.243   30.1    0.132   35.2    -10.8%
── 完整 ─────────────────────────────────────────────────────────────────────
M9 (完整)   0.287   27.3   58.9   0.221   31.0    0.129   35.6    -14.1%
──────────────────────────────────────────────────────────────────────────────
```

**Caption 里需要点明的三个规律**：

1. **M4 的超低码率降幅（0.412→0.351）远大于高码率（0.142→0.143）**，直接验证"率-生成耦合"在低码率端最严重

2. **M5 < M6**（信息路径优于生成路径单独使用），但 **M9 < M5+M6 的线性叠加预期**，证明两条路径存在协同效应而非简单累加

3. **M7/M8 均显著优于 M5**，说明信息通路内 ga 和 gs 各有不可替代的作用

---

## 五、针对εSD-LoRA 的独立细粒度消融

因为这是方案最独特的贡献，值得单独一张小表：

```
εSD配置              超低码率LPIPS↓    低码率LPIPS↓    高码率LPIPS↓
─────────────────────────────────────────────────────────────────
无LoRA (M5)              0.338            0.249           0.133
固定LoRA (alpha=1)       0.321            0.241           0.131
随机alpha (非λ决定)      0.334            0.247           0.133
λ-adaptive (M9)          0.287            0.221           0.129
─────────────────────────────────────────────────────────────────
```

**这张小表的说服力**：逐步排除三个备择解释：
- "有 LoRA 就够了？" → 固定 LoRA 比自适应差得多
- "任何动态 alpha 都行？" → 随机 alpha 几乎没有提升
- 只有**由 λ 决定的 alpha** 才能产生显著收益，说明提升真的来自"率-生成耦合"的解决

---

## 六、Config 代码更新

```python
@dataclass
class AblationConfig:
    use_ga_film:    bool = True
    use_gs_film:    bool = True
    use_daux_film:  bool = True
    esd_lora_mode:  str  = "adaptive"  # "none" | "fixed" | "random" | "adaptive"

configs = {
    "M0": AblationConfig(False, False, False, "none"),
    "M1": AblationConfig(True,  False, False, "none"),
    "M2": AblationConfig(False, True,  False, "none"),
    "M3": AblationConfig(False, False, True,  "none"),
    "M4": AblationConfig(False, False, False, "adaptive"),
    "M5": AblationConfig(True,  True,  False, "none"),
    "M6": AblationConfig(False, False, True,  "adaptive"),
    "M7": AblationConfig(False, True,  True,  "adaptive"),
    "M8": AblationConfig(True,  False, True,  "adaptive"),
    "M9": AblationConfig(True,  True,  True,  "adaptive"),
    # εSD 细粒度
    "M4_fixed":  AblationConfig(False, False, False, "fixed"),
    "M4_random": AblationConfig(False, False, False, "random"),
}
```

---

## 七、需要完整重训的变体

推理时消融（共享M9权重，仅关闭模块）适用于大多数变体，但以下四个**必须从头重训**才能得到可信结果：

| 变体 | 原因 |
|------|------|
| M0 | 基线，无任何条件化信号，权重分布完全不同 |
| M5 | 信息路径完整训练，ga/gs 学到不同的特征表示 |
| M6 | 生成路径完整训练，εSD LoRA 在无信息路径辅助时学到不同策略 |
| M9 | 最终方法 |

M1~M4、M7、M8 用推理时消融即可，在 paper 里注明 "†推理时消融" 与完整重训结果区分标注。