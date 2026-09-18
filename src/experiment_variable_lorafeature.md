LoRA 残差能量图 (The LoRA Residual Energy Map)
**物理意义：** 证明低码率下 LoRA 正在“发力”无中生有，而高码率下 LoRA 在“摸鱼”（只做微调）。

*   **提取方法：**
    在 U-Net 的某一个关键层（建议选择 Decoder 阶段靠近输出的高分辨率层），提取基础 U-Net 的特征 $F_{base}$，以及 $\lambda$-LoRA 分支输出的增量特征 $\Delta F_{LoRA}$。
*   **如何可视化：**
    计算 $\Delta F_{LoRA}$ 在通道维度的平均绝对值（或 L2 范数），得到一张 2D 的能量热力图（Heatmap），使用 `jet` 或 `magma` 伪彩色映射。
*   **你期望看到的现象（讲故事的素材）：**
    *   **注入极低码率 ($\lambda_{low}$)**：热力图呈现**大面积的高亮红色/黄色**。特别是在草地、毛发、砖墙等纹理区域，LoRA 的激活值极高。
    *   **故事包装**：*"At extreme low bitrates, the U-Net perceives the high uncertainty in the latent space. Consequently, the $\lambda$-conditioned LoRA acts as an aggressive 'texture hallucinator', exhibiting high-energy activations to inject generative priors."* (在极低码率下，U-Net 感知到隐空间的高度不确定性。因此，$\lambda$ 条件化的 LoRA 充当了激进的“纹理幻觉器”，表现出高能量激活以注入生成先验。)
    *   **注入高码率 ($\lambda_{high}$)**：热力图呈现**整体暗蓝色**，只有在图像边缘处有微弱的激活。
    *   **故事包装**：*"Conversely, at high bitrates, the LoRA suppresses its generative aggressiveness, acting merely as a conservative high-frequency refiner to preserve exact pixel fidelity."* (相反，在高码率下，LoRA 抑制了其生成激进性，仅仅作为保守的高频细化器来保留精确的像素保真度。)
