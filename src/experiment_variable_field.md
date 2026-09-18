自注意力图的感受野散布 (Self-Attention Receptive Field Spread)
**物理意义：** 证明低码率下模型在“四处张望”寻找上下文线索来脑补，高码率下模型在“专注局部”精确还原。

*   **提取方法：**
    提取 U-Net Spatial Transformer 层中的 Self-Attention Map。选定图像中的某一个 Query 点（例如一片草叶或一根发丝）。
*   **如何可视化：**
    将该 Query 点对整张图的 Attention Weight 画出来。
*   **你期望看到的现象：**
    *   **极低码率 ($\lambda_{low}$)**：Attention 权重分布非常**广泛且分散（Diffused/Global）**。模型因为给定的特征太模糊，只能参考周围所有的草地来“猜”这片草长什么样。
    *   **高码率 ($\lambda_{high}$)**：Attention 权重高度**集中（Highly Localized/Sharp）**在 Query 点及其极小邻域。模型给定的特征非常清晰，不需要看别的地方，直接“照抄”即可。
    *   **故事包装**：*"The attention maps reveal a paradigm shift in context gathering: low-bitrate conditions force the model to aggregate global contextual clues for plausible synthesis, whereas high-bitrate conditions promote localized, deterministic reconstruction."*
