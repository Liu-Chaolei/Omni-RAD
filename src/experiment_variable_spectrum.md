特征的频谱差异 (Frequency Spectrum Analysis of Feature Maps)
**物理意义：** 证明模型在低码率下自主生成了高频纹理。

*   **提取方法：**
    对 U-Net 某一层的最终输出特征 $F_{out}$ 进行 2D 快速傅里叶变换（FFT），获取其频谱幅度图（Magnitude Spectrum）。
*   **如何可视化：**
    展示频谱中心的低频到边缘的高频热力图。或者画一根一维的频率-能量衰减曲线。
*   **你期望看到的现象：**
    *   **极低码率 ($\lambda_{low}$)** 的特征图在**高频区域（图的四周）拥有更多的能量**。
    *   **高码率 ($\lambda_{high}$)** 的特征高频能量相对较少（因为隐变量本身包含了结构，U-Net 不需要自己凭空造高频）。
    *   **故事包装**：*"Frequency analysis confirms that under $\lambda_{low}$, the U-Net autonomously synthesizes missing high-frequency details (textures), completely changing its operational manifold compared to the $\lambda_{high}$ state."*
