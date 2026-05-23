# Gray Extractor v4.0

野外剖面照片灰度序列提取工具。用户在照片上用红色矩形标记提取区域，工具自动识别红框并提取多通道颜色序列。

## 功能

- **多通道提取**：Gray / Redness / Greenness / Blueness / CIE L* / CIE a*
- **光照梯度校正**：二阶多项式去趋势，消除野外光照不均
- **像素→深度校准**：两个已知钻孔深度点校准
- **多红框拼接**：支持多个分离的红框，自动按地层顺序拼接
- **PyQt6 GUI**：拖入照片即可，实时预览

## 快速开始

```bash
conda activate qt6
python gray_extractor.py
```

或直接双击 `run.bat`。

## 用法

1. 在照片上用红色矩形标记要提取的剖面区域（画图/Photoshop 均可）
2. 拖入照片或 Ctrl+O 打开
3. 红框自动检测，6 通道曲线自动绘制
4. Ctrl+D 深度校准 | Ctrl+I 光照校正 | Ctrl+S 导出 CSV

## 依赖

- Python 3.11+
- PyQt6
- OpenCV (cv2)
- NumPy
- matplotlib 3.x
