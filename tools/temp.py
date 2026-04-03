import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ===================== 参数设置 =====================
nx, ny = 1500, 1500          # 全网格尺寸
border = 10                   # 边界禁止流体区宽度（格）
phi_target = 0.05            # 目标孔隙度（仅对内部区域）

# 椭圆形状控制（sigma ≈ 椭圆半径的一半，实际流体斑块大小约为 2~3*sigma）
sigma_x = 2                # 长轴方向标准差（控制椭圆长度）
sigma_y = 1.2             # 短轴方向标准差（控制椭圆宽度）
# 可调整 sigma_x 和 sigma_y 的比值来改变椭圆形状
# 例如：sigma_x=5, sigma_y=2.5  -> 长轴约 10~15 格，短轴约 5~8 格

seed = 12345                 # 随机种子

# ===================== 生成内部区域 =====================
# 内部区域尺寸（去掉边界）
inner_nx = nx - 2 * border
inner_ny = ny - 2 * border

np.random.seed(seed)
# 在内部区域生成白噪声
noise = np.random.randn(inner_nx, inner_ny)

# 各向异性高斯平滑（产生椭圆状结构）
smoothed = gaussian_filter(noise, sigma=(sigma_x, sigma_y), mode='reflect')

# 确定阈值，使得内部区域孔隙度 = phi_target
threshold = np.percentile(smoothed, phi_target * 100)
inner_binary = (smoothed <= threshold).astype(np.uint8)   # 流体=1, 固体=0

# 验证内部区域实际孔隙度
actual_porosity_inner = inner_binary.mean()
print(f"内部区域目标孔隙度: {phi_target:.2%}")
print(f"内部区域实际孔隙度: {actual_porosity_inner:.4f}")

# ===================== 扩展到全网格（边界填充0） =====================
full_binary = np.zeros((nx, ny), dtype=np.uint8)
full_binary[border:nx-border, border:ny-border] = inner_binary

# 全图孔隙度（边界全固体，会略低于目标）
full_porosity = full_binary.mean()
print(f"全图孔隙度（含边界固体）: {full_porosity:.4f}")

# ===================== 可视化（可选，采样显示） =====================
# 由于图很大，可以显示局部或缩小显示
plt.figure(figsize=(10, 10))
# 为了快速显示，可以取部分区域（例如500x500中心区）
# 或者直接显示全图（如果内存足够）
plt.imshow(full_binary, cmap='gray', interpolation='none')
plt.title(f'Fluid (white) / Solid (black)\n'
          f'Inner porosity = {actual_porosity_inner:.2%}, '
          f'Full porosity = {full_porosity:.2%}')
plt.colorbar(label='Fluid (1) / Solid (0)')
plt.tight_layout()
plt.savefig('binary_1500x1500_border5.png', dpi=100)
plt.show()

# 如果需要保存为npy文件，可以取消注释：
# np.save('binary_1500x1500.npy', full_binary)