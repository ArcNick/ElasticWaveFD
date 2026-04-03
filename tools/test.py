import numpy as np
import matplotlib.pyplot as plt

# ============================================
# 用户参数设置
# ============================================

# 网格参数
nx = 701          # 粗网格 x 方向点数
nz = 701          # 粗网格 z 方向点数
dx = 1.5          # 网格步长 (m)
dz = 1.5

# 模型尺寸 (m)
x_min, x_max = 0, (nx-1)*dx
z_min, z_max = 0, (nz-1)*dz

# 层状介质定义 (从上到下)
# 每一层: [z_top, z_bottom, Vp0, Vs0, rho, epsilon, delta, gamma]
layers = [
    # 上覆地层 (0 - 400 m)
    [0, 400, 3500, 2000, 2500, 0.0, 0.0, 0.0],
    # 储层段 (400 - 1000 m)
    [400, 1000, 4600, 2600, 2650, -0.015, -0.025, 0.0],
]

# 断层参数
fault_enable = True
fault_start = (600.0, 200.0)   # (x, z) 单位 m
fault_end   = (900.0, 500.0)
fault_total_width = 10.0        # 断层总宽度 (m)
core_width = 2.0                # 断裂核半宽度 (m)

# 断层物性 (断裂核 和 损伤带)
core_Vp = 2400.0
core_Vs = 900.0
core_rho = 2050.0
damage_Vp = 3600.0
damage_Vs = 1850.0
damage_rho = 2450.0

# ============================================
# 辅助函数：点到线段距离
# ============================================
def point_to_segment_distance(px, pz, x1, z1, x2, z2):
    dx = x2 - x1
    dz = z2 - z1
    l2 = dx*dx + dz*dz
    if l2 == 0:
        return np.hypot(px - x1, pz - z1)
    t = ((px - x1)*dx + (pz - z1)*dz) / l2
    t = max(0, min(1, t))
    proj_x = x1 + t * dx
    proj_z = z1 + t * dz
    return np.hypot(px - proj_x, pz - proj_z)

# ============================================
# 生成网格坐标
# ============================================
x = np.linspace(x_min, x_max, nx)
z = np.linspace(z_min, z_max, nz)
X, Z = np.meshgrid(x, z, indexing='ij')  # shape (nx, nz) 但后面我们转置为 (nz, nx) 方便显示

# 初始化 Vp 数组 (用于可视化)
Vp = np.zeros((nz, nx))

# ============================================
# 1. 按层状背景赋值 Vp (仅用于展示)
# ============================================
for iz in range(nz):
    z_cur = z[iz]
    for layer in layers:
        z_top, z_bottom, vp0, _, _, _, _, _ = layer
        if z_top <= z_cur < z_bottom:
            Vp[iz, :] = vp0
            break

# ============================================
# 2. 添加断层 (覆盖背景)
# ============================================
if fault_enable:
    print("正在计算断层 mask...")
    x1, z1 = fault_start
    x2, z2 = fault_end
    dist = np.zeros((nz, nx))
    for ix in range(nx):
        for iz in range(nz):
            dist[iz, ix] = point_to_segment_distance(x[ix], z[iz], x1, z1, x2, z2)
    
    core_mask = (dist <= core_width)
    damage_mask = (dist > core_width) & (dist <= fault_total_width/2.0)
    
    # 覆盖 Vp
    Vp[core_mask] = core_Vp
    Vp[damage_mask] = damage_Vp

# ============================================
# 3. 绘制结果
# ============================================
plt.figure(figsize=(12, 8))

# 子图1：Vp 分布
plt.subplot(2,1,1)
im = plt.imshow(Vp, origin='lower', extent=[x_min, x_max, z_min, z_max], 
                aspect='auto', cmap='viridis')
plt.colorbar(im, label='Vp (m/s)')
plt.title('P-wave velocity model with fault')
plt.xlabel('X (m)')
plt.ylabel('Z (m)')

# 子图2：断层 mask 示意 (距离场)
plt.subplot(2,1,2)
dist_display = np.ma.masked_where(dist > fault_total_width/2, dist)
im2 = plt.imshow(dist_display, origin='lower', extent=[x_min, x_max, z_min, z_max],
                 aspect='auto', cmap='coolwarm', vmin=0, vmax=fault_total_width/2)
plt.colorbar(im2, label='Distance to fault center (m)')
plt.contour(X, Z, dist, levels=[core_width, fault_total_width/2], 
            colors=['red', 'blue'], linestyles='--', linewidths=1)
plt.title('Fault zone: red=core boundary, blue=damage boundary')
plt.xlabel('X (m)')
plt.ylabel('Z (m)')

plt.tight_layout()
plt.show()

# 打印信息
print(f"断层中心线: ({fault_start[0]}, {fault_start[1]}) -> ({fault_end[0]}, {fault_end[1]})")
print(f"倾角: {np.arctan2(fault_end[1]-fault_start[1], fault_end[0]-fault_start[0])*180/np.pi:.1f}°")
print(f"断裂核宽度: {2*core_width} m, 损伤带总宽: {fault_total_width} m")
print("如果断层位置满意，可以运行保存版本写入二进制文件。")