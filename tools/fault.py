import os
import json
import numpy as np
import matplotlib.pyplot as plt

SOLID = 0
FLUID = 1

TYPE1 = 1
TYPE2 = 2
TYPE3 = 3

# ========== 模型参数 ==========
# 粗网格尺寸
nx = 701
nz = 701
dx = 1.5
dz = 1.5

# 时间参数
fpeak = 30.0
dt = 5e-5
nt = 25000
snapshot = 500

# 顶层
epsilon_1 = -0.01
delta_1 = -0.015
rho_1 = 2280.0
vp_1 = 3600.0
vs_1 = 1800.0
C33_1 = rho_1 * vp_1**2
C55_1 = rho_1 * vs_1**2
C11_1 = C33_1 * (1 + 2 * epsilon_1)
C13_1 = ((C33_1 - C55_1) * (2 * C33_1 * delta_1 + (C33_1 - C55_1)))**0.5 - C55_1

# 中间层
epsilon_2 = -0.015
delta_2 = -0.025
rho_2 = 2580.0
vp_2 = 4400.0
vs_2 = 2300.0
C33_2 = rho_2 * vp_2**2
C55_2 = rho_2 * vs_2**2
C11_2 = C33_2 * (1 + 2 * epsilon_2)
C13_2 = ((C33_2 - C55_2) * (2 * C33_2 * delta_2 + (C33_2 - C55_2)))**0.5 - C55_2

# 底层
epsilon_3 = -0.02
delta_3 = -0.03
rho_3 = 2650
vp_3 = 4650
vs_3 = 2650
C33_3 = rho_3 * vp_3**2
C55_3 = rho_3 * vs_3**2
C11_3 = C33_3 * (1 + 2 * epsilon_3)
C13_3 = ((C33_3 - C55_3) * (2 * C33_3 * delta_3 + (C33_3 - C55_3)))**0.5 - C55_3

# 流体区（本脚本中未使用细网格，仅保留定义）
rho_fluid = 850.0
vp_fluid = 1300.0
vs_fluid = 0
C11_fluid = rho_fluid * vp_fluid**2
C55_fluid = rho_fluid * vs_fluid**2
C13_fluid = C11_fluid - 2 * C55_fluid
C33_fluid = C11_fluid
omega0 = 2 * np.pi * fpeak
Qp = 100
inv_ts = omega0 * Qp
tau = 1 / Qp * (1 + 1 / Qp**2)**0.5

# 细网格区域（本脚本中暂不使用）
fine_regions = [
    # {
    #     "x_start": 120, "x_end": 380,
    #     "z_start": 120, "z_end": 130,
    #     "N": 5
    # }
]

# 震源位置
posx = nx // 2
posz = 45

# CPML参数
cpml_thickness = 20
cpml_N = 3
cp_max = max(vp_1, vp_2, vp_3, vp_fluid)
Rc = 0.0001
kappa0 = 1.2

# ========== 断层参数 ==========
fault_enable = True
# 断层中心线两点 (x, z) 单位：米 (注意z从0向下)
fault_start = (600.0, 300.0)   # 上端点
fault_end   = (900.0, 500.0)   # 下端点
fault_total_width = 12.0        # 断层总宽度 (m)
core_width = 3.0                # 断裂核半宽 (m) → 核总宽 6m
# 断层物性 (核部最弱，损伤带中等)
# 核部物性
core_vp = 2400.0
core_vs = 900.0
core_rho = 2050.0
core_epsilon = 0.0      # 可设为各向同性
core_delta = 0.0
# 损伤带物性
damage_vp = 3600.0
damage_vs = 1850.0
damage_rho = 2450.0
damage_epsilon = 0.0
damage_delta = 0.0

# ========== 创建目录结构 ==========
base_dir = "models"
coarse_dir = os.path.join(base_dir, "coarse")
fine_dir = os.path.join(base_dir, "fine")
os.makedirs(coarse_dir, exist_ok=True)
os.makedirs(fine_dir, exist_ok=True)

# ========== 辅助函数：点到线段距离 ==========
def point_to_segment_distance(px, pz, x1, z1, x2, z2):
    dx = x2 - x1
    dz = z2 - z1
    l2 = dx*dx + dz*dz
    if l2 == 0:
        return np.hypot(px - x1, pz - z1)
    t = ((px - x1)*dx + (pz - z1)*dz) / l2
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_z = z1 + t * dz
    return np.hypot(px - proj_x, pz - proj_z)

# ========== 生成粗网格模型 ==========
coarse_MAT = np.full((nz, nx), SOLID, dtype=np.int32)
coarse_rho = np.full((nz, nx), rho_1, dtype=np.float32)
coarse_C11 = np.full((nz, nx), C11_1, dtype=np.float32)
coarse_C13 = np.full((nz, nx), C13_1, dtype=np.float32)
coarse_C33 = np.full((nz, nx), C33_1, dtype=np.float32)
coarse_C55 = np.full((nz, nx), C55_1, dtype=np.float32)
coarse_tau = np.full((nz, nx), 0, dtype=np.float32)
coarse_inv_taus = np.full((nz, nx), 0, dtype=np.float32)

# 分层赋值（索引范围：0-250为中间层，250-550为底层，其余顶层）
coarse_rho[0:250, :] = rho_2
coarse_C11[0:250, :] = C11_2
coarse_C13[0:250, :] = C13_2
coarse_C33[0:250, :] = C33_2
coarse_C55[0:250, :] = C55_2

coarse_rho[250:550, :] = rho_3
coarse_C11[250:550, :] = C11_3
coarse_C13[250:550, :] = C13_3
coarse_C33[250:550, :] = C33_3
coarse_C55[250:550, :] = C55_3

# ========== 添加断层（覆盖背景） ==========
if fault_enable:
    print("正在添加断层...")
    # 生成网格坐标 (米)
    x_coords = np.arange(nx) * dx
    z_coords = np.arange(nz) * dz
    X, Z = np.meshgrid(x_coords, z_coords, indexing='ij')  # shape (nx, nz)
    # 计算每个网格点到断层中心线的距离
    dist = np.zeros((nz, nx))
    x1, z1 = fault_start
    x2, z2 = fault_end
    for ix in range(nx):
        for iz in range(nz):
            dist[iz, ix] = point_to_segment_distance(x_coords[ix], z_coords[iz], x1, z1, x2, z2)
    
    # 定义 mask
    core_mask = (dist <= core_width)
    damage_mask = (dist > core_width) & (dist <= fault_total_width/2.0)
    
    # 计算断裂核的弹性常数
    C33_core = core_rho * core_vp**2
    C55_core = core_rho * core_vs**2
    C11_core = C33_core * (1 + 2*core_epsilon)
    temp = (C33_core - C55_core) * (2*C33_core*core_delta + (C33_core - C55_core))
    if temp < 0:
        temp = 0
    C13_core = np.sqrt(temp) - C55_core
    
    # 计算损伤带的弹性常数
    C33_damage = damage_rho * damage_vp**2
    C55_damage = damage_rho * damage_vs**2
    C11_damage = C33_damage * (1 + 2*damage_epsilon)
    temp = (C33_damage - C55_damage) * (2*C33_damage*damage_delta + (C33_damage - C55_damage))
    if temp < 0:
        temp = 0
    C13_damage = np.sqrt(temp) - C55_damage
    
    # 覆盖物性
    coarse_rho[core_mask] = core_rho
    coarse_C11[core_mask] = C11_core
    coarse_C13[core_mask] = C13_core
    coarse_C33[core_mask] = C33_core
    coarse_C55[core_mask] = C55_core
    
    coarse_rho[damage_mask] = damage_rho
    coarse_C11[damage_mask] = C11_damage
    coarse_C13[damage_mask] = C13_damage
    coarse_C33[damage_mask] = C33_damage
    coarse_C55[damage_mask] = C55_damage
    
    print("断层添加完成。")

# ========== 可视化检查（不保存二进制文件） ==========
# 为了快速查看效果，绘制 Vp 剖面
vp_field = np.zeros((nz, nx))
# 从 C33 和 rho 反算 Vp (近似，忽略各向异性影响，仅用于显示)
vp_field = np.sqrt(coarse_C33 / coarse_rho)

plt.figure(figsize=(12, 8))
plt.imshow(vp_field, origin='upper', extent=[0, nx*dx, nz*dz, 0],
           cmap='viridis', aspect='auto')
plt.colorbar(label='Vp (m/s)')
plt.title('P-wave velocity model with fault')
plt.xlabel('X (m)')
plt.ylabel('Z (m)')
plt.tight_layout()
plt.savefig('velocity_model_with_fault.png', dpi=150)
plt.show()

# ========== 生成细网格模型（本脚本未使用，保持原样） ==========
fine_list = []
for idx, region in enumerate(fine_regions):
    # ... 原代码不变，但 fine_regions 为空，所以不会执行
    pass

# ========== 生成 models.json 和 params.json ==========
models_config = {
    "coarse": {
        "nx": nx,
        "nz": nz,
        "dx": dx,
        "dz": dz,
        "rho": "models/coarse/rho.bin",
        "C11": "models/coarse/C11.bin",
        "C13": "models/coarse/C13.bin",
        "C33": "models/coarse/C33.bin",
        "C55": "models/coarse/C55.bin",
        "tau": "models/coarse/tau.bin",
        "material": "models/coarse/material.bin",
        "inv_taus": "models/coarse/inv_tau_sigma.bin"
    },
    "fine": fine_list
}

with open(os.path.join(base_dir, "models.json"), "w") as f:
    json.dump(models_config, f, indent=2)
print("models.json 已生成")

params_config = {
    "base": {
        "fpeak": fpeak,
        "dt": dt,
        "nt": nt,
        "posx": posx,
        "posz": posz,
        "snapshot": snapshot
    },
    "cpml": {
        "thickness": cpml_thickness,
        "N": cpml_N,
        "cp_max": cp_max,
        "Rc": Rc,
        "kappa0": kappa0
    }
}

with open(os.path.join(base_dir, "params.json"), "w") as f:
    json.dump(params_config, f, indent=2)
print("params.json 已生成")

# ========== 二进制文件输出（已注释，需要时取消注释） ==========
coarse_MAT.tofile(os.path.join(coarse_dir, "material.bin"))
coarse_rho.tofile(os.path.join(coarse_dir, "rho.bin"))
coarse_C11.tofile(os.path.join(coarse_dir, "C11.bin"))
coarse_C13.tofile(os.path.join(coarse_dir, "C13.bin"))
coarse_C33.tofile(os.path.join(coarse_dir, "C33.bin"))
coarse_C55.tofile(os.path.join(coarse_dir, "C55.bin"))
coarse_tau.tofile(os.path.join(coarse_dir, "tau.bin"))
coarse_inv_taus.tofile(os.path.join(coarse_dir, "inv_tau_sigma.bin"))