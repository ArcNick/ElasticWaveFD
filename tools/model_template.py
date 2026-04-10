import os
import json
import numpy as np
import matplotlib.pyplot as plt
import hudson
from scipy.ndimage import gaussian_filter

SOLID   = 0
VESOLID = 1
FLUID   = 2
# ========== 模型参数 ==========
# 粗网格尺寸
nx = 701
nz = 701
dx = 1.5
dz = 1.5

# 时间参数
fpeak = 30.0
dt = 8e-6
nt = 120000
snapshot = 1000

# 基准模型 VTI
epsilon = -0.015
delta = -0.025

rho_1 = 2650.0
vp_1 = 4600.0
vs_1 = 2300.0
C33_1 = rho_1 * vp_1**2
C55_1 = rho_1 * vs_1**2
C11_1 = C33_1 * (1 + 2 * epsilon)
C13_1 = ((C33_1 - C55_1) * (2 * C33_1 * delta + (C33_1 - C55_1)))**0.5 - C55_1

# 粘弹性的基准模型 VTI
epsilon = -0.015
delta = -0.025

rho_ve = 2650.0
vp_ve = 4600.0
vs_ve = 2300.0
C33_ve = rho_ve * vp_ve**2
C55_ve = rho_ve * vs_ve**2
C11_ve = C33_ve * (1 + 2 * epsilon)
C13_ve = ((C33_ve - C55_ve) * (2 * C33_ve * delta + (C33_ve - C55_ve)))**0.5 - C55_ve
omega0_ve = 2 * np.pi * fpeak
Qp_ve = 40
Qs_ve = 20
inv_ts_ve = omega0_ve * Qp_ve
taup_ve = 1 / Qp_ve * (1 + 1 / Qp_ve**2)**0.5
taus_ve = 1 / Qs_ve * (1 + 1 / Qs_ve**2)**0.5

rho_fluid = 850
vp_fluid = 1300
vs_fluid = 0
C11_fluid = rho_fluid * vp_fluid**2
C55_fluid = rho_fluid * vs_fluid**2
C13_fluid = C11_fluid - 2 * C55_fluid
C33_fluid = C11_fluid
omega0_fluid = 2 * np.pi * fpeak
Qp_fluid = 40
inv_ts_fluid = omega0_fluid * Qp_fluid
taup_fluid = 1 / Qp_fluid * (1 + 1 / Qp_fluid**2)**0.5

# 细网格区域（粗网格坐标）
fine_regions = [
    {
        "x_start": 300, "x_end": 400,
        "z_start": 300, "z_end": 400,
        "N": 9
    }
]

# 细网格边界无流体宽度（格）
border = 50
# 目标孔隙度（仅内部区域）
phi_target = 0.05
# 孔隙斑块形状（sigma ≈ 半径）
sigma_x = 2.5
sigma_y = 1.5
seed = 12345

# 震源位置
posx = nx // 2
posz = 40

# CPML参数
cpml_thickness = 20
cpml_N = 3
cp_max = vp_1
Rc = 0.0001
kappa0 = 1.2

# ========== 创建目录结构 ==========
base_dir = "models"
coarse_dir = os.path.join(base_dir, "coarse")
fine_dir = os.path.join(base_dir, "fine")
os.makedirs(coarse_dir, exist_ok=True)
os.makedirs(fine_dir, exist_ok=True)

# ========== 生成粗网格模型 ==========
coarse_MAT = np.full((nz, nx), SOLID, dtype=np.int32)
coarse_rho = np.full((nz, nx), rho_1, dtype=np.float32)
coarse_C11 = np.full((nz, nx), C11_1, dtype=np.float32)
coarse_C13 = np.full((nz, nx), C13_1, dtype=np.float32)
coarse_C33 = np.full((nz, nx), C33_1, dtype=np.float32)
coarse_C55 = np.full((nz, nx), C55_1, dtype=np.float32)
coarse_taup = np.full((nz, nx), 0, dtype=np.float32)
coarse_taus = np.full((nz, nx), 0, dtype=np.float32)
coarse_inv_tsig = np.full((nz, nx), 0, dtype=np.float32)

# coarse_MAT[100:400, 250:400] = VESOLID
# coarse_rho[100:400, 250:400] = rho_ve
# coarse_C11[100:400, 250:400] = C11_ve
# coarse_C13[100:400, 250:400] = C13_ve
# coarse_C33[100:400, 250:400] = C33_ve
# coarse_C55[100:400, 250:400] = C55_ve
# coarse_taup[100:400, 250:400] = taup_ve
# coarse_taus[100:400, 250:400] = taus_ve
# coarse_inv_tsig[100:400, 250:400] = inv_ts_ve
# ========== 粗网格可视化：纵波阻抗 ==========
coarse_imp = np.sqrt(coarse_C33 * coarse_rho)  # 纵波阻抗
plt.figure(figsize=(12, 10))
plt.imshow(coarse_imp, cmap='viridis', aspect='auto', origin='upper')
plt.colorbar(label='Impedance (kg/(m²·s))')
plt.title('Coarse Model: P-wave Impedance')
plt.xlabel('X (grid cells)')
plt.ylabel('Z (grid cells)')
plt.tight_layout()
plt.savefig(os.path.join(coarse_dir, "impedance.png"), dpi=150)
plt.close()
print("粗网格阻抗图已保存至 models/coarse/impedance.png")

# ========== 辅助函数：生成二值孔隙分布 ======g====
def generate_porosity_mask(shape, border, phi_target, sigma=(2.0,1.2), seed=None):
    """
    生成二值掩膜，内部区域孔隙度=phi_target，边界border内为0。
    shape: (nz, nx) 整个区域的尺寸
    border: 边界宽度（格）
    sigma: (sigma_z, sigma_x) 高斯滤波各向异性半径
    """
    nz_total, nx_total = shape
    inner_nz = nz_total - 2 * border
    inner_nx = nx_total - 2 * border
    if inner_nz <= 0 or inner_nx <= 0:
        raise ValueError(f"border过大: border={border}")
    
    if seed is not None:
        np.random.seed(seed)
    
    # 内部区域白噪声
    noise = np.random.randn(inner_nz, inner_nx)
    # 各向异性高斯平滑
    smoothed = gaussian_filter(noise, sigma=sigma, mode='reflect')
    # 阈值控制孔隙度
    threshold = np.percentile(smoothed, phi_target * 100)
    inner_binary = (smoothed <= threshold).astype(np.uint8)
    
    # 扩展至全区域，边界填0
    full_binary = np.zeros((nz_total, nx_total), dtype=np.uint8)
    full_binary[border:nz_total-border, border:nx_total-border] = inner_binary
    return full_binary

# ========== 生成细网格模型 ==========
fine_list = []
for idx, region in enumerate(fine_regions):
    xstart = region["x_start"]
    xend = region["x_end"]
    zstart = region["z_start"]
    zend = region["z_end"]
    N = region["N"]

    # 细网格尺寸（注意：坐标范围是从粗网格索引转换而来，但这里直接计算点数）
    lenx = (xend - xstart) * N + 1
    lenz = (zend - zstart) * N + 1

    # 创建该区域对应的子目录
    region_dir = os.path.join(fine_dir, str(idx))
    os.makedirs(region_dir, exist_ok=True)
    
    # 初始化细网格模型（背景固体属性，与粗网格相同）
    fine_rho = np.full((lenz, lenx), rho_ve, dtype=np.float32)
    fine_C11 = np.full((lenz, lenx), C11_ve, dtype=np.float32)
    fine_C13 = np.full((lenz, lenx), C13_ve, dtype=np.float32)
    fine_C33 = np.full((lenz, lenx), C33_ve, dtype=np.float32)
    fine_C55 = np.full((lenz, lenx), C55_ve, dtype=np.float32)
    fine_taup = np.full((lenz, lenx), taup_ve, dtype=np.float32)
    fine_taus = np.full((lenz, lenx), taus_ve, dtype=np.float32)
    fine_inv_tsig = np.full((lenz, lenx), inv_ts_ve, dtype=np.float32)
    fine_MAT = np.full((lenz, lenx), VESOLID, dtype=np.int32)

    # ========== 在细网格区域内生成流体分布 ==========
    # 生成二值孔隙掩膜（1表示流体，0表示固体）
    porosity_mask = generate_porosity_mask(
        shape=(lenz, lenx),
        border=border,
        phi_target=phi_target,
        sigma=(sigma_y, sigma_x),  # 注意：sigma顺序为 (sigma_z, sigma_x)
        seed=seed + idx            # 每个区域不同种子，避免重复
    )
    
    # 将流体区域标记为 FLUID
    fine_MAT[porosity_mask == 1] = FLUID
    fine_C11[fine_MAT == FLUID] = C11_fluid
    fine_C13[fine_MAT == FLUID] = C13_fluid
    fine_C33[fine_MAT == FLUID] = C33_fluid
    fine_C55[fine_MAT == FLUID] = C55_fluid
    fine_rho[fine_MAT == FLUID] = rho_fluid
    fine_taup[fine_MAT == FLUID] = taup_fluid
    fine_taus[fine_MAT == FLUID] = 0
    fine_inv_tsig[fine_MAT == FLUID] = inv_ts_fluid
    
    # ========== 细网格可视化：孔隙掩膜 ==========
    plt.figure(figsize=(10, 8))
    plt.imshow(porosity_mask, cmap='gray', interpolation='none')
    plt.title(f"Region {idx}: porosity = {porosity_mask.mean():.2%}")
    plt.colorbar()
    plt.savefig(os.path.join(region_dir, "porosity_map.png"), dpi=100)
    plt.close()
    
    # ========== 细网格可视化：纵波阻抗 ==========
    fine_imp = np.sqrt(fine_C33 * fine_rho)  # 纵波阻抗
    plt.figure(figsize=(12, 10))
    plt.imshow(fine_imp, cmap='viridis', aspect='auto', origin='upper')
    plt.colorbar(label='Impedance (kg/(m²·s))')
    plt.title(f'Fine Region {idx}: P-wave Impedance\n'
              f'Size = {lenz}×{lenx} cells')
    plt.xlabel('X (fine grid cells)')
    plt.ylabel('Z (fine grid cells)')
    plt.tight_layout()
    plt.savefig(os.path.join(region_dir, "impedance.png"), dpi=150)
    plt.close()
    print(f"Region {idx} 阻抗图已保存至 {region_dir}/impedance.png")
    
    # 记录JSON信息
    fine_list.append({
        "x_start": xstart,
        "x_end": xend,
        "z_start": zstart,
        "z_end": zend,
        "N": N,
        "rho": f"models/fine/{idx}/rho.bin",
        "C11": f"models/fine/{idx}/C11.bin",
        "C13": f"models/fine/{idx}/C13.bin",
        "C33": f"models/fine/{idx}/C33.bin",
        "C55": f"models/fine/{idx}/C55.bin",
        "taup": f"models/fine/{idx}/taup.bin",
        "taus": f"models/fine/{idx}/taus.bin",
        "inv_tsig": f"models/fine/{idx}/inv_tsig.bin",
        "material": f"models/fine/{idx}/material.bin"
    })
    
    print(f"Region {idx}: size=({lenz},{lenx}), actual porosity = {porosity_mask.mean():.4f}")

# ========== 生成 models.json ==========
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
        "taup": "models/coarse/taup.bin",
        "taus": "models/coarse/taus.bin",
        "material": "models/coarse/material.bin",
        "inv_tsig": "models/coarse/inv_tsig.bin"
    },
    "fine": fine_list
}

with open(os.path.join(base_dir, "models.json"), "w") as f:
    json.dump(models_config, f, indent=2)
print("models.json 已生成")

# ========== 生成 params.json ==========
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

# ========== 保存粗网格文件 ==========
coarse_MAT.tofile(os.path.join(coarse_dir, "material.bin"))
coarse_rho.tofile(os.path.join(coarse_dir, "rho.bin"))
coarse_C11.tofile(os.path.join(coarse_dir, "C11.bin"))
coarse_C13.tofile(os.path.join(coarse_dir, "C13.bin"))
coarse_C33.tofile(os.path.join(coarse_dir, "C33.bin"))
coarse_C55.tofile(os.path.join(coarse_dir, "C55.bin"))
coarse_taup.tofile(os.path.join(coarse_dir, "taup.bin"))
coarse_taus.tofile(os.path.join(coarse_dir, "taus.bin"))
coarse_inv_tsig.tofile(os.path.join(coarse_dir, "inv_tsig.bin"))

# ========== 保存细网格文件 ==========
fine_MAT.tofile(os.path.join(region_dir, "material.bin"))
fine_rho.tofile(os.path.join(region_dir, "rho.bin"))
fine_C11.tofile(os.path.join(region_dir, "C11.bin"))
fine_C13.tofile(os.path.join(region_dir, "C13.bin"))
fine_C33.tofile(os.path.join(region_dir, "C33.bin"))
fine_C55.tofile(os.path.join(region_dir, "C55.bin"))
fine_taup.tofile(os.path.join(region_dir, "taup.bin"))
fine_taus.tofile(os.path.join(region_dir, "taus.bin"))
fine_inv_tsig.tofile(os.path.join(region_dir, "inv_tsig.bin"))