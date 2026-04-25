import os
import json
import numpy as np
import matplotlib.pyplot as plt
import hudson as hd
import fault as ft
import sls
from scipy.ndimage import gaussian_filter

SOLID   = 0
VESOLID = 1
FLUID   = 2
# ========== 模型参数 ==========
# 粗网格尺寸
nx = 501
nz = 701
dx = 1.5
dz = 1.5

# 时间参数
fpeak = 30.0
dt = 5e-6
nt = 120000
snapshot = 5000

# 基准模型 VTI
epsilon_1 = 0.08
delta_1 = 0.03

rho_1 = 2400.0
vp_1 = 4000.0
vs_1 = 2100.0
C33_1 = rho_1 * vp_1**2
C55_1 = rho_1 * vs_1**2
C11_1 = C33_1 * (1 + 2 * epsilon_1)
C13_1 = ((C33_1 - C55_1) * (2 * C33_1 * delta_1 + (C33_1 - C55_1)))**0.5 - C55_1

epsilon_2 = 0.06
delta_2 = 0.02
gamma_2 = 0.04
rho_2 = 2500.0
vp_2 = 4300.0
vs_2 = 2200.0
C33_2 = rho_2 * vp_2**2
C55_2 = rho_2 * vs_2**2
C11_2 = C33_2 * (1 + 2 * epsilon_2)
C13_2 = ((C33_2 - C55_2) * (2 * C33_2 * delta_2 + (C33_2 - C55_2)))**0.5 - C55_2

Qp_ve = 1000
Qs_ve = 600
sls_params = sls.get_sls_parameters(Qp_ve, Qs_ve, 3, 2, 50)
inv_tss = 1 / sls_params["tau_sigmas"]
taup = sls_params["taup"]
taus = sls_params["taus"]

# 第二层的空洞流体
rho_fluid = 850
vp_fluid = 1300
vs_fluid = 0
C11_fluid = rho_fluid * vp_fluid**2
C55_fluid = rho_fluid * vs_fluid**2
C13_fluid = C11_fluid - 2 * C55_fluid
C33_fluid = C11_fluid
Qp_fluid = 50
zeta = C11_fluid / (2 * np.pi * fpeak * Qp_fluid)

# 第三层模型 VTI
epsilon_3 = 0.05
delta_3 = 0.02
gamma_3 = 0.03
rho_3 = 2600.0
vp_3 = 4400.0
vs_3 = 2250.0
C33_3 = rho_3 * vp_3**2
C55_3 = rho_3 * vs_3**2
C11_3 = C33_3 * (1 + 2 * epsilon_3)
C13_3 = ((C33_3 - C55_3) * (2 * C33_3 * delta_3 + (C33_3 - C55_3)))**0.5 - C55_3

# 细网格区域（粗网格坐标）
fine_regions = [
    {
        "x_start": 200, "x_end": 300,
        "z_start": 350, "z_end": 500,
        "N": 9
    }
]

# 细网格边界无流体宽度（格）
border = 30
# 目标孔隙度（仅内部区域）
phi_target = 0.05
# 孔隙斑块形状（sigma ≈ 半径）
sigma_x = 8
sigma_y = 6
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
coarse_rho = np.full((nz, nx), rho_2, dtype=np.float32)
coarse_C11 = np.full((nz, nx), C11_2, dtype=np.float32)
coarse_C13 = np.full((nz, nx), C13_2, dtype=np.float32)
coarse_C33 = np.full((nz, nx), C33_2, dtype=np.float32)
coarse_C55 = np.full((nz, nx), C55_2, dtype=np.float32)
coarse_zeta = np.full((nz, nx), 0, dtype=np.float32)
coarse_taup = np.full((nz, nx), 0, dtype=np.float32)
coarse_taus = np.full((nz, nx), 0, dtype=np.float32)
coarse_inv_tsig1 = np.full((nz, nx), 0, dtype=np.float32)
coarse_inv_tsig2 = np.full((nz, nx), 0, dtype=np.float32)
coarse_inv_tsig3 = np.full((nz, nx), 0, dtype=np.float32)

# coarse_rho[300:550, :] = rho_2
# coarse_C11[300:550, :] = C11_2
# coarse_C13[300:550, :] = C13_2
# coarse_C33[300:550, :] = C33_2
# coarse_C55[300:550, :] = C55_2

coarse_rho[600:, :] = rho_3
coarse_C11[600:, :] = C11_3
coarse_C13[600:, :] = C13_3
coarse_C33[600:, :] = C33_3
coarse_C55[600:, :] = C55_3

# ft.add_fault(rho=coarse_rho, C11=coarse_C11, C13=coarse_C13, C33=coarse_C33, C55=coarse_C55,
#              nx=nx, nz=nz, fault_start=(250, 180), fault_end=(310, 400),
#              fault_total_width=6, core_width=1.5)
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

    # 细网格尺寸
    lenx = (xend - xstart) * N + 1
    lenz = (zend - zstart) * N + 1

    # 创建该区域对应的子目录
    region_dir = os.path.join(fine_dir, str(idx))
    os.makedirs(region_dir, exist_ok=True)
    
    frac_r = 0.08
    frac_thickness = 0.0008
    alpha = frac_thickness / frac_r
    phi = 0.05 * 4 * np.pi * alpha / 3
    print(f"phi = {phi}")
    hudson_params = hd.hudson_crack_sls_parameters(
        Vp0=vp_2, Vs0=vs_2, rho=rho_2,
        epsilon=epsilon_2, delta=delta_2, gamma=gamma_2,
        lambda_fluid=vp_fluid**2 * rho_fluid,
        eta=0.08, f0=fpeak,
        crack_density=0.05, a=frac_r, c=frac_thickness,
    )

    # 初始化细网格模型（背景固体属性，与粗网格相同）
    fine_rho = np.full((lenz, lenx), rho_2, dtype=np.float32)
    fine_C11 = np.full((lenz, lenx), C11_2, dtype=np.float32)
    fine_C13 = np.full((lenz, lenx), C13_2, dtype=np.float32)
    fine_C33 = np.full((lenz, lenx), C33_2, dtype=np.float32)
    fine_C55 = np.full((lenz, lenx), C55_2, dtype=np.float32)
    fine_taup = np.full((lenz, lenx), taup, dtype=np.float32)
    fine_taus = np.full((lenz, lenx), taus, dtype=np.float32)
    fine_inv_tsig1 = np.full((lenz, lenx), inv_tss[0], dtype=np.float32)
    fine_inv_tsig2 = np.full((lenz, lenx), inv_tss[1], dtype=np.float32)
    fine_inv_tsig3 = np.full((lenz, lenx), inv_tss[2], dtype=np.float32)
    fine_zeta = np.full((lenz, lenx), 0, dtype=np.float32)
    fine_MAT = np.full((lenz, lenx), SOLID, dtype=np.int32)

    fine_MAT[2:-2, 2:-2] = VESOLID
    # fine_C11[fine_MAT == VESOLID] = hudson_params['C11']
    # fine_C13[fine_MAT == VESOLID] = hudson_params['C13']
    # fine_C33[fine_MAT == VESOLID] = hudson_params['C33']
    # fine_C55[fine_MAT == VESOLID] = hudson_params['C55']
    fine_taup[fine_MAT == VESOLID] = taup
    fine_taus[fine_MAT == VESOLID] = taus
    fine_inv_tsig1[fine_MAT == VESOLID] = inv_tss[0]
    fine_inv_tsig2[fine_MAT == VESOLID] = inv_tss[1]
    fine_inv_tsig3[fine_MAT == VESOLID] = inv_tss[2]

    # 生成二值孔隙掩膜（1表示流体，0表示固体）
    porosity_mask = generate_porosity_mask(
        shape=(lenz, lenx),
        border=border,
        phi_target=phi_target,
        sigma=(sigma_y, sigma_x),  # 注意：sigma顺序为 (sigma_z, sigma_x)
        seed=seed + idx            # 每个区域不同种子，避免重复
    )

    # 将流体区域标记为 FLUID
    # fine_MAT[porosity_mask == 1] = FLUID
    fine_C11[fine_MAT == FLUID] = C11_fluid
    fine_C13[fine_MAT == FLUID] = C13_fluid
    fine_C33[fine_MAT == FLUID] = C33_fluid
    fine_C55[fine_MAT == FLUID] = C55_fluid
    fine_rho[fine_MAT == FLUID] = rho_fluid
    fine_zeta[fine_MAT == FLUID] = zeta
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
    
    # ========== 保存细网格文件 ==========
    fine_MAT.tofile(os.path.join(region_dir, "material.bin"))
    fine_rho.tofile(os.path.join(region_dir, "rho.bin"))
    fine_C11.tofile(os.path.join(region_dir, "C11.bin"))
    fine_C13.tofile(os.path.join(region_dir, "C13.bin"))
    fine_C33.tofile(os.path.join(region_dir, "C33.bin"))
    fine_C55.tofile(os.path.join(region_dir, "C55.bin"))
    fine_zeta.tofile(os.path.join(region_dir, "zeta.bin"))
    fine_taup.tofile(os.path.join(region_dir, "taup.bin"))
    fine_taus.tofile(os.path.join(region_dir, "taus.bin"))
    fine_inv_tsig1.tofile(os.path.join(region_dir, "inv_tsig1.bin"))
    fine_inv_tsig2.tofile(os.path.join(region_dir, "inv_tsig2.bin"))
    fine_inv_tsig3.tofile(os.path.join(region_dir, "inv_tsig3.bin"))

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
        "zeta": f"models/fine/{idx}/zeta.bin",
        "taup": f"models/fine/{idx}/taup.bin",
        "taus": f"models/fine/{idx}/taus.bin",
        "inv_tsig1": f"models/fine/{idx}/inv_tsig1.bin",
        "inv_tsig2": f"models/fine/{idx}/inv_tsig2.bin",
        "inv_tsig3": f"models/fine/{idx}/inv_tsig3.bin",
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
        "zeta": "models/coarse/zeta.bin",
        "taup": "models/coarse/taup.bin",
        "taus": "models/coarse/taus.bin",
        "inv_tsig1": "models/coarse/inv_tsig1.bin",
        "inv_tsig2": "models/coarse/inv_tsig2.bin",
        "inv_tsig3": "models/coarse/inv_tsig3.bin",
        "material": "models/coarse/material.bin",
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
coarse_inv_tsig1.tofile(os.path.join(coarse_dir, "inv_tsig1.bin"))
coarse_inv_tsig2.tofile(os.path.join(coarse_dir, "inv_tsig2.bin"))
coarse_inv_tsig3.tofile(os.path.join(coarse_dir, "inv_tsig3.bin"))
coarse_zeta.tofile(os.path.join(coarse_dir, "zeta.bin"))
