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
nz = 501
dx = 2
dz = 2

# 时间参数
fpeak = 30.0
dt = 2e-6
nt = 300000
snapshot = 6400

# 基准模型
epsilon_1 = 0.00
delta_1 = 0.00

rho_1 = 2650.0
vp_1 = 4600.0
vs_1 = 4600 / 3**0.5
C33_1 = rho_1 * vp_1**2
C55_1 = rho_1 * vs_1**2
C11_1 = C33_1 * (1 + 2 * epsilon_1)
C13_1 = ((C33_1 - C55_1) * (2 * C33_1 * delta_1 + (C33_1 - C55_1)))**0.5 - C55_1

# Qp_ve = 1000
# Qs_ve = 600
# sls_params = sls.get_sls_parameters(Qp_ve, Qs_ve, 3, 2, 50)
# inv_tss = 1 / sls_params["tau_sigmas"]
# taup = sls_params["taup"]
# taus = sls_params["taus"]

# 第二层的空洞流体
rho_fluid = 850
vp_fluid = 1300
vs_fluid = 0
C11_fluid = rho_fluid * vp_fluid**2
C55_fluid = rho_fluid * vs_fluid**2
C13_fluid = C11_fluid - 2 * C55_fluid
C33_fluid = C11_fluid
Qp_fluid = 40
zeta = C11_fluid / (2 * np.pi * fpeak * Qp_fluid)

# 细网格区域（粗网格坐标）
fine_regions = [
    {
        "x_start": 245, "x_end": 255,
        "z_start": 245, "z_end": 255,
        "N": 11
    }
]

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
coarse_zeta = np.full((nz, nx), 0, dtype=np.float32)
coarse_taup = np.full((nz, nx), 0, dtype=np.float32)
coarse_taus = np.full((nz, nx), 0, dtype=np.float32)
coarse_inv_tsig1 = np.full((nz, nx), 0, dtype=np.float32)
coarse_inv_tsig2 = np.full((nz, nx), 0, dtype=np.float32)
coarse_inv_tsig3 = np.full((nz, nx), 0, dtype=np.float32)

# coarse_MAT[nz // 2, nx // 2] = FLUID
# coarse_rho[coarse_MAT == FLUID] = rho_fluid
# coarse_C11[coarse_MAT == FLUID] = C11_fluid
# coarse_C13[coarse_MAT == FLUID] = C13_fluid
# coarse_C33[coarse_MAT == FLUID] = C33_fluid
# coarse_C55[coarse_MAT == FLUID] = C55_fluid
# coarse_zeta[coarse_MAT == FLUID] = zeta

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
    
    # 初始化细网格模型（背景固体属性，与粗网格相同）
    fine_rho = np.full((lenz, lenx), rho_1, dtype=np.float32)
    fine_C11 = np.full((lenz, lenx), C11_1, dtype=np.float32)
    fine_C13 = np.full((lenz, lenx), C13_1, dtype=np.float32)
    fine_C33 = np.full((lenz, lenx), C33_1, dtype=np.float32)
    fine_C55 = np.full((lenz, lenx), C55_1, dtype=np.float32)
    fine_taup = np.full((lenz, lenx), 0, dtype=np.float32)
    fine_taus = np.full((lenz, lenx), 0, dtype=np.float32)
    fine_inv_tsig1 = np.full((lenz, lenx), 0, dtype=np.float32)
    fine_inv_tsig2 = np.full((lenz, lenx), 0, dtype=np.float32)
    fine_inv_tsig3 = np.full((lenz, lenx), 0, dtype=np.float32)
    fine_zeta = np.full((lenz, lenx), 0, dtype=np.float32)
    fine_MAT = np.full((lenz, lenx), SOLID, dtype=np.int32)

    midz = lenz // 2
    midx = lenx // 2
    pz1 = midz - 5
    px1 = midx - 5
    pz2 = midz + 5
    px2 = midx + 5
    def getdis(x1, z1, x2, z2):
        return ((x1 - x2)**2 + (z1 - z2)**2)**0.5
    for ix in range(0, lenx):
        for iz in range(0, lenz):
            # dis = getdis(ix, iz, px1, pz1) + getdis(ix, iz, px2, pz2)
            # if dis <= 15:
            #     fine_MAT[iz, ix] = FLUID
            dis = getdis(ix, iz, midx, midz)
            if dis <= 5.5:
                fine_MAT[iz, ix] = FLUID
    # for ix in range(midx - 5, midx + 6):
    #     for iz in range(midz - 5, midz + 6):
    #         fine_MAT[iz, ix] = FLUID

    fine_C11[fine_MAT == FLUID] = C11_fluid
    fine_C13[fine_MAT == FLUID] = C13_fluid
    fine_C33[fine_MAT == FLUID] = C33_fluid
    fine_C55[fine_MAT == FLUID] = C55_fluid
    fine_rho[fine_MAT == FLUID] = rho_fluid
    fine_zeta[fine_MAT == FLUID] = zeta
    
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
