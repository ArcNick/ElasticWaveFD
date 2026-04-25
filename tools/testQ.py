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
nx = 1001
nz = 1001
dx = 2
dz = 2

# 时间参数
fpeak = 30.0
dt = 5e-5
nt = 50000
snapshot = 400

Qp_ve = 200
Qs_ve = 120
sls_params = sls.get_sls_parameters(Qp_ve, Qs_ve, 3, 2, 50)
inv_tss = 1 / sls_params["tau_sigmas"]
taup = sls_params["taup"]
taus = sls_params["taus"]

epsilon = 0.00
delta = 0.00
gamma = 0.00
rho1 = 2400.0
vp1 = 4000.0
vs1 = 4000 / 3**0.5
C33_1 = rho1 * vp1**2
C55_1 = rho1 * vs1**2
C11_1 = C33_1 * (1 + 2 * epsilon)
C13_1 = ((C33_1 - C55_1) * (2 * C33_1 * delta + (C33_1 - C55_1)))**0.5 - C55_1

rho2 = 2800.0
vp2 = 5000.0
vs2 = 5000 / 3**0.5
C33_2 = rho2 * vp2**2
C55_2 = rho2 * vs2**2
C11_2 = C33_2 * (1 + 2 * epsilon)
C13_2 = ((C33_2 - C55_2) * (2 * C33_2 * delta + (C33_2 - C55_2)))**0.5 - C55_2

# 震源位置
posx = nx // 2
posz = nz // 2

# CPML参数
cpml_thickness = 20
cpml_N = 3
cp_max = 5200
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
coarse_rho = np.full((nz, nx), rho1, dtype=np.float32)
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

coarse_MAT[30:-30, 30:-30] = VESOLID
coarse_taup[coarse_MAT == VESOLID] = taup
coarse_taus[coarse_MAT == VESOLID] = taus
coarse_inv_tsig1[coarse_MAT == VESOLID] = inv_tss[0]
coarse_inv_tsig2[coarse_MAT == VESOLID] = inv_tss[1]
coarse_inv_tsig3[coarse_MAT == VESOLID] = inv_tss[2]

# coarse_rho[600:, :] = rho2
# coarse_C11[600:, :] = C11_2
# coarse_C13[600:, :] = C13_2
# coarse_C33[600:, :] = C33_2
# coarse_C55[600:, :] = C55_2

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
    "fine": []
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
