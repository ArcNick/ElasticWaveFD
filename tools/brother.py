import numpy as np
import matplotlib.pyplot as plt
import sls
import os
import json
SOLID = 0
VESOLID = 1
FLUID = 2

# --- 参数设置 ---
nz, nx = 706, 690  # 原始尺寸
dx = 1.5               # 目标间距
dz = 1.5

# 时间参数
fpeak = 30.0
dt = 1e-5
nt = 60000
snapshot = 800

input_file = "vp.bin"

vp = np.fromfile(input_file, dtype=np.float32)
vp = vp.reshape((nz, nx))
vp_max = vp.max()
vs = vp / 1.89
coarse_rho = 310 * np.power(vp, 0.25)

epsilon_1 = 0.0
delta_1 = 0.0
coarse_C33 = coarse_rho * vp**2
coarse_C55 = coarse_rho * vs**2
coarse_C11 = coarse_C33 * (1 + 2 * epsilon_1)
coarse_C13 = ((coarse_C33 - coarse_C55) * (2 * coarse_C33 * delta_1 + (coarse_C33 - coarse_C55)))**0.5 - coarse_C55

Qp1 = 70
Qs1 = 40
sls_params = sls.get_sls_parameters(Qp1, Qs1, 3, 2, 100)
inv_tss1 = 1 / sls_params["tau_sigmas"]
taup1 = sls_params["taup"]
taus1 = sls_params["taus"]

Qp2 = 80
Qs2 = 50
sls_params = sls.get_sls_parameters(Qp2, Qs2, 3, 2, 100)
inv_tss2 = 1 / sls_params["tau_sigmas"]
taup2 = sls_params["taup"]
taus2 = sls_params["taus"]

# 震源位置
posx = nx // 2
posz = 38

# CPML参数
cpml_thickness = 20
cpml_N = 3
cp_max = 6500
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
coarse_zeta = np.full((nz, nx), 0, dtype=np.float32)
coarse_taup = np.full((nz, nx), 0, dtype=np.float32)
coarse_taus = np.full((nz, nx), 0, dtype=np.float32)
coarse_inv_tsig1 = np.full((nz, nx), 0, dtype=np.float32)
coarse_inv_tsig2 = np.full((nz, nx), 0, dtype=np.float32)
coarse_inv_tsig3 = np.full((nz, nx), 0, dtype=np.float32)

coarse_MAT[(vp == 4000) | (vp == 5500) | (vp == 5550) | (vp == 5580) | (vp == 5630) | (vp == 5750)] = VESOLID
coarse_MAT[-30:, :] = SOLID
coarse_taup[vp == 4000] = taup1
coarse_taus[vp == 4000] = taus1
coarse_inv_tsig1[vp == 4000] = inv_tss1[0]
coarse_inv_tsig2[vp == 4000] = inv_tss1[1]
coarse_inv_tsig3[vp == 4000] = inv_tss1[2]

coarse_taup[(vp == 5500) | (vp == 5550) | (vp == 5580) | (vp == 5630) | (vp == 5750)] = taup2
coarse_taus[(vp == 5500) | (vp == 5550) | (vp == 5580) | (vp == 5630) | (vp == 5750)] = taus2
coarse_inv_tsig1[(vp == 5500) | (vp == 5550) | (vp == 5580) | (vp == 5630) | (vp == 5750)] = inv_tss2[0]
coarse_inv_tsig2[(vp == 5500) | (vp == 5550) | (vp == 5580) | (vp == 5630) | (vp == 5750)] = inv_tss2[1]
coarse_inv_tsig3[(vp == 5500) | (vp == 5550) | (vp == 5580) | (vp == 5630) | (vp == 5750)] = inv_tss2[2]

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
