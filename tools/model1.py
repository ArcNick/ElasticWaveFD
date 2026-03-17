import os
import json
import numpy as np
import random

SOLID = 0
FLUID = 1

# ========== 模型参数 ==========
# 粗网格尺寸
nx = 501
nz = 501
dx = 1.0
dz = 1.0

# 时间参数
fpeak = 30.0
dt = 1e-5
nt = 30000
snapshot = 200

# 介质参数（均匀介质）
# 固体区域

# 盖层 0 - 250m
rho_1 = 2600.0
vp_1 = 5000.0
vs_1 = 2800.0
C11_1 = rho_1 * vp_1**2
C55_1 = rho_1 * vs_1**2
C13_1 = C11_1 - 2 * C55_1
C33_1 = C11_1
# 储层 250 - 400m
rho_2 = 2650
vp_2 = 5200
vs_2 = 2600
C11_2 = rho_2 * vp_2**2
C55_2 = rho_2 * vs_2**2
C13_2 = C11_2 - 2 * C55_2
C33_2 = C11_2
# 基底 400 - 500m
rho_3 = 2700
vp_3 = 5500
vs_3 = 2700
C11_3 = rho_3 * vp_3**2
C55_3 = rho_3 * vs_3**2
C13_3 = C11_3 - 2 * C55_3
C33_3 = C11_3

# 流体区
rho_fluid = 850.0
vp_fluid = 1300.0
vs_fluid = 0
C11_fluid = rho_fluid * vp_fluid**2
C55_fluid = rho_fluid * vs_fluid**2
C13_fluid = C11_fluid - 2 * C55_fluid
C33_fluid = C11_fluid
omega0 = 2 * np.pi * fpeak
Qp = 20
inv_ts = omega0 * Qp
tau = 1 / Qp * (1 + 1 / Qp**2)**0.5

# 细网格区域（粗网格坐标）
fine_regions = [
    {
        "x_start": 200, "x_end": 300,
        "z_start": 280, "z_end": 380,
        "N": 3
    }
]

# 震源位置
posx = nx // 2
posz = 40

# CPML参数
cpml_thickness = 20
cpml_N = 3
cp_max = max(vp_1, vp_2, vp_3, vp_fluid)
Rc = 0.001
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
coarse_tau = np.full((nz, nx), 0, dtype=np.float32)
coarse_inv_taus = np.full((nz, nx), 0, dtype=np.float32)

coarse_rho[250:400,] = rho_2
coarse_C11[250:400,] = C11_2
coarse_C13[250:400,] = C13_2
coarse_C33[250:400,] = C33_2
coarse_C55[250:400,] = C55_2

coarse_rho[400:501,] = rho_3
coarse_C11[400:501,] = C11_3
coarse_C13[400:501,] = C13_3
coarse_C33[400:501,] = C33_3
coarse_C55[400:501,] = C55_3

# 标记流体区域
# coarse_MAT[300:400, 200:300] = FLUID
# coarse_rho[coarse_MAT == FLUID] = rho_fluid
# coarse_C11[coarse_MAT == FLUID] = C11_fluid
# coarse_C13[coarse_MAT == FLUID] = C13_fluid
# coarse_C33[coarse_MAT == FLUID] = C33_fluid
# coarse_C55[coarse_MAT == FLUID] = C55_fluid
# coarse_tau[coarse_MAT == FLUID] = tau
# coarse_inv_taus[coarse_MAT == FLUID] = inv_ts

coarse_MAT.tofile(os.path.join(coarse_dir, "material.bin"))
coarse_rho.tofile(os.path.join(coarse_dir, "rho.bin"))
coarse_C11.tofile(os.path.join(coarse_dir, "C11.bin"))
coarse_C13.tofile(os.path.join(coarse_dir, "C13.bin"))
coarse_C33.tofile(os.path.join(coarse_dir, "C33.bin"))
coarse_C55.tofile(os.path.join(coarse_dir, "C55.bin"))
coarse_tau.tofile(os.path.join(coarse_dir, "tau.bin"))
coarse_inv_taus.tofile(os.path.join(coarse_dir, "inv_tau_sigma.bin"))

# ========== 生成细网格模型 ==========
fine_list = []
for idx, region in enumerate(fine_regions):
    xstart = region["x_start"]
    xend = region["x_end"]
    zstart = region["z_start"]
    zend = region["z_end"]
    N = region["N"]

    lenx = (xend - xstart) * N + 1
    lenz = (zend - zstart) * N + 1

    # 创建该区域对应的子目录
    region_dir = os.path.join(fine_dir, str(idx))
    os.makedirs(region_dir, exist_ok=True)
    
    # 细网格模型（均匀介质，与粗网格相同）
    fine_rho = np.full((lenz, lenx), rho_2, dtype=np.float32)
    fine_C11 = np.full((lenz, lenx), C11_2, dtype=np.float32)
    fine_C13 = np.full((lenz, lenx), C13_2, dtype=np.float32)
    fine_C33 = np.full((lenz, lenx), C33_2, dtype=np.float32)
    fine_C55 = np.full((lenz, lenx), C55_2, dtype=np.float32)
    fine_tau = np.full((lenz, lenx), 0, dtype=np.float32)
    fine_inv_taus = np.full((lenz, lenx), 0, dtype=np.float32)
    fine_MAT = np.full((lenz, lenx), SOLID, dtype=np.int32)


        
    # fine_MAT[200 : 800 : 40, 200 : 800] = FLUID
    # fine_MAT[201 : 800 : 40, 200 : 800] = FLUID
    # fine_MAT[202 : 800 : 40, 200 : 800] = FLUID
    # fine_MAT[203 : 800 : 40, 200 : 800] = FLUID
    # fine_MAT[204 : 800 : 40, 200 : 800] = FLUID
    # fine_MAT[205 : 800 : 40, 200 : 800] = FLUID
    # fine_MAT[206 : 800 : 40, 200 : 800] = FLUID
    # fine_MAT[207 : 800 : 40, 200 : 800] = FLUID
    # fine_MAT[208 : 800 : 40, 200 : 800] = FLUID
    # fine_MAT[209 : 800 : 40, 200 : 800] = FLUID
    fine_MAT[10 : fine_MAT.shape[0] - 10 : 10, 10 : fine_MAT.shape[1] - 10 : 10] = FLUID
    fine_MAT[11 : fine_MAT.shape[0] - 10 : 10, 10 : fine_MAT.shape[1] - 10 : 10] = FLUID
    fine_MAT[10 : fine_MAT.shape[0] - 10 : 10, 11 : fine_MAT.shape[1] - 10 : 10] = FLUID

    fine_rho[fine_MAT == FLUID] = rho_fluid
    fine_C11[fine_MAT == FLUID] = C11_fluid
    fine_C13[fine_MAT == FLUID] = C13_fluid
    fine_C33[fine_MAT == FLUID] = C33_fluid
    fine_C55[fine_MAT == FLUID] = C55_fluid
    fine_tau[fine_MAT == FLUID] = tau
    fine_inv_taus[fine_MAT == FLUID] = inv_ts

    # 保存文件
    fine_rho.tofile(os.path.join(region_dir, "rho.bin"))
    fine_C11.tofile(os.path.join(region_dir, "C11.bin"))
    fine_C13.tofile(os.path.join(region_dir, "C13.bin"))
    fine_C33.tofile(os.path.join(region_dir, "C33.bin"))
    fine_C55.tofile(os.path.join(region_dir, "C55.bin"))
    fine_tau.tofile(os.path.join(region_dir, "tau.bin"))
    fine_inv_taus.tofile(os.path.join(region_dir, "inv_tau_sigma.bin"))
    fine_MAT.tofile(os.path.join(region_dir, "material.bin"))
    
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
        "tau": f"models/fine/{idx}/tau.bin",
        "inv_taus": f"models/fine/{idx}/inv_tau_sigma.bin",
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
        "tau": "models/coarse/tau.bin",
        "material": "models/coarse/material.bin",
        "inv_taus": "models/coarse/inv_tau_sigma.bin"
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
