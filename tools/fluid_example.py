import os
import json
import numpy as np

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
dt = 1e-4
nt = 16000
snapshot = 1000

# 介质参数（均匀介质）
# 固体区域
rho_solid = 2500.0
vp_solid = 3600.0
vs_solid = 2000.0
C11_solid = rho_solid * vp_solid**2
C55_solid = rho_solid * vs_solid**2
C13_solid = C11_solid - 2 * C55_solid
C33_solid = C11_solid

# 流体区
rho_fluid = 1000.0
vp_fluid = 1400.0
vs_fluid = 0.0
C11_fluid = rho_fluid * vp_fluid**2
C55_fluid = rho_fluid * vs_fluid**2
C13_fluid = C11_fluid - 2 * C55_fluid
C33_fluid = C11_fluid
omega0 = 2 * np.pi * fpeak
Qp = 80
inv_ts = omega0 * Qp
tau = 1 / Qp * (1 + 1 / Qp**2)**0.5

# 细网格区域（粗网格坐标）
fine_regions = [
    # {
    #     "x_start": 100, "x_end": 150,
    #     "z_start": 150, "z_end": 200,
    #     "N": 3
    # }
]

# 震源位置
posx = nx // 2
posz = nz // 4

# CPML参数
cpml_thickness = 20
cpml_N = 3
cp_max = vp_solid
Rc = 0.001
kappa0 = 1.5

# ========== 创建目录结构 ==========
base_dir = "models"
coarse_dir = os.path.join(base_dir, "coarse")
fine_dir = os.path.join(base_dir, "fine")
os.makedirs(coarse_dir, exist_ok=True)
os.makedirs(fine_dir, exist_ok=True)

# ========== 生成粗网格模型 ==========
coarse_MAT = np.full((nz, nx), SOLID, dtype=np.int32)
coarse_rho = np.full((nz, nx), rho_solid, dtype=np.float32)
coarse_C11 = np.full((nz, nx), C11_solid, dtype=np.float32)
coarse_C13 = np.full((nz, nx), C13_solid, dtype=np.float32)
coarse_C33 = np.full((nz, nx), C33_solid, dtype=np.float32)
coarse_C55 = np.full((nz, nx), C55_solid, dtype=np.float32)
coarse_tau = np.full((nz, nx), 0, dtype=np.float32)
coarse_inv_taus = np.full((nz, nx), 0, dtype=np.float32)

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
    fine_rho = np.full((lenz, lenx), rho_fluid, dtype=np.float32)
    fine_C11 = np.full((lenz, lenx), C11_fluid, dtype=np.float32)
    fine_C13 = np.full((lenz, lenx), C13_fluid, dtype=np.float32)
    fine_C33 = np.full((lenz, lenx), C33_fluid, dtype=np.float32)
    fine_C55 = np.full((lenz, lenx), C55_fluid, dtype=np.float32)
    
    # 保存文件
    fine_rho.tofile(os.path.join(region_dir, "rho.bin"))
    fine_C11.tofile(os.path.join(region_dir, "C11.bin"))
    fine_C13.tofile(os.path.join(region_dir, "C13.bin"))
    fine_C33.tofile(os.path.join(region_dir, "C33.bin"))
    fine_C55.tofile(os.path.join(region_dir, "C55.bin"))
    
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
        "C55": f"models/fine/{idx}/C55.bin"
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
