import os
import json
import numpy as np

# ==================== 参数设置 ====================
# 粗网格尺寸
nx = 500
nz = 500
dx = 1.0          # 网格步长（米）
dz = 1.0

# 介质参数（各向同性均匀）
rho = 2000.0       # 密度 kg/m³
vp = 3000.0        # P波速度 m/s
vs = 1500.0        # S波速度 m/s
# 弹性常数（根据各向同性关系计算）
C11 = rho * vp**2
C55 = rho * vs**2
C13 = C11 - 2 * C55
C33 = C11

# 细网格区域（粗网格坐标，确保在范围内）
fine_regions = [
    {
        "x_start": 420, "x_end": 427,   # x方向范围
        "z_start": 420, "z_end": 427,   # z方向范围
        "N": 3                           # 加密倍数
    }
]

# 震源位置（中心）
posx = nx // 2
posz = nz // 2

# 时间参数（示例值，可根据需要调整）
fpeak = 20.0       # 主频 Hz
dt = 0.00005         # 时间步长 s
nt = 20000          # 总时间步数
snapshot = 200     # 快照间隔

# CPML参数（标准值）
cpml_thickness = 10
cpml_N = 2
cp_max = vp        # 最大纵波速度
Rc = 0.001
kappa0 = 1.0

# ==================== 创建目录结构 ====================
base_dir = "models"
coarse_dir = os.path.join(base_dir, "coarse")
fine_dir = os.path.join(base_dir, "fine")
os.makedirs(coarse_dir, exist_ok=True)
os.makedirs(fine_dir, exist_ok=True)

# ==================== 生成粗网格模型 ====================
print("生成粗网格模型...")
coarse_rho = np.full((nz, nx), rho, dtype=np.float32)
coarse_C11 = np.full((nz, nx), C11, dtype=np.float32)
coarse_C13 = np.full((nz, nx), C13, dtype=np.float32)
coarse_C33 = np.full((nz, nx), C33, dtype=np.float32)
coarse_C55 = np.full((nz, nx), C55, dtype=np.float32)

coarse_rho.tofile(os.path.join(coarse_dir, "rho.bin"))
coarse_C11.tofile(os.path.join(coarse_dir, "C11.bin"))
coarse_C13.tofile(os.path.join(coarse_dir, "C13.bin"))
coarse_C33.tofile(os.path.join(coarse_dir, "C33.bin"))
coarse_C55.tofile(os.path.join(coarse_dir, "C55.bin"))

# ==================== 生成细网格模型 ====================
fine_list = []
for idx, region in enumerate(fine_regions):
    print(f"生成细网格区域 {idx}...")
    xs = region["x_start"]
    xe = region["x_end"]
    zs = region["z_start"]
    ze = region["z_end"]
    N = region["N"]
    
    lenx = (xe - xs) * N + 1
    lenz = (ze - zs) * N + 1
    
    # 创建该区域对应的子目录
    region_dir = os.path.join(fine_dir, str(idx))
    os.makedirs(region_dir, exist_ok=True)
    
    # 细网格模型（均匀介质，与粗网格相同）
    fine_rho = np.full((lenz, lenx), rho, dtype=np.float32)
    fine_C11 = np.full((lenz, lenx), C11, dtype=np.float32)
    fine_C13 = np.full((lenz, lenx), C13, dtype=np.float32)
    fine_C33 = np.full((lenz, lenx), C33, dtype=np.float32)
    fine_C55 = np.full((lenz, lenx), C55, dtype=np.float32)
    
    # 保存文件
    fine_rho.tofile(os.path.join(region_dir, "rho.bin"))
    fine_C11.tofile(os.path.join(region_dir, "C11.bin"))
    fine_C13.tofile(os.path.join(region_dir, "C13.bin"))
    fine_C33.tofile(os.path.join(region_dir, "C33.bin"))
    fine_C55.tofile(os.path.join(region_dir, "C55.bin"))
    
    # 记录JSON信息（使用相对路径）
    fine_list.append({
        "x_start": xs,
        "x_end": xe,
        "z_start": zs,
        "z_end": ze,
        "N": N,
        "rho": f"models/fine/{idx}/rho.bin",
        "C11": f"models/fine/{idx}/C11.bin",
        "C13": f"models/fine/{idx}/C13.bin",
        "C33": f"models/fine/{idx}/C33.bin",
        "C55": f"models/fine/{idx}/C55.bin"
    })

# ==================== 生成 models.json ====================
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
        "C55": "models/coarse/C55.bin"
    },
    "fine": fine_list
}

with open(os.path.join(base_dir, "models.json"), "w") as f:
    json.dump(models_config, f, indent=2)
print("models.json 已生成")

# ==================== 生成 params.json ====================
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
        "kappa0": kappa0,
        "fpeak": fpeak
    }
}

with open(os.path.join(base_dir, "params.json"), "w") as f:
    json.dump(params_config, f, indent=2)
print("params.json 已生成")

print("所有文件生成完毕。")