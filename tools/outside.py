import numpy as np
import matplotlib.pyplot as plt
import sls
import os
import json
import schoenberg as sch
SOLID = 0
VESOLID = 1
FLUID = 2

def visualize_vp(vp, nz=706, nx=690, dx=1.5, dz=1.5, cmap='jet', save_path=None):
    """
    可视化 Vp 模型二进制文件。
    
    参数:
    - file_path: bin 文件路径
    - nz, nx: 垂直和水平方向的采样点数
    - dx, dz: 空间网格间距 (米)
    - cmap: 颜色映射方案 (默认 'jet')
    - save_path: 如果提供路径，则将图片保存到该位置
    """
    # 创建画布
    plt.figure(figsize=(12, 7))
    extent = [0, nx, nz, 0]
    
    im = plt.imshow(vp, cmap=cmap, aspect='auto', extent=extent)
    
    cbar = plt.colorbar(im)
    cbar.set_label('Velocity (m/s)', fontsize=12)
    
    plt.title(f'Vp Model', fontsize=14, pad=20)
    plt.xlabel('Horizontal Distance (m)', fontsize=12)
    plt.ylabel('Depth (m)', fontsize=12)
    
    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().xaxis.set_label_position('top')
    
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"图像已保存至: {save_path}")
    
    plt.show()

# --- 参数设置 ---
nz, nx = 400, 900  # 原始尺寸
dx = 1.25
dz = 1.25

# 时间参数
fpeak = 25.0
dt = 5e-5
nt = 25000
snapshot = 200

input_vp = "vp_cyz.bin"
input_vs = "vs_cyz.bin"
input_rho = "rho_cyz.bin"
vp = np.fromfile(input_vp, dtype=np.float32)
vp = vp.reshape((nz, nx))
vp_max = vp.max()

vs = np.fromfile(input_vs, dtype=np.float32)
vs = vs.reshape((nz, nx))
rho = np.fromfile(input_rho, dtype=np.float32)
rho = rho.reshape((nz, nx))

epsilon_1 = 0.0
delta_1 = 0.0
coarse_C33 = rho * vp**2
coarse_C55 = rho * vs**2
coarse_C11 = coarse_C33 * (1 + 2 * epsilon_1)
coarse_C13 = ((coarse_C33 - coarse_C55) * (2 * coarse_C33 * delta_1 + (coarse_C33 - coarse_C55)))**0.5 - coarse_C55

# 震源位置
posx = nx // 2
posz = 40

# CPML参数
cpml_thickness = 20
cpml_N = 3
cp_max = float(vp_max)
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

visualize_vp(vs)

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
rho.tofile(os.path.join(coarse_dir, "rho.bin"))
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