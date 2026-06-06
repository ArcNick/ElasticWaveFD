import numpy as np
import matplotlib.pyplot as plt
import sls
import os
import json
import schoenberg as sch
from scipy.ndimage import zoom
SOLID = 0
VESOLID = 1
FLUID = 2
MULTISHOT = False
def main():
    # 基本参数
    nz = 706
    nx = 691
    dx = 1.5
    dz = 1.5

    fpeak = 30.0
    dt = 5e-5
    nt = 20000
    snapshot = 500

    # 读入速度模型
    input_vp = "Vp_brother.bin"
    vp = np.fromfile(input_vp, dtype=np.float32)
    vp = vp.reshape((nz, nx))
    vp_max = vp.max()
    vs = vp / 1.83

    epsilon = 0
    delta = 0

    # 刚度参数
    coarse_rho = 280 * np.power(vp, 0.265)
    coarse_C33 = coarse_rho * vp**2
    coarse_C55 = coarse_rho * vs**2
    coarse_C11 = coarse_C33 * (1 + 2 * epsilon)
    coarse_C13 = ((coarse_C33 - coarse_C55) * (2 * coarse_C33 * delta + (coarse_C33 - coarse_C55)))**0.5 - coarse_C55
    
    # Q 衰减
    Qp = 40
    Qs = 25
    sls_params = sls.get_sls_parameters(Qp, Qs, 3, 1, 100)
    inv_tss = 1 / sls_params["tau_sigmas"]
    taup = sls_params["taup"]
    taus = sls_params["taus"]

    # 炮点
    basex = nx // 2
    if MULTISHOT:
        posx = [basex - 100, basex - 50, basex, basex + 50, basex + 100]
    else:
        posx = [basex]
    posz = 38

    # CPML
    cpml_thickness = 20
    cpml_N = 3
    cp_max = float(vp_max)
    Rc = 0.0001
    kappa0 = 1.2

    # 创建文件夹
    base_dir = "models"
    coarse_dir = os.path.join(base_dir, "coarse")
    fine_dir = os.path.join(base_dir, "fine")
    os.makedirs(coarse_dir, exist_ok=True)
    os.makedirs(fine_dir, exist_ok=True)

    # 粗网格
    coarse_MAT = np.full((nz, nx), SOLID, dtype=np.int32)
    coarse_zeta = np.full((nz, nx), 0, dtype=np.float32)
    coarse_taup = np.full((nz, nx), 0, dtype=np.float32)
    coarse_taus = np.full((nz, nx), 0, dtype=np.float32)
    coarse_inv_tsig1 = np.full((nz, nx), 0, dtype=np.float32)
    coarse_inv_tsig2 = np.full((nz, nx), 0, dtype=np.float32)
    coarse_inv_tsig3 = np.full((nz, nx), 0, dtype=np.float32)
    coarse_zeta = np.full((nz, nx), 0, dtype=np.float32)

    targets = [5500, 5550, 5580, 5630, 5750]
    coarse_MAT[np.isin(np.round(vp).astype(int), targets)] = VESOLID
    coarse_MAT[-25:, :] = SOLID
    coarse_taup[coarse_MAT == VESOLID] = taup
    coarse_taus[coarse_MAT == VESOLID] = taus
    coarse_inv_tsig1[coarse_MAT == VESOLID] = inv_tss[0]
    coarse_inv_tsig2[coarse_MAT == VESOLID] = inv_tss[1]
    coarse_inv_tsig3[coarse_MAT == VESOLID] = inv_tss[2]
    
    visualize(vp)
    # 细网格
    fine_regions = [
        # {
        #     "x_start": 300, "x_end": 340,
        #     "z_start": 500, "z_end": 540,
        #     "N": 5
        # }
    ]
    border = 5
    phi_target = 0.1
    sigmax, sigmaz = 2, 2
    seed = 12345

    coarse_dict = {
        'rho': coarse_rho,
        'C11': coarse_C11,
        'C13': coarse_C13,
        'C33': coarse_C33,
        'C55': coarse_C55,
        'taup': coarse_taup,
        'taus': coarse_taus,
        'inv_tsig1': coarse_inv_tsig1,
        'inv_tsig2': coarse_inv_tsig2,
        'inv_tsig3': coarse_inv_tsig3,
        'zeta': coarse_zeta,
        'MAT': coarse_MAT,
    }

    fine_list = []
    for idx, region in enumerate(fine_regions):
        x_start = region["x_start"]
        x_end = region["x_end"]
        z_start = region["z_start"]
        z_end = region["z_end"]
        N = region["N"]
    
        lenx = (x_end - x_start) * N + 1
        lenz = (z_end - z_start) * N + 1

        fine = build_fine_grid_from_coarse(
            coarse_dict,
            region["x_start"], region["x_end"],
            region["z_start"], region["z_end"],
            region["N"]
        )

        fine_rho = fine['rho']
        fine_C11 = fine['C11']
        fine_C13 = fine['C13']
        fine_C33 = fine['C33']
        fine_C55 = fine['C55']
        fine_taup = fine['taup']
        fine_taus = fine['taus']
        fine_inv_tsig1 = fine['inv_tsig1']
        fine_inv_tsig2 = fine['inv_tsig2']
        fine_inv_tsig3 = fine['inv_tsig3']
        fine_zeta = fine['zeta']
        fine_MAT = fine['MAT']

        region_dir = os.path.join(fine_dir, str(idx))
        os.makedirs(region_dir, exist_ok=True)

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

        visualize(fine_rho, nz=lenz, nx=lenx, dx=dx/N, dz=dz/N, save_path=os.path.join(region_dir, "rho.png"))

        fine_list.append({
            "x_start": x_start,
            "x_end": x_end,
            "z_start": z_start,
            "z_end": z_end,
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

    with open(os.path.join(base_dir, "models.json"), "w") as f:
        json.dump(models_config, f, indent=2)
    print("models.json 已生成")

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

    

def visualize(data, nz=706, nx=691, dx=1.5, dz=1.5, cmap='jet', save_path=None):
    plt.figure(figsize=(12, 9))
    extent = [0, nx, nz, 0]
    im = plt.imshow(data, cmap=cmap, aspect='auto', extent=extent)
    
    cbar = plt.colorbar(im)
    cbar.set_label('Velocity (m/s)', fontsize=12)
    
    plt.title(f'Velocity Model ', fontsize=14, pad=20)
    plt.xlabel('Horizontal Distance (m)', fontsize=12)
    plt.ylabel('Depth (m)', fontsize=12)
    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().xaxis.set_label_position('top')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"图像已保存至: {save_path}")
    
    plt.show()

def build_fine_grid_from_coarse(
    coarse_arrays,        # dict，包含所有粗网格属性
    x_start, x_end,
    z_start, z_end,
    N
):
    """
    参数:
        coarse_arrays: dict, 必须包含键:
            'rho', 'C11', 'C13', 'C33', 'C55',
            'taup', 'taus', 'inv_tsig1', 'inv_tsig2', 'inv_tsig3',
            'zeta', 'MAT'
        x_start, x_end: 粗网格 x 索引起止（包含两端）
        z_start, z_end: 粗网格 z 索引起止（包含两端）
        N: 加密倍数
    返回:
        dict: 细网格的所有属性，键名与 coarse_arrays 一致
    """
    # 细网格尺寸
    lenx = (x_end - x_start) * N + 1
    lenz = (z_end - z_start) * N + 1

    sub = {}
    for key, arr in coarse_arrays.items():
        sub[key] = arr[z_start:z_end+1, x_start:x_end+1]

    zoom_z = lenz / sub['rho'].shape[0]
    zoom_x = lenx / sub['rho'].shape[1]

    fine = {}

    float_keys = [
        'rho', 'C11', 'C13', 'C33', 'C55', 'taup', 'taus', 
        'inv_tsig1', 'inv_tsig2', 'inv_tsig3', 'zeta'
    ]
    for key in float_keys:
        fine[key] = zoom(sub[key], (zoom_z, zoom_x), order=1)

    fine['MAT'] = zoom(sub['MAT'], (zoom_z, zoom_x), order=0).astype(np.int32)

    return fine

if __name__ == '__main__':
    main()