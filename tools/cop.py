import os
import numpy as np
import json
import concurrent.futures
import multiprocessing
import shutil
from scipy.ndimage import zoom

# 必须在导入 pyplot 前设置后端
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 可选：提高 figure 打开数量警告阈值
plt.rcParams['figure.max_open_warning'] = 50

# ==================== 颜色范围常量 ====================
VX_VZ_RANGE = (-1e-7, 1e-7)          # 速度分量
SX_SZ_RANGE = (-5e-1, 5e-1)          # 正应力
TXZ_RANGE   = (-2e-1, 2e-1)          # 切应力

FIELD_NAMES = ['vx', 'vz', 'sx', 'sz', 'txz']
FIELD_CMAP  = {f:'seismic' for f in FIELD_NAMES}
FIELD_RANGE = {
    'vx': VX_VZ_RANGE,
    'vz': VX_VZ_RANGE,
    'sx': SX_SZ_RANGE,
    'sz': SX_SZ_RANGE,
    'txz': TXZ_RANGE
}

# ==================== 固定路径 ====================
OUTPUT_BASE = './output'
IMAGE_BASE  = './images'
MODEL_JSON = './models/models.json'
PARAMS_JSON = './models/params.json'
RECORD_DIR = os.path.join(OUTPUT_BASE, 'record')

# ==================== 辅助函数 ====================
def ensure_dirs():
    # 清空 IMAGE_BASE 目录（如果存在）
    if os.path.exists(IMAGE_BASE):
        for item in os.listdir(IMAGE_BASE):
            item_path = os.path.join(IMAGE_BASE, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
    else:
        os.makedirs(IMAGE_BASE)

    # 创建各场量子目录
    for field in FIELD_NAMES:
        os.makedirs(os.path.join(IMAGE_BASE, field), exist_ok=True)
    os.makedirs(os.path.join(IMAGE_BASE, "record"), exist_ok=True)

def load_json(fp):
    with open(fp, 'r') as f:
        return json.load(f)

def get_cpu_count():
    return max(1, multiprocessing.cpu_count() - 2)

def print_progress(msg):
    print(msg, flush=True)

# ---------- 根据场量类型计算粗网格总点数和物理区域切片 ----------
# halo 定义（八阶差分）
HALO = {
    'full':   {'left': 3, 'right': 4, 'top': 3, 'bottom': 4},
    'half_x': {'left': 4, 'right': 3, 'top': 3, 'bottom': 4},
    'half_z': {'left': 3, 'right': 4, 'top': 4, 'bottom': 3},
    'half_xy':{'left': 4, 'right': 3, 'top': 4, 'bottom': 3}
}

def get_coarse_dims_by_type(nx_full, nz_full, pml, field_type):
    """
    根据整网格总点数（nx_full, nz_full）、PML厚度和场量类型，
    计算该场量的粗网格总点数、物理区域切片及物理区域大小。
    返回 (nx_coarse, nz_coarse, phys_slice, phys_nx, phys_nz)
    """
    # 整网格物理区域大小（不含PML和halo）
    nx_phys = nx_full - 2*pml - 7
    nz_phys = nz_full - 2*pml - 7

    if field_type == 'full':
        phys_nx = nx_phys
        phys_nz = nz_phys
        nx_coarse = nx_full
        nz_coarse = nz_full
    elif field_type == 'half_x':
        phys_nx = nx_phys + 1
        phys_nz = nz_phys
        nx_coarse = phys_nx + 2*(pml-1) + 7
        nz_coarse = nz_phys + 2*pml + 7
    elif field_type == 'half_z':
        phys_nx = nx_phys
        phys_nz = nz_phys + 1
        nx_coarse = nx_phys + 2*pml + 7
        nz_coarse = phys_nz + 2*(pml-1) + 7
    elif field_type == 'half_xy':
        phys_nx = nx_phys + 1
        phys_nz = nz_phys + 1
        nx_coarse = phys_nx + 2*(pml-1) + 7
        nz_coarse = phys_nz + 2*(pml-1) + 7
    else:
        raise ValueError(f"Unknown field type: {field_type}")

    # 物理区域切片
    halo_cfg = HALO[field_type]
    left = pml + halo_cfg['left']
    right = nx_coarse - (pml + halo_cfg['right'])
    top = pml + halo_cfg['top']
    bottom = nz_coarse - (pml + halo_cfg['bottom'])
    phys_slice = (slice(top, bottom), slice(left, right))

    return nx_coarse, nz_coarse, phys_slice, phys_nx, phys_nz

# ---------- 构建完整波场快照 ----------
# def build_full_snapshot(fieldname, filepath, modelinfo, pml_thick):
#     cgrid = modelinfo["coarse"]
#     fine_list = modelinfo.get("fine", [])
#     dx, dz = cgrid["dx"], cgrid["dz"]

#     # 确定场量类型
#     if fieldname == 'txz':
#         field_type = 'half_xy'
#     elif fieldname == 'vx':
#         field_type = 'half_x'
#     elif fieldname == 'vz':
#         field_type = 'half_z'
#     else:
#         field_type = 'full'

#     # 获取该场量的粗网格总点数及物理切片
#     nx_coarse, nz_coarse, phys_slice, phys_nx, phys_nz = get_coarse_dims_by_type(
#         cgrid["nx"], cgrid["nz"], pml_thick, field_type)
#     coarse_size = nx_coarse * nz_coarse

#     # 读取整个文件
#     data = np.fromfile(filepath, dtype=np.float32)
#     total_elem = len(data)

#     if total_elem < coarse_size:
#         print_progress(f"错误: {filepath} 需要至少 {coarse_size} 个元素（粗网格），实际 {total_elem}，跳过")
#         return None

#     # 提取粗网格数据并裁剪物理区域
#     coarse_data = data[:coarse_size].reshape((nz_coarse, nx_coarse))
#     coarse_phys = coarse_data[phys_slice]  # (phys_nz, phys_nx)

#     # 初始化完整数组（物理区域大小）
#     full_arr = coarse_phys.copy()

#     # 处理细网格块
#     offset = coarse_size
#     for idx, fine in enumerate(fine_list):
#         N = fine["N"]
#         x0, x1 = fine["x_start"], fine["x_end"]
#         z0, z1 = fine["z_start"], fine["z_end"]

#         # 计算细网格物理区域大小（无halo）
#         if field_type == 'half_xy':
#             nx_fine = (x1 - x0) * N
#             nz_fine = (z1 - z0) * N
#         elif field_type == 'half_x':
#             nx_fine = (x1 - x0) * N
#             nz_fine = (z1 - z0) * N + 1
#         elif field_type == 'half_z':
#             nx_fine = (x1 - x0) * N + 1
#             nz_fine = (z1 - z0) * N
#         else:  # full
#             nx_fine = (x1 - x0) * N + 1
#             nz_fine = (z1 - z0) * N + 1

#         block_size = nx_fine * nz_fine
#         remain = total_elem - offset
#         if remain < block_size:
#             print_progress(f"错误: {filepath} 细网格块 {idx} 需要 {block_size} 个元素，剩余 {remain}，跳过后续")
#             break

#         fine_data = data[offset:offset+block_size].reshape((nz_fine, nx_fine))

#         # 计算细网格块在粗网格物理区域中的起始索引
#         start_i = x0 - (pml_thick + HALO[field_type]['left'])
#         start_j = z0 - (pml_thick + HALO[field_type]['top'])

#         # 边界检查
#         if start_i < 0 or start_j < 0 or start_i >= phys_nx or start_j >= phys_nz:
#             offset += block_size
#             continue

#         # 细网格步长
#         fine_dx = dx / N
#         fine_dz = dz / N

#         # 重采样到粗网格分辨率
#         fine_resampled = zoom(fine_data, (dz/fine_dz, dx/fine_dx), order=1)
#         rh, rw = fine_resampled.shape

#         # 插入区域
#         i0 = start_i
#         i1 = min(start_i + rw, phys_nx)
#         j0 = start_j
#         j1 = min(start_j + rh, phys_nz)

#         full_arr[j0:j1, i0:i1] = fine_resampled[0:(j1-j0), 0:(i1-i0)]
#         offset += block_size

#     return full_arr
def build_full_snapshot(fieldname, filepath, modelinfo, pml_thick):
    cgrid = modelinfo["coarse"]
    fine_list = modelinfo.get("fine", [])
    dx, dz = cgrid["dx"], cgrid["dz"]

    # 场量类型
    if fieldname == 'txz':
        field_type = 'half_xy'
    elif fieldname == 'vx':
        field_type = 'half_x'
    elif fieldname == 'vz':
        field_type = 'half_z'
    else:
        field_type = 'full'

    nx_coarse, nz_coarse, phys_slice, phys_nx, phys_nz = get_coarse_dims_by_type(
        cgrid["nx"], cgrid["nz"], pml_thick, field_type)
    coarse_size = nx_coarse * nz_coarse

    data = np.fromfile(filepath, dtype=np.float32)
    if len(data) < coarse_size:
        print_progress(f"错误: {filepath} 需要至少 {coarse_size} 个元素（粗网格），实际 {len(data)}，跳过")
        return None

    coarse_data = data[:coarse_size].reshape((nz_coarse, nx_coarse))
    coarse_phys = coarse_data[phys_slice]

    full_arr = coarse_phys.copy()

    offset = coarse_size
    for idx, fine in enumerate(fine_list):
        N = fine["N"]
        x0, x1 = fine["x_start"], fine["x_end"]
        z0, z1 = fine["z_start"], fine["z_end"]

        # 细网格有效区尺寸
        if field_type == 'half_xy':
            nx_fine = (x1 - x0) * N     # 注意：边界是闭区间
            nz_fine = (z1 - z0) * N
        elif field_type == 'half_x':
            nx_fine = (x1 - x0) * N
            nz_fine = (z1 - z0) * N + 1
        elif field_type == 'half_z':
            nx_fine = (x1 - x0) * N + 1
            nz_fine = (z1 - z0) * N
        else:
            nx_fine = (x1 - x0) * N + 1
            nz_fine = (z1 - z0) * N + 1

        block_size = nx_fine * nz_fine
        remain = len(data) - offset
        if remain < block_size:
            print_progress(f"细网格块 {idx} 长度不足, 跳过")
            break

        fine_data = data[offset:offset + block_size].reshape((nz_fine, nx_fine))

        # --- 起始下标严格修正 ---
        # halo, pml 修正: 有效区（粗网格裁剪后）起点 = x0 - (pml + halo[left])、z0同理
        if field_type == 'half_xy':
            halo_left = HALO[field_type]['left']
            halo_top  = HALO[field_type]['top']
        elif field_type == 'half_x':
            halo_left = HALO[field_type]['left']
            halo_top  = HALO[field_type]['top']
        elif field_type == 'half_z':
            halo_left = HALO[field_type]['left']
            halo_top  = HALO[field_type]['top']
        else:
            halo_left = HALO[field_type]['left']
            halo_top  = HALO[field_type]['top']

        # 粗网格有效区目标插入索引
        insert_i0 = x0 - (pml_thick + halo_left)
        insert_i1 = x1 - (pml_thick + halo_left) + 1  # 因为右闭包

        insert_j0 = z0 - (pml_thick + halo_top)
        insert_j1 = z1 - (pml_thick + halo_top) + 1

        # 粗网格有效区像素宽度
        target_width  = insert_i1 - insert_i0
        target_height = insert_j1 - insert_j0

        # 细网格区插值到粗网格像素数
        zoom_x = target_width / fine_data.shape[1]
        zoom_y = target_height / fine_data.shape[0]
        fine_resampled = zoom(fine_data, (zoom_y, zoom_x), order=1)

        # 安全防护，必须完全对齐
        actual_h, actual_w = fine_resampled.shape
        if actual_h != target_height or actual_w != target_width:
            # 防止插值取整偏差
           fine_resampled = fine_resampled[:target_height, :target_width]

        full_arr[insert_j0:insert_j1, insert_i0:insert_i1] = fine_resampled

    return full_arr
def plot_snapshot(fieldname, filebase, arr, cgrid):
    dx, dz = cgrid["dx"], cgrid["dz"]
    vmin, vmax = FIELD_RANGE[fieldname]
    cmap = FIELD_CMAP[fieldname]
    extent = [0, dx * arr.shape[1], dz * arr.shape[0], 0]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, interpolation='none', aspect='auto')
    ax.set_aspect('equal')  # 确保物理比例正确
    fig.colorbar(im, ax=ax)
    ax.set_title(f"{fieldname}  :  {filebase}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    fig.tight_layout()
    
    outdir = os.path.join(IMAGE_BASE, fieldname)
    os.makedirs(outdir, exist_ok=True)
    savepath = os.path.join(outdir, filebase.replace('.bin', '.png'))
    fig.savefig(savepath, dpi=300)
    plt.close(fig)
    return savepath

# ---------- 地震记录绘图（wigb）----------
def wigb(data, scale=1.0, skip=1, xstep=1, gain=1.0, lwidth=0.3, color='k'):
    nt, nx = data.shape
    t = np.arange(nt)
    x = np.arange(nx) * xstep
    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(0, nx, skip):
        trace = data[:, i] * gain * scale
        ax.plot(x[i] + trace, t, color, linewidth=lwidth)
    ax.invert_yaxis()
    ax.set_xlabel("x (m)")
    ax.set_ylabel("time (s)")
    ax.set_title("Seismogram (wigb)")
    fig.tight_layout()
    return fig, ax

def plot_record_wigb(record_bin, out_png, nx, dt, dx, skip=4, gain=1.0, lwidth=0.3):
    rec = np.fromfile(record_bin, dtype=np.float32)
    nt = rec.size // nx
    if rec.size % nx != 0:
        return None
    rec_mat = rec.reshape((nt, nx))
    fig, ax = wigb(rec_mat, scale=1.0, skip=skip, xstep=dx, gain=gain, lwidth=lwidth, color='k')
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return out_png

# ---------- 处理单个波场文件 ----------
def process_snapshot(fieldname, binfile, modelinfo, pml_thick, task_idx, total_tasks):
    filepath = os.path.join(OUTPUT_BASE, fieldname, binfile)
    cgrid = modelinfo["coarse"]
    arr = build_full_snapshot(fieldname, filepath, modelinfo, pml_thick)
    if arr is not None:
        savepath = plot_snapshot(fieldname, binfile, arr, cgrid)
        print_progress(f"已保存 {savepath} [{task_idx}/{total_tasks}]")

# ---------- 主程序 ----------
def main():
    ensure_dirs()
    modelinfo = load_json(MODEL_JSON)
    paramsinfo = load_json(PARAMS_JSON)
    pml_thick = paramsinfo["cpml"]["thickness"]

    # 收集所有波场任务
    tasks = []
    for fieldname in FIELD_NAMES:
        field_dir = os.path.join(OUTPUT_BASE, fieldname)
        if not os.path.isdir(field_dir):
            continue
        for fname in sorted(os.listdir(field_dir)):
            if fname.endswith('.bin'):
                tasks.append((fieldname, fname))

    # 可选：添加地震记录任务
    # if os.path.exists(RECORD_DIR):
    #     for fname in sorted(os.listdir(RECORD_DIR)):
    #         if fname.endswith('.bin'):
    #             tasks.append(('record', fname))

    total = len(tasks)
    print_progress(f"共发现 {total} 个文件需要处理")

    with concurrent.futures.ThreadPoolExecutor(max_workers=get_cpu_count()) as exe:
        futures = {}
        for idx, (fieldname, fname) in enumerate(tasks, start=1):
            if fieldname == 'record':
                future = exe.submit(plot_record_wigb,
                                     os.path.join(RECORD_DIR, fname),
                                     os.path.join(IMAGE_BASE, 'record', fname.replace('.bin', '.png')),
                                     modelinfo["coarse"]["nx"],
                                     paramsinfo["base"]["dt"],
                                     modelinfo["coarse"]["dx"],
                                     skip=5, gain=1.0, lwidth=0.3)
            else:
                future = exe.submit(process_snapshot, fieldname, fname, modelinfo, pml_thick, idx, total)
            futures[future] = (fieldname, fname)

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                fieldname, fname = futures[future]
                print_progress(f"处理 {fieldname}/{fname} 时出错: {e}")

    print_progress("所有任务处理完成")

if __name__ == "__main__":
    main()