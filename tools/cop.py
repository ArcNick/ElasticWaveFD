import os
import numpy as np
import json
import concurrent.futures
import multiprocessing
import shutil
from scipy.ndimage import zoom

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['figure.max_open_warning'] = 50

# ==================== 控制开关 ====================
CROP_PML_HALO = False
SHOW_FINE_GRIDS = True
FINE_GRID_INDICES = []
NORMALIZE_PER_TRACE = True   # 默认 False：使用全局最大值归一化（所有道共用同一个最大值）

# ==================== 颜色范围常量 ====================
VX_VZ_RANGE = (-1e-9, 1e-9)
SX_SZ_RANGE = (-4e-2, 4e-2)
TXZ_RANGE   = (-2e-2, 2e-2)

FIELD_NAMES = ['vx', 'vz', 'sx', 'sz', 'txz']
FIELD_CMAP  = {f:'seismic' for f in FIELD_NAMES}
FIELD_RANGE = {
    'vx': VX_VZ_RANGE,
    'vz': VX_VZ_RANGE,
    'sx': SX_SZ_RANGE,
    'sz': SX_SZ_RANGE,
    'txz': TXZ_RANGE
}

OUTPUT_BASE = './output'
IMAGE_BASE  = './images'
MODEL_JSON = './models/models.json'
PARAMS_JSON = './models/params.json'
RECORD_DIR = os.path.join(OUTPUT_BASE, 'record')

# ---------- 辅助函数 ----------
def ensure_dirs():
    if os.path.exists(IMAGE_BASE):
        for item in os.listdir(IMAGE_BASE):
            item_path = os.path.join(IMAGE_BASE, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
    else:
        os.makedirs(IMAGE_BASE)
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

# ---------- 波场处理函数 ----------
HALO = {
    'full':   {'left': 3, 'right': 4, 'top': 3, 'bottom': 4},
    'half_x': {'left': 4, 'right': 3, 'top': 3, 'bottom': 4},
    'half_z': {'left': 3, 'right': 4, 'top': 4, 'bottom': 3},
    'half_xy':{'left': 4, 'right': 3, 'top': 4, 'bottom': 3}
}

def get_coarse_dims_by_type(nx_full, nz_full, pml, field_type):
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
    if CROP_PML_HALO:
        halo_cfg = HALO[field_type]
        left = pml + halo_cfg['left']
        right = nx_coarse - (pml + halo_cfg['right'])
        top = pml + halo_cfg['top']
        bottom = nz_coarse - (pml + halo_cfg['bottom'])
        phys_slice = (slice(top, bottom), slice(left, right))
    else:
        phys_slice = (slice(0, nz_coarse), slice(0, nx_coarse))
        phys_nx = nx_coarse
        phys_nz = nz_coarse
    return nx_coarse, nz_coarse, phys_slice, phys_nx, phys_nz

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

    if not SHOW_FINE_GRIDS:
        return full_arr

    offset = coarse_size
    for idx, fine in enumerate(fine_list):
        N = fine["N"]
        x0, x1 = fine["x_start"], fine["x_end"]
        z0, z1 = fine["z_start"], fine["z_end"]

        # 细网格有效区尺寸
        if field_type == 'half_xy':
            nx_fine = (x1 - x0) * N
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

        show_this = (FINE_GRID_INDICES is None) or (idx in FINE_GRID_INDICES)

        if show_this:
            fine_data = data[offset:offset + block_size].reshape((nz_fine, nx_fine))

            # --- 起始下标严格修正 ---
            if CROP_PML_HALO:
                halo_left = HALO[field_type]['left']
                halo_top  = HALO[field_type]['top']
                insert_i0 = x0 - (pml_thick + halo_left)
                insert_i1 = x1 - (pml_thick + halo_left) + 1
                insert_j0 = z0 - (pml_thick + halo_top)
                insert_j1 = z1 - (pml_thick + halo_top) + 1
            else:
                insert_i0 = x0
                insert_i1 = x1 + 1
                insert_j0 = z0
                insert_j1 = z1 + 1

            # 边界检查
            if insert_i0 < 0 or insert_i1 > phys_nx or insert_j0 < 0 or insert_j1 > phys_nz:
                print_progress(f"警告: 细网格块 {idx} 插入区域超出边界，跳过")
                offset += block_size
                continue

            target_width  = insert_i1 - insert_i0
            target_height = insert_j1 - insert_j0

            # 细网格区插值到粗网格像素数
            zoom_x = target_width / fine_data.shape[1]
            zoom_y = target_height / fine_data.shape[0]
            fine_resampled = zoom(fine_data, (zoom_y, zoom_x), order=1)

            # 安全防护，必须完全对齐
            actual_h, actual_w = fine_resampled.shape
            if actual_h != target_height or actual_w != target_width:
                fine_resampled = fine_resampled[:target_height, :target_width]

            full_arr[insert_j0:insert_j1, insert_i0:insert_i1] = fine_resampled

        offset += block_size

    return full_arr

def plot_snapshot(fieldname, filebase, arr, cgrid):
    dx, dz = cgrid["dx"], cgrid["dz"]
    vmin, vmax = FIELD_RANGE[fieldname]
    cmap = FIELD_CMAP[fieldname]
    extent = [0, dx * arr.shape[1], dz * arr.shape[0], 0]
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent, interpolation='none', aspect='auto')
    ax.set_aspect('equal')
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

# ---------- 地震记录绘图函数 wigb（支持 per-trace 归一化）----------
def wigb(a, scal=None, x=None, z=None, amx=None):
    """
    WIGB: Plot seismic data using wiggles.
    amx : float or 1D array, optional
        - If float, use global maximum for all traces.
        - If 1D array of length nx, each trace is normalized by its own maximum (per-trace).
        - If None, global maximum is used.
    """
    nz, nx = a.shape
    trmx = np.max(np.abs(a), axis=0)

    if x is None:
        x = np.arange(1, nx + 1)
    if z is None:
        z = np.arange(1, nz + 1)
    if scal is None:
        scal = 1.0

    # 处理 amx
    if amx is None:
        amx = np.max(np.abs(a))          # 全局最大值
        per_trace = False
    elif np.isscalar(amx):
        per_trace = False                # 全局标量
    else:
        amx = np.asarray(amx)
        if amx.shape[0] != nx:
            raise ValueError(f"amx array must have length nx ({nx}), got {amx.shape[0]}")
        per_trace = True                  # 每道单独最大值

    x = np.asarray(x)
    z = np.asarray(z)

    if nx <= 1:
        print(' ERR: wigb: nx has to be more than 1')
        return

    dx = np.median(np.diff(x))
    dz = z[1] - z[0]

    xmx = np.max(a)
    xmn = np.min(a)

    if scal == 0:
        scal = 1.0

    # 应用归一化
    if per_trace:
        a_norm = np.zeros_like(a)
        for i in range(nx):
            if amx[i] != 0:
                a_norm[:, i] = a[:, i] * dx / amx[i] * scal
            else:
                a_norm[:, i] = 0.0
        a = a_norm
        print(f' PlotWig (per-trace): data range [{xmn:.4f}, {xmx:.4f}]')
    else:
        a = a * dx / amx * scal
        print(f' PlotWig (global): data range [{xmn:.4f}, {xmx:.4f}], using amx={amx:.4f}')

    x1 = np.min(x) - 2.0 * dx
    x2 = np.max(x) + 2.0 * dx
    z1 = np.min(z) - dz
    z2 = np.max(z) + dz

    ax = plt.gca()
    ax.set_xlim(x1, x2)
    ax.set_ylim(z2, z1)
    ax.set_box_aspect(None)
    ax.set_title('Wiggle plot')
    ax.set_xlabel('Offset')
    ax.set_ylabel('Time/Depth')

    fillcolor = 'k'
    linecolor = 'k'
    linewidth = 0.1

    zstart = z[0]
    zend = z[-1]

    for i in range(nx):
        if trmx[i] != 0:
            tr = a[:, i]
            s = np.sign(tr)
            idx_change = np.where(s[:-1] != s[1:])[0]
            if len(idx_change) > 0:
                zadd = idx_change + 1 + tr[idx_change] / (tr[idx_change] - tr[idx_change + 1])
            else:
                zadd = np.array([])

            zpos = np.where(tr > 0)[0] + 1
            apos = tr[tr > 0]

            if len(zpos) == 0 and len(zadd) == 0:
                ax.plot(tr + x[i], z, color=linecolor, linewidth=linewidth)
                ax.plot([x[i], x[i]], [zstart, zend], color='white', linewidth=linewidth)
                continue

            z_all = np.concatenate((zpos, zadd))
            a_all = np.concatenate((apos, np.zeros_like(zadd)))
            sort_idx = np.argsort(z_all)
            zz = z_all[sort_idx]
            aa = a_all[sort_idx]

            if tr[0] > 0:
                a0 = 0.0
                z0 = 1.0
            else:
                a0 = 0.0
                z0 = zadd[0] if len(zadd) > 0 else 1.0

            if tr[-1] > 0:
                a1 = 0.0
                z1 = float(nz)
            else:
                a1 = 0.0
                z1 = zadd[-1] if len(zadd) > 0 else float(nz)

            zz = np.concatenate(([z0], zz, [z1], [z0]))
            aa = np.concatenate(([a0], aa, [a1], [a0]))

            zzz = zstart + (zz - 1) * dz

            ax.fill(aa + x[i], zzz, color=fillcolor, linewidth=0)
            ax.plot([x[i], x[i]], [zstart, zend], color='white', linewidth=linewidth)
            ax.plot(tr + x[i], z, color=linecolor, linewidth=linewidth)
        else:
            ax.plot([x[i], x[i]], [zstart, zend], color='red', linewidth=linewidth)

    plt.draw()

def plot_record_wigb(record_bin, out_png, nx_total, dt, dx, pml_thick, field_type='full', skip=4, gain=1.0, output_dt=0.001):
    """
    绘制地震记录（支持不同分量）
    nx_total: 该分量粗网格总点数（包含PML和halo）
    field_type: 分量类型，用于裁剪时使用正确的halo
    """
    rec = np.fromfile(record_bin, dtype=np.float32)
    nt = rec.size // nx_total
    if rec.size % nx_total != 0:
        print_progress(f"警告: {record_bin} 数据大小不是 nx_total={nx_total} 的整数倍，丢弃多余数据")
        rec = rec[:nt*nx_total]
    rec_mat = rec.reshape((nt, nx_total))

    # 根据 CROP_PML_HALO 和场量类型裁剪道
    if CROP_PML_HALO:
        halo_cfg = HALO[field_type]
        left_cut = pml_thick + halo_cfg['left']
        right_cut = pml_thick + halo_cfg['right']
        start_trace = left_cut
        end_trace = nx_total - right_cut
        if start_trace >= end_trace:
            print_progress("警告：裁剪后无有效道，跳过裁剪")
            start_trace = 0
            end_trace = nx_total
        rec_mat = rec_mat[:, start_trace:end_trace]
        print_progress(f"  裁剪道：保留 {start_trace} 到 {end_trace-1} (原 nx_total={nx_total})")
    else:
        start_trace = 0
        end_trace = nx_total

    # 时间降采样
    if output_dt <= 0:
        raise ValueError("output_dt 必须为正数")
    step_t = int(round(output_dt / dt))
    if abs(step_t * dt - output_dt) > 1e-12:
        print_progress(f"警告: output_dt={output_dt}s 不是原始 dt={dt}s 的整数倍，将使用 step_t={step_t}，实际输出间隔={step_t*dt:.6f}s")
    rec_mat = rec_mat[::step_t, :]
    nt_new = rec_mat.shape[0]

    # 道降采样
    rec_mat = rec_mat[:, ::skip]
    n_trace_plot = rec_mat.shape[1]

    # 物理坐标
    x_coords = (start_trace + np.arange(n_trace_plot) * skip) * dx
    z_coords = np.arange(nt_new) * output_dt

    # 根据 NORMALIZE_PER_TRACE 决定 amx 参数
    if NORMALIZE_PER_TRACE:
        amx = np.max(np.abs(rec_mat), axis=0)   # 每道最大值
    else:
        amx = None                               # 使用全局最大值

    fig, ax = plt.subplots(figsize=(30, 15))
    plt.sca(ax)
    wigb(rec_mat, scal=gain, x=x_coords, z=z_coords, amx=amx)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print_progress(f"地震记录已保存: {out_png}")
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

    mode_str = "裁剪 PML 和 halo" if CROP_PML_HALO else "显示完整区域（包含 PML 和 halo）"
    print_progress(f"当前模式: {mode_str}")
    fine_mode_str = f"显示细网格: {SHOW_FINE_GRIDS}"
    if SHOW_FINE_GRIDS:
        if FINE_GRID_INDICES is None:
            fine_mode_str += " (所有细网格)"
        else:
            fine_mode_str += f" (索引列表: {FINE_GRID_INDICES})"
    else:
        fine_mode_str += " (不显示细网格)"
    print_progress(f"当前细网格设置: {fine_mode_str}")
    if NORMALIZE_PER_TRACE:
        print_progress("地震记录归一化: 每道单独归一化")
    else:
        print_progress("地震记录归一化: 全局归一化（所有道共用同一个最大值）")

    # ---------- 处理地震记录（串行）----------
    record_tasks = []
    if os.path.exists(RECORD_DIR):
        for fname in sorted(os.listdir(RECORD_DIR)):
            if fname.endswith('.bin'):
                record_tasks.append(fname)

    if record_tasks:
        print_progress(f"\n开始串行处理地震记录，共 {len(record_tasks)} 个文件...")
        for idx, fname in enumerate(record_tasks, start=1):
            # 根据文件名推断分量类型
            if 'vx' in fname:
                field_type = 'half_x'
            elif 'vz' in fname:
                field_type = 'half_z'
            elif 'sx' in fname or 'sz' in fname:
                field_type = 'full'
            elif 'txz' in fname:
                field_type = 'half_xy'
            else:
                field_type = 'full'   # 默认

            # 计算该分量的粗网格总点数
            nx_total, _, _, _, _ = get_coarse_dims_by_type(
                modelinfo["coarse"]["nx"], modelinfo["coarse"]["nz"], pml_thick, field_type)
            
            print_progress(f"处理记录 [{idx}/{len(record_tasks)}]: {fname} (分量类型: {field_type})")
            plot_record_wigb(
                os.path.join(RECORD_DIR, fname),
                os.path.join(IMAGE_BASE, 'record', fname.replace('.bin', '.png')),
                nx_total,
                paramsinfo["base"]["dt"],
                modelinfo["coarse"]["dx"],
                pml_thick,
                field_type=field_type,
                skip=5,
                gain=1.0,
                output_dt=0.001
            )
        print_progress("地震记录处理完成。\n")
    else:
        print_progress("未找到地震记录文件，跳过。")

    # ---------- 并行处理波场快照 ----------
    snapshot_tasks = []
    for fieldname in FIELD_NAMES:
        field_dir = os.path.join(OUTPUT_BASE, fieldname)
        if not os.path.isdir(field_dir):
            continue
        for fname in sorted(os.listdir(field_dir)):
            if fname.endswith('.bin'):
                snapshot_tasks.append((fieldname, fname))

    total = len(snapshot_tasks)
    if total == 0:
        print_progress("未找到波场快照文件，程序结束。")
        return

    print_progress(f"开始并行处理波场快照，共 {total} 个文件，使用 {get_cpu_count()} 线程...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=get_cpu_count()) as exe:
        futures = {}
        for idx, (fieldname, fname) in enumerate(snapshot_tasks, start=1):
            future = exe.submit(process_snapshot, fieldname, fname, modelinfo, pml_thick, idx, total)
            futures[future] = (fieldname, fname)

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                fieldname, fname = futures[future]
                print_progress(f"处理 {fieldname}/{fname} 时出错: {e}")

    print_progress("所有波场快照处理完成。")

if __name__ == "__main__":
    main()