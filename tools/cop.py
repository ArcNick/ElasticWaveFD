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

# ========== 开关 ==========
CROP_PML_HALO = True          # 裁剪 PML/halo
SHOW_FINE_GRIDS = True
FINE_GRID_INDICES = [0]        # 只显示第一个细网格块（设为 None 显示所有）
NORMALIZE_PER_TRACE = True     # 每道缩放到全局最大值

# ========== 颜色范围 ==========
VX_VZ_RANGE = (-2e-9, 2e-9)
SX_SZ_RANGE = (-2e-2, 2e-2)
TXZ_RANGE   = (-1e-2, 1e-2)
FIELD_NAMES = ['vx','vz','sx','sz','txz']
FIELD_RANGE = {'vx':VX_VZ_RANGE,'vz':VX_VZ_RANGE,'sx':SX_SZ_RANGE,
               'sz':SX_SZ_RANGE,'txz':TXZ_RANGE}
FIELD_CMAP  = {f:'seismic' for f in FIELD_NAMES}

# ========== 路径 ==========
OUTPUT_BASE = './output'
IMAGE_BASE  = './images'
MODEL_JSON  = './models/models.json'
PARAMS_JSON = './models/params.json'
RECORD_DIR  = os.path.join(OUTPUT_BASE, 'record')

# ========== 辅助 ==========
def ensure_dirs():
    if os.path.exists(IMAGE_BASE):
        for item in os.listdir(IMAGE_BASE):
            p = os.path.join(IMAGE_BASE, item)
            os.remove(p) if os.path.isfile(p) else shutil.rmtree(p)
    else:
        os.makedirs(IMAGE_BASE)
    for f in FIELD_NAMES + ['record']:
        os.makedirs(os.path.join(IMAGE_BASE, f), exist_ok=True)

def load_json(fp):
    with open(fp) as f: return json.load(f)

def get_cpu_count(): return max(1, multiprocessing.cpu_count() - 2)

def print_progress(msg): print(msg, flush=True)

# ========== 波场处理 ==========
HALO = {
    'full':   {'left':3,'right':4,'top':3,'bottom':4},
    'half_x': {'left':4,'right':3,'top':3,'bottom':4},
    'half_z': {'left':3,'right':4,'top':4,'bottom':3},
    'half_xz':{'left':4,'right':3,'top':4,'bottom':3}
}

def get_coarse_dims_by_type(nx_full, nz_full, pml, ftype):
    nx_phys = nx_full - 2 * pml - 7
    nz_phys = nz_full - 2 * pml - 7
    if ftype == 'full':
        phys_nx, phys_nz = nx_phys, nz_phys
        nx_coarse, nz_coarse = nx_full, nz_full
    elif ftype == 'half_x':
        phys_nx, phys_nz = nx_phys+1, nz_phys
        nx_coarse = phys_nx + 2 * (pml - 1) + 7
        nz_coarse = nz_phys + 2 * pml + 7
    elif ftype == 'half_z':
        phys_nx, phys_nz = nx_phys, nz_phys+1
        nx_coarse = nx_phys + 2 * pml + 7
        nz_coarse = phys_nz + 2 * (pml - 1) + 7
    elif ftype == 'half_xz':
        phys_nx, phys_nz = nx_phys+1, nz_phys+1
        nx_coarse = phys_nx + 2 * (pml - 1) + 7
        nz_coarse = phys_nz + 2 * (pml - 1) + 7
    else:
        raise ValueError('Unknown field type')
    if CROP_PML_HALO:
        h = HALO[ftype]
        left, right = pml + h['left'], nx_coarse - (pml + h['right'])
        top, bottom = pml + h['top'], nz_coarse - (pml + h['bottom'])
        phys_slice = (slice(top,bottom), slice(left,right))
    else:
        phys_slice = (slice(0,nz_coarse), slice(0,nx_coarse))
        phys_nx, phys_nz = nx_coarse, nz_coarse
    return nx_coarse, nz_coarse, phys_slice, phys_nx, phys_nz

def build_full_snapshot(fname, fpath, model, pml):
    cgrid = model['coarse']
    fine_list = model.get('fine', [])
    dx, dz = cgrid['dx'], cgrid['dz']
    ftype = {'txz': 'half_xz', 'vx':'half_x', 'vz':'half_z'}.get(fname,'full')
    nx_c, nz_c, sl, phys_nx, phys_nz = get_coarse_dims_by_type(
        cgrid['nx'], cgrid['nz'], pml, ftype
    )
    data = np.fromfile(fpath, dtype=np.float32)
    if len(data) < nx_c * nz_c:
        print_progress(f'错误: {fpath} 粗网格不足，跳过')
        return None
    
    coarse = data[:nx_c * nz_c].reshape((nz_c, nx_c))[sl]
    full = coarse.copy()
    if not SHOW_FINE_GRIDS:
        return full
    
    offset = nx_c * nz_c
    for idx, fine in enumerate(fine_list):
        N = fine['N']
        x0 = fine['x_start']
        z0 = fine['z_start']
        x1 = fine['x_end']
        z1 = fine['z_end']
        # 细网格物理区域大小（无 halo）
        if ftype == 'half_xz':
            nxf, nzf = (x1 - x0) * N, (z1 - z0) * N
        elif ftype == 'half_x':
            nxf, nzf = (x1 - x0) * N, (z1 - z0) * N + 1
        elif ftype == 'half_z':
            nxf, nzf = (x1 - x0) * N + 1, (z1 - z0) * N
        else:
            nxf, nzf = (x1 - x0) * N + 1, (z1 - z0) * N + 1
        block = nxf * nzf
        if offset + block > len(data):
            print_progress(f'error: {fpath} 细网格块 {idx} 数据不足')
            exit(1)

        if FINE_GRID_INDICES is None or idx in FINE_GRID_INDICES:
            fine_data = data[offset:offset+block].reshape((nzf, nxf))
            # 插入位置（在粗网格物理区域中的索引）
            if CROP_PML_HALO:
                hl, ht = HALO[ftype]['left'], HALO[ftype]['top']
                i0 = x0 - (pml + hl)
                i1 = x1 - (pml + hl) + 1
                j0 = z0 - (pml + ht)
                j1 = z1 - (pml + ht) + 1
            else:
                i0 = x0
                i1 = x1 + 1
                j0 = z0
                j1 = z1 + 1
            # 边界检查
            if i0 < 0 or i1 > phys_nx or j0 < 0 or j1 > phys_nz:
                print_progress(f'error: 细网格块 {idx} 插入区域 [{j0}:{j1}, {i0}:{i1}] 超出物理区域 [{phys_nz},{phys_nx}]')
                exit(1)

            target_h, target_w = j1 - j0, i1 - i0
            # 插值到粗网格分辨率
            zoom_y = target_h / fine_data.shape[0]
            zoom_x = target_w / fine_data.shape[1]
            fine_rs = zoom(fine_data, (zoom_y, zoom_x), order=1)
            # 确保尺寸一致
            if fine_rs.shape[0] != target_h or fine_rs.shape[1] != target_w:
                fine_rs = fine_rs[:target_h, :target_w]
            full[j0:j1, i0:i1] = fine_rs

        offset += block
    return full

def plot_snapshot(fname, fbase, arr, cgrid):
    dx, dz = cgrid['dx'], cgrid['dz']
    nz, nx = arr.shape
    fig, ax = plt.subplots(figsize=(10, 10))
    # 固定坐标轴在 figure 中的位置（左,下,宽,高），比例为 0.8，留出边距
    ax.set_position([0.1, 0.1, 0.8, 0.8])
    im = ax.imshow(
        arr, cmap='seismic',
        vmin=FIELD_RANGE[fname][0], 
        vmax=FIELD_RANGE[fname][1],
        extent=[0, dx * nx, dz * nz, 0],
        aspect='equal'
    )
    # 添加颜色条，调整大小和间距
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f'{fname} : {fbase}')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    out = os.path.join(IMAGE_BASE, fname, fbase.replace('.bin', '.png'))
    fig.savefig(out, dpi=300, bbox_inches=None)
    plt.close(fig)
    print_progress(f'已保存 {out}')

# ========== 地震记录绘图 ==========
def wigb(a, scal=1.0, x=None, z=None, amx=None):
    nz,nx = a.shape
    if x is None: x = np.arange(1,nx+1)
    if z is None: z = np.arange(1,nz+1)
    x, z = np.asarray(x), np.asarray(z)
    if nx <= 1: return
    dx = np.median(np.diff(x))
    dz = z[1] - z[0]
    if amx is None: amx = np.max(np.abs(a))
    if np.isscalar(amx):
        a = a * dx / amx * scal
    else:
        a_norm = np.zeros_like(a)
        for i in range(nx):
            if amx[i] != 0: a_norm[:, i] = a[:, i] * dx / amx[i] * scal
        a = a_norm
    ax = plt.gca()
    ax.set_xlim(np.min(x)-2*dx, np.max(x)+2*dx)
    ax.set_ylim(np.max(z)+dz, np.min(z)-dz)
    for i in range(nx):
        tr = a[:, i]
        if np.max(np.abs(tr))!=0:
            s = np.sign(tr)
            idx = np.where(s[:-1]!=s[1:])[0]
            zadd = idx+1 + tr[idx]/(tr[idx]-tr[idx+1]) if len(idx)>0 else []
            zpos = np.where(tr>0)[0]+1
            apos = tr[tr>0]
            if len(zpos) == 0 and len(zadd) == 0:
                ax.plot(tr+x[i], z, 'k', lw=0.1)
                ax.axvline(x[i], color='white', lw=0.1)
                continue
            z_all = np.concatenate((zpos, zadd))
            a_all = np.concatenate((apos, np.zeros_like(zadd)))
            sort = np.argsort(z_all)
            zz, aa = z_all[sort], a_all[sort]
            a0 = 0.0; 
            z0 = 1.0 if tr[0] <= 0 else (zadd[0] if len(zadd) > 0 else 1.0)
            a1 = 0.0
            z1 = float(nz) if tr[-1] >= 0 else (zadd[-1] if len(zadd) > 0 else float(nz))
            zz = np.concatenate(([z0], zz, [z1], [z0]))
            aa = np.concatenate(([a0], aa, [a1], [a0]))
            ax.fill(aa + x[i], z[0] + (zz-1)*dz, 'k', lw=0)
            ax.axvline(x[i], color='white', lw=0.1)
            ax.plot(tr + x[i], z, 'k', lw=0.1)
        else:
            ax.axvline(x[i], color='red', lw=0.1)
    plt.draw()

def plot_record_wigb(bin_path, out_png, nx_total, dt, dx, pml, ftype='full', skip=1, gain=1.0, out_dt=0.0005):
    rec = np.fromfile(bin_path, dtype=np.float32)
    nt = len(rec) // nx_total
    if len(rec) % nx_total != 0:
        print_progress(f'错误: {bin_path} 大小 {len(rec)} 不是 nx_total={nx_total} 整数倍')
        return
    rec = rec.reshape((nt, nx_total))
    if CROP_PML_HALO:
        h = HALO[ftype]
        l, r = pml + h['left'], nx_total - (pml + h['right'])
        rec = rec[:, l:r]
        start_trace = l
    else:
        start_trace = 0
    # 时间降采样到 out_dt
    step = max(1, int(round(out_dt / dt)))
    rec = rec[::step, :]
    nt_new = rec.shape[0]
    # 道降采样
    rec = rec[:, ::skip]
    n_trace = rec.shape[1]
    if n_trace == 0:
        print_progress(f'警告: {bin_path} 降采样后无有效道')
        return
    # 归一化
    if NORMALIZE_PER_TRACE:
        global_max = np.max(np.abs(rec))
        for i in range(n_trace):
            tm = np.max(np.abs(rec[:,i]))
            if tm>0: rec[:,i] = rec[:,i] / tm * global_max
        wigb_amx = 1.0
    else:
        wigb_amx = None
    # 坐标
    x_coords = (start_trace + np.arange(n_trace) * skip) * dx
    z_coords = np.arange(nt_new) * out_dt
    fig, ax = plt.subplots(figsize=(30,15))
    plt.sca(ax)
    wigb(rec, scal=gain, x=x_coords, z=z_coords, amx=wigb_amx)
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print_progress(f'地震记录已保存: {out_png}')

# ========== 任务处理 ==========
def process_snapshot(field, fname, model, pml, idx, total):
    arr = build_full_snapshot(field, os.path.join(OUTPUT_BASE, field, fname), model, pml)
    if arr is not None:
        plot_snapshot(field, fname, arr, model['coarse'])

# ========== 主程序 ==========
def main():
    ensure_dirs()
    model = load_json(MODEL_JSON)
    params = load_json(PARAMS_JSON)
    pml = params['cpml']['thickness']
    dt = params['base']['dt']
    print_progress(f'原始 dt = {dt:.2e} s, 输出 dt = 0.0005 s (步长 {int(round(0.0005/dt))})')

    # 地震记录（串行）—— 仅处理 vz 文件
    if os.path.exists(RECORD_DIR):
        files = [f for f in sorted(os.listdir(RECORD_DIR)) if f.endswith('.bin')]
        vz_files = [f for f in files if 'vz' in f]
        print_progress(f'\n处理 {len(vz_files)} 个vz地震记录...')
        for idx, f in enumerate(vz_files, 1):
            ftype = 'half_z'
            nx_tot = model['coarse']['nx']   # vz 的道数等于粗网格 nx
            plot_record_wigb(
                os.path.join(RECORD_DIR, f),
                os.path.join(IMAGE_BASE, 'record', f.replace('.bin','.png')),
                nx_tot, dt, model['coarse']['dx'], pml,
                ftype=ftype, skip=5, gain=-4e7, out_dt=0.0005
            )

    # 波场快照（并行）
    tasks = []
    for f in FIELD_NAMES:
        dir_path = os.path.join(OUTPUT_BASE, f)
        if os.path.isdir(dir_path):
            for fn in os.listdir(dir_path):
                if fn.endswith('.bin'):
                    tasks.append((f, fn))
    total = len(tasks)
    if total == 0:
        return
    print_progress(f'\n并行处理 {total} 个波场快照，线程数 {get_cpu_count()}')
    with concurrent.futures.ThreadPoolExecutor(max_workers=get_cpu_count()) as exe:
        fut = {exe.submit(process_snapshot, f, fn, model, pml, i, total):(f,fn) for i,(f,fn) in enumerate(tasks,1)}
        for f in concurrent.futures.as_completed(fut):
            try:
                f.result()
            except Exception as e:
                print_progress(f'出错: {fut[f]} {e}')
    print_progress('完成')

if __name__ == '__main__':
    main()