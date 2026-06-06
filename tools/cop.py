import os
import json
import shutil
import multiprocessing
import concurrent.futures

import numpy as np
from scipy.ndimage import zoom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['figure.max_open_warning'] = 50


# =============================================================================
#                               全局配置与开关
# =============================================================================
GAIN                = 1
CROP_PML_HALO       = True      # 是否裁剪 PML 和 Halo 区域
SHOW_FINE_GRIDS     = True      # 是否显示细网格
FINE_GRID_INDICES   = [0]       # 仅显示指定的细网格块（设为 None 则显示所有）
NORMALIZE_PER_TRACE = False     # 是否每道缩放到全局最大值


# =============================================================================
#                               颜色与范围设定
# =============================================================================
VX_VZ_RANGE = (-4e-8, 4e-8)
SX_SZ_RANGE = (-4e-1, 4e-1)
TXZ_RANGE   = (-2e-1, 2e-1)

FIELD_NAMES = ['vx', 'vz', 'sx', 'sz', 'txz']

FIELD_RANGE = {
    'vx':  VX_VZ_RANGE,
    'vz':  VX_VZ_RANGE,
    'sx':  SX_SZ_RANGE,
    'sz':  SX_SZ_RANGE,
    'txz': TXZ_RANGE
}

FIELD_CMAP = {f: 'seismic' for f in FIELD_NAMES}


# =============================================================================
#                               目录与路径设置
# =============================================================================
OUTPUT_BASE = './output'
IMAGE_BASE  = './images'
MODEL_JSON  = './models/models.json'
PARAMS_JSON = './models/params.json'
RECORD_DIR  = os.path.join(OUTPUT_BASE, 'record')


# =============================================================================
#                               辅助工具函数
# =============================================================================
def ensure_dirs():
    """初始化并清理输出图像目录"""
    if os.path.exists(IMAGE_BASE):
        for item in os.listdir(IMAGE_BASE):
            p = os.path.join(IMAGE_BASE, item)
            if os.path.isfile(p):
                os.remove(p)
            else:
                shutil.rmtree(p)
    else:
        os.makedirs(IMAGE_BASE)
        
    for f in FIELD_NAMES + ['record']:
        os.makedirs(os.path.join(IMAGE_BASE, f), exist_ok=True)

def load_json(fp):
    """读取 JSON 文件"""
    with open(fp, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_cpu_count():
    """保留两个核心，防止电脑卡死"""
    return max(1, multiprocessing.cpu_count() - 2)

def print_progress(msg):
    print(msg, flush=True)


# =============================================================================
#                               波场与网格处理
# =============================================================================
HALO = {
    'full':    {'left': 3, 'right': 4, 'top': 3, 'bottom': 4},
    'half_x':  {'left': 4, 'right': 3, 'top': 3, 'bottom': 4},
    'half_z':  {'left': 3, 'right': 4, 'top': 4, 'bottom': 3},
    'half_xz': {'left': 4, 'right': 3, 'top': 4, 'bottom': 3}
}

def get_coarse_dims_by_type(nx_full, nz_full, pml, ftype):
    """根据场分量类型，计算粗网格维度和物理区域切片"""
    nx_phys = nx_full - 2 * pml - 7
    nz_phys = nz_full - 2 * pml - 7
    
    if ftype == 'full':
        phys_nx, phys_nz = nx_phys, nz_phys
        nx_coarse, nz_coarse = nx_full, nz_full
    elif ftype == 'half_x':
        phys_nx, phys_nz = nx_phys + 1, nz_phys
        nx_coarse = phys_nx + 2 * (pml - 1) + 7
        nz_coarse = nz_phys + 2 * pml + 7
    elif ftype == 'half_z':
        phys_nx, phys_nz = nx_phys, nz_phys + 1
        nx_coarse = nx_phys + 2 * pml + 7
        nz_coarse = phys_nz + 2 * (pml - 1) + 7
    elif ftype == 'half_xz':
        phys_nx, phys_nz = nx_phys + 1, nz_phys + 1
        nx_coarse = phys_nx + 2 * (pml - 1) + 7
        nz_coarse = phys_nz + 2 * (pml - 1) + 7
    else:
        raise ValueError('Unknown field type')
        
    if CROP_PML_HALO:
        h = HALO[ftype]
        left   = pml + h['left']
        right  = nx_coarse - (pml + h['right'])
        top    = pml + h['top']
        bottom = nz_coarse - (pml + h['bottom'])
        phys_slice = (slice(top, bottom), slice(left, right))
    else:
        phys_slice = (slice(0, nz_coarse), slice(0, nx_coarse))
        phys_nx, phys_nz = nx_coarse, nz_coarse
        
    return nx_coarse, nz_coarse, phys_slice, phys_nx, phys_nz

def build_full_snapshot(fname, fpath, model, pml):
    """重构全波场快照（包含细网格插值拼装）"""
    cgrid = model['coarse']
    fine_list = model.get('fine', [])
    dx, dz = cgrid['dx'], cgrid['dz']
    
    ftype = {'txz': 'half_xz', 'vx': 'half_x', 'vz': 'half_z'}.get(fname, 'full')
    nx_c, nz_c, sl, phys_nx, phys_nz = get_coarse_dims_by_type(
        cgrid['nx'], cgrid['nz'], pml, ftype
    )
    
    data = np.fromfile(fpath, dtype=np.float32) * GAIN
    if len(data) < nx_c * nz_c:
        print_progress(f'错误: {fpath} 粗网格不足，跳过')
        return None
    
    # 提取并复制粗网格物理区域
    coarse = data[:nx_c * nz_c].reshape((nz_c, nx_c))[sl]
    full = coarse.copy()
    
    if not SHOW_FINE_GRIDS:
        return full
    
    # 开始拼接细网格
    offset = nx_c * nz_c
    for idx, fine in enumerate(fine_list):
        N  = fine['N']
        x0, z0 = fine['x_start'], fine['z_start']
        x1, z1 = fine['x_end'],   fine['z_end']
        
        # 确定细网格物理区域大小
        if ftype == 'half_xz':
            nxf, nzf = (x1 - x0) * N,     (z1 - z0) * N
        elif ftype == 'half_x':
            nxf, nzf = (x1 - x0) * N,     (z1 - z0) * N + 1
        elif ftype == 'half_z':
            nxf, nzf = (x1 - x0) * N + 1, (z1 - z0) * N
        else:
            nxf, nzf = (x1 - x0) * N + 1, (z1 - z0) * N + 1
            
        block = nxf * nzf
        if offset + block > len(data):
            print_progress(f'错误: {fpath} 细网格块 {idx} 数据不足')
            exit(1)

        # 仅处理指定的细网格
        if FINE_GRID_INDICES is None or idx in FINE_GRID_INDICES:
            fine_data = data[offset : offset+block].reshape((nzf, nxf))
            
            # 确定插入到粗网格中的物理坐标
            if CROP_PML_HALO:
                hl, ht = HALO[ftype]['left'], HALO[ftype]['top']
                i0, i1 = x0 - (pml + hl), x1 - (pml + hl) + 1
                j0, j1 = z0 - (pml + ht), z1 - (pml + ht) + 1
            else:
                i0, i1 = x0, x1 + 1
                j0, j1 = z0, z1 + 1
                
            # 边界检查
            if i0 < 0 or i1 > phys_nx or j0 < 0 or j1 > phys_nz:
                print_progress(f'错误: 细网格 {idx} 插入区 [{j0}:{j1}, {i0}:{i1}] 越界 [{phys_nz}, {phys_nx}]')
                exit(1)

            target_h, target_w = j1 - j0, i1 - i0
            
            # 将细网格插值回粗网格分辨率
            zoom_y = target_h / fine_data.shape[0]
            zoom_x = target_w / fine_data.shape[1]
            fine_rs = zoom(fine_data, (zoom_y, zoom_x), order=1)
            
            # 强制对齐尺寸
            if fine_rs.shape[0] != target_h or fine_rs.shape[1] != target_w:
                fine_rs = fine_rs[:target_h, :target_w]
                
            full[j0:j1, i0:i1] = fine_rs

        offset += block
        
    return full


# =============================================================================
#                               绘图模块
# =============================================================================
def plot_snapshot(fname, fbase, arr, cgrid):
    """绘制并保存波场快照图"""
    dx, dz = cgrid['dx'], cgrid['dz']
    nz, nx = arr.shape
    
    fig, ax = plt.subplots(figsize=(12, nz / nx * 12))
    im = ax.imshow(
        arr, 
        cmap='seismic',
        vmin=FIELD_RANGE[fname][0], 
        vmax=FIELD_RANGE[fname][1],
        extent=[-dx * (nx // 2), dx * (nx - nx // 2), dz * nz, 0],
        aspect='equal'
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax.set_title(f'{fname} : {fbase}')
    ax.set_xlabel('offset (m)', fontsize=14)
    ax.set_ylabel('z (m)', fontsize=14)
    fig.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.05)
    
    out_png = os.path.join(IMAGE_BASE, fname, fbase.replace('.bin', '.png'))
    fig.savefig(out_png, dpi=80, bbox_inches=None)
    plt.close(fig)
    print_progress(f'已保存 {out_png}')


def plot_record_cmap(bin_path, out_png, nx_total, dt, dx, pml, 
                     ftype='full', skip=1, gain=1.0, out_dt=0.001, fixed_limit=2.0):
    """绘制并保存地震记录（VZ道集）"""
    rec = np.fromfile(bin_path, dtype=np.float32)
    nt = len(rec) // nx_total
    
    if len(rec) % nx_total != 0:
        print_progress(f'错误: {bin_path} 大小 {len(rec)} 不是 nx_total={nx_total} 整数倍')
        return
        
    rec = rec.reshape((nt, nx_total))
    
    # 裁剪
    if CROP_PML_HALO:
        h = HALO[ftype]
        left_idx  = pml + h['left']
        right_idx = nx_total - (pml + h['right'])
        rec = rec[:, left_idx:right_idx]

    # 时间降采样
    step = max(1, int(round(out_dt / dt)))
    rec = rec[::step, :]
    nt_new = rec.shape[0]
    
    # 空间抽道
    rec = rec[:, ::skip]
    n_trace = rec.shape[1]
    
    if n_trace == 0:
        print_progress(f'警告: {bin_path} 降采样后无有效道')
        return
        
    # 应用缩放
    amx = 1
    rec_scaled = (rec * dx / amx) if amx > 0 else rec
        
    print_progress(f'  应用 gain={gain:.4e} 后数据范围: [{np.min(rec_scaled):.4e}, {np.max(rec_scaled):.4e}]')
    
    vmin, vmax = -fixed_limit, fixed_limit
    print_progress(f'  固定颜色范围: [{vmin}, {vmax}]')
    
    # 坐标系换算
    temp = np.arange(n_trace) * skip
    x_coords = (temp - temp[-1] // 2 - 1) * dx
    z_coords = np.arange(nt_new) * out_dt
    extent = [x_coords[0], x_coords[-1], z_coords[-1], z_coords[0]]
    
    # 开始绘图
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(rec_scaled, aspect='auto', cmap='seismic', extent=extent, 
                   origin='upper', vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax)
    
    ax.set_title('vz')
    ax.set_xlabel('offset (m)')
    ax.set_ylabel('time (s)')
    
    fig.savefig(out_png, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print_progress(f'地震记录图已保存: {out_png}')


# =============================================================================
#                               并行任务入口
# =============================================================================
def process_snapshot(field, fname, model, pml, idx, total):
    """单个快照文件的处理线程目标函数"""
    fpath = os.path.join(OUTPUT_BASE, field, fname)
    arr = build_full_snapshot(field, fpath, model, pml)
    
    if arr is not None:
        plot_snapshot(field, fname, arr, model['coarse'])


# =============================================================================
#                               主 函 数
# =============================================================================
def main():
    ensure_dirs()
    
    model  = load_json(MODEL_JSON)
    params = load_json(PARAMS_JSON)
    
    pml      = params['cpml']['thickness']
    shot_num = len(params['base']['posx'])
    dt       = 0.001  # 固定的采样间隔
    
    print_progress(f'使用固定采样间隔 dt = {dt} s')

    # ---------------------------------------------------------
    # 1. 串行处理地震记录 (VZ道集)
    # ---------------------------------------------------------
    if os.path.exists(RECORD_DIR):
        vz_files = []
        
        # 扫描包含炮号的子目录 (0, 1, 2...)
        for shot_id in range(shot_num):
            shot_dir = os.path.join(RECORD_DIR, str(shot_id))
            if os.path.isdir(shot_dir):
                for fn in os.listdir(shot_dir):
                    if fn.endswith('.bin') and 'vz' in fn:
                        rel_path = os.path.join(str(shot_id), fn).replace('\\', '/')
                        vz_files.append(rel_path)
        
        # 兜底：如果根目录下直接散落了文件
        if not vz_files:
            for fn in os.listdir(RECORD_DIR):
                if fn.endswith('.bin') and 'vz' in fn:
                    vz_files.append(fn)

        print_progress(f'\n处理 {len(vz_files)} 个 vz 地震记录...')
        
        for idx, f in enumerate(vz_files, 1):
            nx_tot  = model['coarse']['nx']
            out_png = os.path.join(IMAGE_BASE, 'record', f.replace('.bin', '_cmap.png'))
            
            # 确保对应的图像输出子目录存在
            os.makedirs(os.path.dirname(out_png), exist_ok=True)
            
            plot_record_cmap(
                bin_path=os.path.join(RECORD_DIR, f),
                out_png=out_png,
                nx_total=nx_tot, 
                dt=dt, 
                dx=model['coarse']['dx'], 
                pml=pml,
                ftype='half_z', 
                skip=1, 
                out_dt=0.001, 
                fixed_limit=2e-8
            )

    # ---------------------------------------------------------
    # 2. 并行处理波场快照
    # ---------------------------------------------------------
    tasks = []
    
    for field in FIELD_NAMES:
        dir_path = os.path.join(OUTPUT_BASE, field)
        
        if not os.path.isdir(dir_path):
            continue
            
        # 扫描每个物理量的炮号子目录
        for shot_id in range(shot_num):
            shot_dir = os.path.join(dir_path, str(shot_id))
            
            if not os.path.isdir(shot_dir):
                continue
                
            # 提前创建对应的出图目录
            os.makedirs(os.path.join(IMAGE_BASE, field, str(shot_id)), exist_ok=True)
            
            for fn in os.listdir(shot_dir):
                if fn.endswith('.bin'):
                    rel_fname = os.path.join(str(shot_id), fn).replace('\\', '/')
                    tasks.append((field, rel_fname))
    
    total_tasks = len(tasks)
    if total_tasks == 0:
        print_progress("\n没有找到需要处理的波场快照数据。")
        return
        
    print_progress(f'\n并行处理 {total_tasks} 个波场快照，线程数 {get_cpu_count()}')
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=get_cpu_count()) as exe:
        # 提交所有任务并构建 Future 字典
        futures = {
            exe.submit(process_snapshot, field, fname, model, pml, i, total_tasks): (field, fname) 
            for i, (field, fname) in enumerate(tasks, 1)
        }
        
        # 收集结果
        for f in concurrent.futures.as_completed(futures):
            try:
                f.result()
            except Exception as e:
                task_info = futures[f]
                print_progress(f'处理失败: {task_info} -> {e}')
                
    print_progress('\n全部任务处理完成！')

if __name__ == '__main__':
    main()