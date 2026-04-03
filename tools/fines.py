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
CROP_PML_HALO = True          # 裁剪 PML/halo (对细网格物质可视化无影响)
SHOW_FINE_GRIDS = True        # 仅用于波场快照，此处保留但未使用
FINE_GRID_INDICES = [0]       # 仅用于波场快照，此处保留但未使用
NORMALIZE_PER_TRACE = True    # 仅用于地震记录，此处保留但未使用

# ========== 颜色范围（不再使用）==========
# ... 保留原定义，但不再用于绘图

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
    # 为细网格物质可视化创建目录
    os.makedirs(os.path.join(IMAGE_BASE, 'fine'), exist_ok=True)
    # 原有波场和记录目录（不再使用，但保留以防后续需要）
    # for f in FIELD_NAMES + ['record']:
    #     os.makedirs(os.path.join(IMAGE_BASE, f), exist_ok=True)

def load_json(fp):
    with open(fp) as f: return json.load(f)

def get_cpu_count(): return max(1, multiprocessing.cpu_count() - 2)

def print_progress(msg): print(msg, flush=True)

# ========== 细网格物质属性可视化 ==========
def plot_fine_material(fine_idx, block, model):
    """
    绘制细网格块的流体-固体二元图。
    block: 细网格块信息字典
    model: 整体模型信息（包含 coarse 信息，用于坐标换算）
    """
    # 获取加密倍数和坐标范围
    N = block['N']
    x0, x1 = block['x_start'], block['x_end']
    z0, z1 = block['z_start'], block['z_end']
    dx = model['coarse']['dx']
    dz = model['coarse']['dz']

    # 计算细网格物理区域大小（整网格点，因为 rho 和 C55 在整网格上定义）
    nx_fine = (x1 - x0) * N + 1
    nz_fine = (z1 - z0) * N + 1

    # 读取 rho 和 C55 文件
    rho_path = block['rho']
    c55_path = block['C55']
    if not os.path.exists(rho_path) or not os.path.exists(c55_path):
        print_progress(f'警告: 细网格块 {fine_idx} 的物质文件不存在，跳过')
        return

    rho = np.fromfile(rho_path, dtype=np.float32)
    c55 = np.fromfile(c55_path, dtype=np.float32)
    # 验证大小
    if len(rho) != nx_fine * nz_fine or len(c55) != nx_fine * nz_fine:
        print_progress(f'错误: 细网格块 {fine_idx} 文件大小不匹配，期望 {nx_fine*nz_fine}，实际 rho={len(rho)}, C55={len(c55)}')
        return
    rho = rho.reshape((nz_fine, nx_fine))
    c55 = c55.reshape((nz_fine, nx_fine))

    # 判断固体/流体：固体有剪切模量（C55 > 0），流体为0或极小值
    # 设置阈值，考虑数值精度
    eps = 1e-12
    is_solid = c55 > eps

    # 创建图像：固体红色，流体蓝色
    # 使用 RGBA 数组或直接 imshow 用 colormap
    # 这里用两种颜色显示，cmap 可自定义
    # 方法1：将布尔数组转为整型 0/1，然后用 ListedColormap
    import matplotlib.colors as mcolors
    cmap = mcolors.ListedColormap(['blue', 'red'])
    bounds = [0, 0.5, 1]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 8))
    # 注意：imshow 的 extent 需要与物理坐标对应
    # 细网格物理坐标范围：x 从 x0*dx 到 (x1*dx + dx/N * (nx_fine-1)) 即 x1*dx
    x_min = x0 * dx
    x_max = x1 * dx
    z_min = z0 * dz
    z_max = z1 * dz
    extent = [x_min, x_max, z_max, z_min]  # origin='upper' 时，z 从上到下
    im = ax.imshow(is_solid.astype(int), cmap=cmap, norm=norm,
                   extent=extent, origin='upper', interpolation='none')
    ax.set_title(f'Fine Grid {fine_idx}: Solid (red) / Fluid (blue)')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    fig.colorbar(im, ax=ax, ticks=[0.25, 0.75], label='Material')
    # 保存图像
    out_dir = os.path.join(IMAGE_BASE, 'fine')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'fine_{fine_idx}.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print_progress(f'已保存细网格物质图像: {out_path}')

# ========== 主程序 ==========
def main():
    ensure_dirs()
    model = load_json(MODEL_JSON)
    # params 不再需要，但可保留以防后续
    # params = load_json(PARAMS_JSON)
    # pml = params['cpml']['thickness']
    # dt = params['base']['dt']

    # 处理细网格物质可视化
    fine_blocks = model.get('fine', [])
    if not fine_blocks:
        print_progress('未找到细网格块，程序结束。')
        return

    print_progress(f'\n处理 {len(fine_blocks)} 个细网格块的流体-固体可视化...')
    for idx, block in enumerate(fine_blocks):
        print_progress(f'处理细网格块 {idx} ...')
        plot_fine_material(idx, block, model)

    print_progress('所有细网格块处理完成。')

if __name__ == '__main__':
    main()