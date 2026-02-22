#!/usr/bin/env python3
"""
波场快照生成脚本（最终版）
- 使用 seismic 色标
- z 轴向下为正（深度增加）
- 同一场所有图片颜色范围一致
- 多线程处理，线程数 = CPU核心数 - 2
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import shutil
import traceback

# ==================== 配置 ====================
MODEL_JSON = "./models/models.json"
PARAM_JSON = "./models/params.json"
OUTPUT_DIR = "./output"
SNAPSHOT_DIR = "./snapshot"

FIELD_NAMES = ["sx", "sz", "txz", "vx", "vz"]
RECORD_DIR = "record"

# ==================== 辅助函数 ====================
def load_json(path):
    """加载 JSON 文件"""
    with open(path, 'r') as f:
        return json.load(f)

def get_coarse_size(field, nx, nz):
    """粗网格数据量及形状，返回 (元素数, (nz, nx))"""
    if field == "sx" or field == "sz":
        return nx * nz, (nz, nx)
    elif field == "vx":
        return (nx - 1) * nz, (nz, nx - 1)
    elif field == "vz":
        return nx * (nz - 1), (nz - 1, nx)
    elif field == "txz":
        return (nx - 1) * (nz - 1), (nz - 1, nx - 1)
    else:
        raise ValueError(f"Unknown field: {field}")

def get_fine_size(field, fine):
    """单个细网格patch的数据量及形状，返回 (元素数, (nz, nx))"""
    lenx = (fine["x_end"] - fine["x_start"]) * fine["N"] + 1
    lenz = (fine["z_end"] - fine["z_start"]) * fine["N"] + 1
    if field == "sx" or field == "sz":
        return lenx * lenz, (lenz, lenx)
    elif field == "vx":
        return (lenx - 1) * lenz, (lenz, lenx - 1)
    elif field == "vz":
        return lenx * (lenz - 1), (lenz - 1, lenx)
    elif field == "txz":
        return (lenx - 1) * (lenz - 1), (lenz - 1, lenx - 1)
    else:
        raise ValueError(f"Unknown field: {field}")

def get_vertex_offset(field):
    """返回数据点中心相对于整数网格索引的偏移 (dx_off, dz_off)"""
    if field == "sx": return 0.5, 0.5
    elif field == "sz": return 0.5, 0.5
    elif field == "vx": return 0.0, 0.5
    elif field == "vz": return 0.5, 0.0
    elif field == "txz": return 0.0, 0.0
    else: return 0.5, 0.5

def get_crop_counts(field, nx_dim, nz_dim, thickness):
    """
    计算每侧需要裁剪的点数（halo + CPML）
    返回 (left, right, top, bottom)
    """
    if field in ["sx", "sz"]:
        left_halo, right_halo = 3, 4
        top_halo, bottom_halo = 3, 4
        cpml_x, cpml_z = thickness, thickness
    elif field == "vx":
        left_halo, right_halo = 4, 3
        top_halo, bottom_halo = 3, 4
        cpml_x, cpml_z = thickness - 1, thickness
    elif field == "vz":
        left_halo, right_halo = 3, 4
        top_halo, bottom_halo = 4, 3
        cpml_x, cpml_z = thickness, thickness - 1
    elif field == "txz":
        left_halo, right_halo = 4, 3
        top_halo, bottom_halo = 4, 3
        cpml_x, cpml_z = thickness - 1, thickness - 1
    else:
        raise ValueError(f"Unknown field: {field}")

    left = left_halo + cpml_x
    right = right_halo + cpml_x
    top = top_halo + cpml_z
    bottom = bottom_halo + cpml_z

    left = min(left, nx_dim - 1)
    right = min(right, nx_dim - 1)
    top = min(top, nz_dim - 1)
    bottom = min(bottom, nz_dim - 1)
    return left, right, top, bottom

def read_binary_float32(filepath):
    """读取二进制文件返回 float32 数组，失败时返回 None"""
    try:
        return np.fromfile(filepath, dtype=np.float32)
    except Exception:
        return None

def compute_field_ranges(field_files_dict):
    """
    计算每个场的全局最小最大值
    返回两个字典: field_min, field_max
    """
    field_min = {}
    field_max = {}
    for field, files in field_files_dict.items():
        vmin = float('inf')
        vmax = -float('inf')
        for fpath in files:
            data = read_binary_float32(fpath)
            if data is not None:
                vmin = min(vmin, data.min())
                vmax = max(vmax, data.max())
        if vmin == float('inf'):   # 该场无有效数据
            print(f"警告: 场 {field} 无有效数据，使用对称范围 [-1,1]")
            vmin, vmax = -1.0, 1.0
        field_min[field] = vmin
        field_max[field] = vmax
        print(f"场 {field}: 全局范围 [{vmin:.4e}, {vmax:.4e}]")
    return field_min, field_max

def plot_snapshot(field, filepath, coarse_params, fine_list, output_dir,
                  thickness, field_min, field_max):
    """
    处理单个波场快照文件，生成图片
    """
    data = read_binary_float32(filepath)
    if data is None:
        print(f"无法读取文件: {filepath}")
        return

    nx_c = coarse_params["nx"]
    nz_c = coarse_params["nz"]

    # 粗网格部分
    coarse_size, coarse_shape = get_coarse_size(field, nx_c, nz_c)

    # 计算所有细网格patch的数据量及形状
    fine_sizes = []
    fine_shapes = []
    total_fine = 0
    for f in fine_list:
        size, shape = get_fine_size(field, f)
        fine_sizes.append(size)
        fine_shapes.append(shape)
        total_fine += size

    expected = coarse_size + total_fine
    if len(data) != expected:
        print(f"警告: 文件 {filepath} 大小不匹配，期望 {expected}，实际 {len(data)}，跳过")
        return

    # 分割粗网格
    coarse_data = data[:coarse_size].reshape(coarse_shape)

    # 分割细网格
    patches = []
    offset = coarse_size
    for i, f in enumerate(fine_list):
        size = fine_sizes[i]
        shape = fine_shapes[i]
        patch_data = data[offset:offset+size].reshape(shape)
        offset += size
        patches.append((f, patch_data))

    # 裁剪粗网格
    nz_cur, nx_cur = coarse_shape
    left, right, top, bottom = get_crop_counts(field, nx_cur, nz_cur, thickness)
    if left + right >= nx_cur or top + bottom >= nz_cur:
        print(f"警告: 裁剪后无有效数据，跳过 {filepath}")
        return

    coarse_cropped = coarse_data[top:nz_cur-bottom, left:nx_cur-right]
    if coarse_cropped.size == 0:
        print(f"警告: 裁剪后数据为空，跳过 {filepath}")
        return

    # 顶点坐标（索引坐标，z 递增，即从上到下为正）
    dx_off, dz_off = get_vertex_offset(field)
    x_verts = np.arange(left, nx_cur - right + 1) - 0.5 + dx_off
    z_verts = np.arange(top, nz_cur - bottom + 1) - 0.5 + dz_off
    Xc, Zc = np.meshgrid(x_verts, z_verts)

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    norm = plt.Normalize(vmin=field_min, vmax=field_max)
    mesh_c = ax.pcolormesh(Xc, Zc, coarse_cropped, shading='flat',
                           cmap='seismic', norm=norm)

    # 绘制细网格patch
    for f, patch_data in patches:
        x_start = f["x_start"]
        x_end = f["x_end"]
        z_start = f["z_start"]
        z_end = f["z_end"]
        N = f["N"]
        nz_f, nx_f = patch_data.shape  # (nz, nx)

        # 生成数据点中心坐标
        x_pts = np.linspace(x_start, x_end, nx_f, endpoint=True)
        z_pts = np.linspace(z_start, z_end, nz_f, endpoint=True)

        # 顶点边界（确保单元格对齐）
        half = 0.5 / N
        x_edges = np.concatenate((
            [x_pts[0] - half],
            (x_pts[:-1] + x_pts[1:]) / 2,
            [x_pts[-1] + half]
        ))
        z_edges = np.concatenate((
            [z_pts[0] - half],
            (z_pts[:-1] + z_pts[1:]) / 2,
            [z_pts[-1] + half]
        ))
        Xf, Zf = np.meshgrid(x_edges, z_edges)
        ax.pcolormesh(Xf, Zf, patch_data, shading='flat',
                      cmap='seismic', norm=norm)

    ax.set_aspect('equal')
    ax.set_xlabel('x (grid index)')
    ax.set_ylabel('z (grid index)')
    ax.set_title(f'{field} snapshot')
    fig.colorbar(mesh_c, ax=ax)   # 使用 fig.colorbar 避免警告

    # 保存图片
    out_path = os.path.join(output_dir, os.path.splitext(os.path.basename(filepath))[0] + '.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"已生成: {out_path}")

# ==================== 主函数 ====================
def main():
    # 读取配置
    try:
        model = load_json(MODEL_JSON)
        param = load_json(PARAM_JSON)
    except Exception as e:
        print(f"读取配置文件失败: {e}")
        sys.exit(1)

    coarse = model["coarse"]
    fine_list = model["fine"]
    thickness = param["cpml"]["thickness"]

    # 准备输出目录
    if os.path.exists(SNAPSHOT_DIR):
        shutil.rmtree(SNAPSHOT_DIR)
    os.makedirs(SNAPSHOT_DIR)
    for field in FIELD_NAMES + [RECORD_DIR]:
        os.makedirs(os.path.join(SNAPSHOT_DIR, field), exist_ok=True)

    # 收集文件并按场分类
    field_files = {field: [] for field in FIELD_NAMES}
    for field in FIELD_NAMES:
        dir_path = os.path.join(OUTPUT_DIR, field)
        if not os.path.isdir(dir_path):
            print(f"警告: 场目录 {dir_path} 不存在")
            continue
        for fname in os.listdir(dir_path):
            if fname.endswith('.bin'):
                field_files[field].append(os.path.join(dir_path, fname))

    # 计算每个场的全局范围
    field_min, field_max = compute_field_ranges(field_files)

    # 多线程处理
    max_workers = max(1, os.cpu_count() - 2)
    print(f"使用线程数: {max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for field, files in field_files.items():
            for fpath in files:
                out_dir = os.path.join(SNAPSHOT_DIR, field)
                futures.append(executor.submit(
                    plot_snapshot, field, fpath, coarse, fine_list, out_dir,
                    thickness, field_min[field], field_max[field]
                ))

        # 等待所有任务完成，打印异常
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"任务异常: {e}")
                traceback.print_exc()

    # 地震记录预留
    record_in = os.path.join(OUTPUT_DIR, RECORD_DIR)
    if os.path.isdir(record_in):
        for fname in os.listdir(record_in):
            if fname.endswith('.bin'):
                print(f"地震记录文件待处理: {fname}")

    print("所有任务完成")

if __name__ == "__main__":
    main()