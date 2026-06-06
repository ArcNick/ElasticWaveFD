import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ==========================================================
# 路径
# ==========================================================

OUTPUT_BASE = './output'
IMAGE_BASE  = './images_kk'
MODEL_JSON  = './models/models.json'

os.makedirs(IMAGE_BASE, exist_ok=True)

# ==========================================================
# 配置
# ==========================================================

FIELD = 'vz'

# 分析哪个细网格
FINE_GRID_INDEX = 0

# 要分析哪个 snapshot
SNAPSHOT_FILE = 'vz_00352ms.bin'

# 是否取 log
USE_LOG = True

# ==========================================================
# 读取模型
# ==========================================================

with open(MODEL_JSON) as f:
    model = json.load(f)

coarse = model['coarse']
fine   = model['fine'][FINE_GRID_INDEX]

# ==========================================================
# 网格参数
# ==========================================================

N = fine['N']

x0 = fine['x_start']
x1 = fine['x_end']

z0 = fine['z_start']
z1 = fine['z_end']

# ==========================================================
# vz 属于 half_z
#
# 细网格尺寸：
#
# nxf = (x1-x0)*N + 1
# nzf = (z1-z0)*N
# ==========================================================

nxf = (x1 - x0) * N + 1
nzf = (z1 - z0) * N

print('fine grid size = ', nxf, nzf)

# ==========================================================
# 粗网格尺寸（用于跳过）
# ==========================================================

pml = 20  # 这里只用于跳 offset，不重要

nx_c = coarse['nx']
nz_c = coarse['nz']

coarse_size = nx_c * (nz_c - 1)

# ==========================================================
# 读取 snapshot
# ==========================================================

fpath = os.path.join(OUTPUT_BASE, FIELD, SNAPSHOT_FILE)

data = np.fromfile(fpath, dtype=np.float32)

print('total float count = ', len(data))

# ==========================================================
# 跳过 coarse
# ==========================================================

offset = coarse_size

# ==========================================================
# 跳过前面的 fine block
# ==========================================================

for idx in range(FINE_GRID_INDEX):

    ftmp = model['fine'][idx]

    Nt = ftmp['N']

    xx0 = ftmp['x_start']
    xx1 = ftmp['x_end']

    zz0 = ftmp['z_start']
    zz1 = ftmp['z_end']

    block_nx = (xx1 - xx0) * Nt + 1
    block_nz = (zz1 - zz0) * Nt

    offset += block_nx * block_nz

# ==========================================================
# 提取当前 fine block
# ==========================================================

block_size = nxf * nzf

fine_data = data[offset : offset + block_size]

fine_data = fine_data.reshape((nzf, nxf))

print('fine data shape = ', fine_data.shape)

# ==========================================================
# 去均值（非常重要）
# ==========================================================

fine_data = fine_data - np.mean(fine_data)

# ==========================================================
# 2D FFT
# ==========================================================

U = np.fft.fft2(fine_data)

# shift 到中心
U = np.fft.fftshift(U)

# 振幅谱
A = np.abs(U)

# ==========================================================
# log 显示
# ==========================================================

if USE_LOG:
    A = np.log10(A + 1e-12)

# ==========================================================
# 归一化
# ==========================================================

A = A - np.min(A)

if np.max(A) > 0:
    A = A / np.max(A)

# ==========================================================
# 绘图
# ==========================================================

fig, ax = plt.subplots(figsize=(8, 8))

im = ax.imshow(
    A,
    cmap='jet',
    origin='lower',
    extent=[-1, 1, -1, 1],
    aspect='equal'
)

fig.colorbar(im, ax=ax)

ax.set_title(f'kk spectrum : {SNAPSHOT_FILE}')
ax.set_xlabel(r'$k_x / \pi$')
ax.set_ylabel(r'$k_z / \pi$')

# ==========================================================
# 保存
# ==========================================================

out_png = os.path.join(
    IMAGE_BASE,
    SNAPSHOT_FILE.replace('.bin', '_kk.png')
)

fig.savefig(out_png, dpi=150, bbox_inches='tight')

plt.close(fig)

print('saved : ', out_png)