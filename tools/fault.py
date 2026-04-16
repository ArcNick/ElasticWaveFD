"""
fault_model.py

外部调用模块，提供向弹性参数矩阵中添加断层（断裂核 + 损伤带）的功能。
所有距离均以格点数为单位（即网格索引的差值），不涉及物理长度。
断层中心线使用网格索引 (iz, ix) 定义，其中 iz 对应 z 方向（行），ix 对应 x 方向（列）。
断裂核与损伤带的物性参数在模块内定义，用户可按需修改。
"""

import numpy as np

# ========== 断层物性参数（可直接在此修改） ==========
# 断裂核（core）物性
core_vp = 2400.0      # 纵波速度 (m/s)
core_vs = 900.0       # 横波速度 (m/s)
core_rho = 2050.0     # 密度 (kg/m^3)
core_epsilon = 0.0    # Thomsen epsilon
core_delta = 0.0      # Thomsen delta

# 损伤带（damage zone）物性
damage_vp = 3600.0
damage_vs = 1850.0
damage_rho = 2450.0
damage_epsilon = 0.0
damage_delta = 0.0
# =================================================


def _point_to_segment_distance_vectorized(pz, px, z1, x1, z2, x2):
    """
    向量化计算多个格点到一条线段的最短距离（单位：格点数）。

    参数
    ----------
    pz, px : 形状为 (nz, nx) 的整数数组
        所有网格点的 z 和 x 索引。
    z1, x1, z2, x2 : int
        线段端点的索引坐标 (z, x)。

    返回
    -------
    dist : 形状为 (nz, nx) 的浮点数组
        每个格点到线段的最短距离（格点数）。
    """
    dz = z2 - z1
    dx = x2 - x1
    l2 = dz*dz + dx*dx
    if l2 == 0:
        return np.hypot(pz - z1, px - x1)

    t = ((pz - z1)*dz + (px - x1)*dx) / l2
    t = np.clip(t, 0.0, 1.0)

    proj_z = z1 + t * dz
    proj_x = x1 + t * dx
    return np.hypot(pz - proj_z, px - proj_x)


def add_fault(rho, C11, C13, C33, C55,
              nx, nz,
              fault_start, fault_end,
              fault_total_width, core_width):
    """
    在原地修改弹性参数矩阵，添加一条断层（断裂核 + 损伤带）。

    参数
    ----------
    rho, C11, C13, C33, C55 : numpy.ndarray，形状 (nz, nx)
        密度和各向异性刚度系数（原地修改）。
    nx, nz : int
        网格点数（x 方向和 z 方向）。
    fault_start, fault_end : tuple of (int, int)
        断层中心线两个端点的网格索引，格式为 (iz, ix)。
    fault_total_width : float
        断层总宽度（以格点数为单位），损伤带的外边界半宽 = fault_total_width/2。
    core_width : float
        断裂核半宽（以格点数为单位），核部总宽度为 2*core_width。

    注意
    ----
    函数假定所有网格点均为固体（无流体），且不修改材料类型标志。
    刚度系数由模块内定义的 core_*/damage_* 物性重新计算。
    """
    z1, x1 = fault_start
    z2, x2 = fault_end

    # 生成所有网格点的索引坐标
    x_coords = np.arange(nx)
    z_coords = np.arange(nz)
    Z, X = np.meshgrid(z_coords, x_coords, indexing='ij')  # 形状 (nz, nx)

    # 计算每个格点到断层中心线的距离（格点数）
    dist = _point_to_segment_distance_vectorized(Z, X, z1, x1, z2, x2)

    # 创建掩码
    core_mask = (dist <= core_width)
    damage_mask = (dist > core_width) & (dist <= fault_total_width / 2.0)

    # ---------- 计算断裂核的刚度系数 ----------
    C33_core = core_rho * core_vp**2
    C55_core = core_rho * core_vs**2
    C11_core = C33_core * (1 + 2 * core_epsilon)
    temp = (C33_core - C55_core) * (2 * C33_core * core_delta + (C33_core - C55_core))
    if temp < 0:
        temp = 0.0
    C13_core = np.sqrt(temp) - C55_core

    # ---------- 计算损伤带的刚度系数 ----------
    C33_damage = damage_rho * damage_vp**2
    C55_damage = damage_rho * damage_vs**2
    C11_damage = C33_damage * (1 + 2 * damage_epsilon)
    temp = (C33_damage - C55_damage) * (2 * C33_damage * damage_delta + (C33_damage - C55_damage))
    if temp < 0:
        temp = 0.0
    C13_damage = np.sqrt(temp) - C55_damage

    # ---------- 原地修改数组 ----------
    if np.any(core_mask):
        rho[core_mask] = core_rho
        C11[core_mask] = C11_core
        C13[core_mask] = C13_core
        C33[core_mask] = C33_core
        C55[core_mask] = C55_core

    if np.any(damage_mask):
        rho[damage_mask] = damage_rho
        C11[damage_mask] = C11_damage
        C13[damage_mask] = C13_damage
        C33[damage_mask] = C33_damage
        C55[damage_mask] = C55_damage