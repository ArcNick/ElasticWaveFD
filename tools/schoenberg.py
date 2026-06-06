import numpy as np

def single_fracture_HTI(
    c11: np.float32, 
    c13: np.float32, 
    c33: np.float32, 
    c55: np.float32, 
    z_n: np.float32 = 1e-13, 
    z_t: np.float32 = 4e-13, 
    num_frac: int = 1
) -> tuple:
    """
    [O(1) 解析版本] 针对单个网格点，在VTI背景介质中加入垂直裂缝（法向为x）
    """
    # 1. 提升至 float64 精度
    c11_64 = np.float64(c11)
    c13_64 = np.float64(c13)
    c33_64 = np.float64(c33)
    c55_64 = np.float64(c55)
    
    z_n_total = np.float64(z_n) * num_frac
    z_t_total = np.float64(z_t) * num_frac

    # 2. 垂直裂缝更新逻辑 (法向为 x)
    # 此时分母项由 C11 决定
    delta_n = 1.0 + c11_64 * z_n_total
    
    # 刚度项更新：C11受直接扰动，C33受耦合扰动
    c11_new = c11_64 / delta_n
    c13_new = c13_64 / delta_n
    c33_new = c33_64 - (c13_64**2 * z_n_total) / delta_n
    
    # 剪切波更新（在2D中，x方向和z方向裂缝对C55的解析解形式一致）
    delta_t = 1.0 + c55_64 * z_t_total
    c55_new = c55_64 / delta_t

    # 3. 严格退回 float32
    return (
        np.float32(c11_new), 
        np.float32(c13_new), 
        np.float32(c33_new), 
        np.float32(c55_new)
    )

def single_fracture_VTI(
    c11: np.float32, 
    c13: np.float32, 
    c33: np.float32, 
    c55: np.float32, 
    z_n: np.float32 = 1e-13, 
    z_t: np.float32 = 4e-13, 
    num_frac: int = 1
) -> tuple:
    c11_64 = np.float64(c11)
    c13_64 = np.float64(c13)
    c33_64 = np.float64(c33)
    c55_64 = np.float64(c55)
    
    z_n_total = np.float64(z_n) * num_frac
    z_t_total = np.float64(z_t) * num_frac

    delta_n = 1.0 + c33_64 * z_n_total
    
    # 等效刚度计算
    c33_new = c33_64 / delta_n
    c13_new = c13_64 / delta_n
    c11_new = c11_64 - (c13_64**2 * z_n_total) / delta_n
    
    # 剪切波独立解耦
    delta_t = 1.0 + c55_64 * z_t_total
    c55_new = c55_64 / delta_t

    return (
        np.float32(c11_new), 
        np.float32(c13_new), 
        np.float32(c33_new), 
        np.float32(c55_new)
    )

def grid_fracture_VTI(
    c11: np.ndarray, 
    c13: np.ndarray, 
    c33: np.ndarray, 
    c55: np.ndarray, 
    z_n_base: np.float32 = 1e-13, 
    z_t_base: np.float32 = 4e-13, 
    seed: int = None
) -> tuple:
    nz, nx = c11.shape
    
    # 裂缝组数分布控制
    if seed is not None:
        np.random.seed(seed)
        num_frac = np.random.randint(2, 21, size=(nz, nx)).astype(np.float64)
    else:
        num_frac = np.full((nz, nx), 8.0, dtype=np.float64)
        
    z_n_arr = np.float64(z_n_base) * num_frac
    z_t_arr = np.float64(z_t_base) * num_frac

    # 1. 强制转换为 float64
    c11_64 = c11.astype(np.float64)
    c13_64 = c13.astype(np.float64)
    c33_64 = c33.astype(np.float64)
    c55_64 = c55.astype(np.float64)
    
    # 2. 向量化 O(1) 解析求解
    # 法向应力耦合分母项场
    delta_n_arr = 1.0 + c33_64 * z_n_arr
    
    # 等效刚度场计算
    c33_new = c33_64 / delta_n_arr
    c13_new = c13_64 / delta_n_arr
    c11_new = c11_64 - (c13_64**2 * z_n_arr) / delta_n_arr
    
    # 剪切波独立解耦场
    delta_t_arr = 1.0 + c55_64 * z_t_arr
    c55_new = c55_64 / delta_t_arr
    
    # 3. 抽取结果并退回 float32 精度
    return (
        c11_new.astype(np.float32),
        c13_new.astype(np.float32),
        c33_new.astype(np.float32),
        c55_new.astype(np.float32)
    )