import numpy as np

# ============================================================
# 1. Thomsen 参数 -> 二维 VTI 背景刚度（精确公式）
# ============================================================
def thomsen_to_vti_stiffness_2d(Vp0, Vs0, rho, epsilon, delta, gamma=0.0):
    """
    将 Thomsen 各向异性参数转换为二维 VTI 背景刚度系数。
    返回: C11, C13, C33, C55 (实数, Pa)
    """
    C33 = rho * Vp0**2
    C55 = rho * Vs0**2 * (1.0 + 2.0 * gamma)
    C11 = C33 * (1.0 + 2.0 * epsilon)
    term = 2.0 * delta * C33 * (C33 - C55) + (C33 - C55)**2
    C13 = np.sqrt(term) - C55
    return C11, C13, C33, C55


# ============================================================
# 2. Hudson 一阶扰动（水平裂缝，法向平行 Z 轴）
# ============================================================
def hudson_horizontal_crack_2d(lambda_matrix, mu_matrix,
                               lambda_fluid, eta, omega,
                               crack_density, a, c):
    """ 返回复数扰动: dC11, dC13, dC33, dC55 """
    mu_fluid = -1j * omega * eta
    M = (4.0 / np.pi) * (a * mu_fluid / (c * mu_matrix)) * \
        ((lambda_matrix + 2.0*mu_matrix) / (3.0*lambda_matrix + 4.0*mu_matrix))
    K = (1.0 / np.pi) * (a / c) * ((lambda_fluid + 2.0*mu_fluid) / mu_matrix) * \
        ((lambda_matrix + 2.0*mu_matrix) / (lambda_matrix + mu_matrix))

    U11 = (16.0/3.0) * ((lambda_matrix + 2.0*mu_matrix) / (3.0*lambda_matrix + 4.0*mu_matrix)) / (1.0 + M)
    U33 = (4.0/3.0)  * ((lambda_matrix + 2.0*mu_matrix) / (lambda_matrix + mu_matrix))       / (1.0 + K)

    factor = -crack_density / mu_matrix
    dC11 = factor * (lambda_matrix**2) * U33
    dC13 = factor * lambda_matrix * (lambda_matrix + 2.0*mu_matrix) * U33
    dC33 = factor * (lambda_matrix + 2.0*mu_matrix)**2 * U33
    dC55 = factor * (2.0 * mu_matrix**2) * U11
    return dC11, dC13, dC33, dC55


# ============================================================
# 3. 主接口：直接返回实数刚度 + SLS 参数
# ============================================================
def hudson_crack_sls_parameters(
    Vp0, Vs0, rho,
    epsilon, delta, gamma,
    lambda_fluid, eta, f0,
    crack_density, a, c,
    Qp_boost=1.0, Qs_boost=1.0
):
    """
    计算含水平裂缝（法向 // Z）的二维 VTI 等效介质参数，
    直接返回用于时域 SLS 波动方程的实数未松弛刚度及松弛参数。

    参数:
        Vp0, Vs0 : 基质纵、横波速度 (m/s)
        rho : 基质密度 (kg/m³)
        epsilon, delta, gamma : Thomsen 各向异性参数
        lambda_fluid : 裂缝流体体积模量 (Pa)
        eta : 流体剪切粘度 (Pa·s)
        f0 : 参考频率 / 震源主频 (Hz)
        crack_density : 裂缝密度 ε = ν a³
        a : 裂缝平均半径 (m)
        c : 裂缝厚度 (m)
        Qp_boost, Qs_boost : 衰减增强因子，范围 (0,1]。
                             若 Hudson 衰减过弱可设为 0.1~0.5；默认 1.0。

    返回:
        dict: {
            'C11': float,   # 实数未松弛刚度 (Pa)
            'C13': float,
            'C33': float,
            'C55': float,
            'tau_p': float, # 纵波无量纲松弛强度
            'tau_s': float, # 横波无量纲松弛强度
            'tau_sigma': float, # 应力松弛时间 (s)
            'Qp': float,    # 实际使用的品质因子
            'Qs': float
        }
    """
    omega = 2.0 * np.pi * f0

    # 背景刚度
    C11_0, C13_0, C33_0, C55_0 = thomsen_to_vti_stiffness_2d(
        Vp0, Vs0, rho, epsilon, delta, gamma
    )

    # 基质拉梅常数
    mu_matrix = rho * Vs0**2
    lambda_matrix = rho * Vp0**2 - 2.0 * mu_matrix

    # 裂缝扰动
    dC11, dC13, dC33, dC55 = hudson_horizontal_crack_2d(
        lambda_matrix, mu_matrix, lambda_fluid, eta, omega,
        crack_density, a, c
    )

    # 总复刚度
    C11_c = C11_0 + dC11
    C13_c = C13_0 + dC13
    C33_c = C33_0 + dC33
    C55_c = C55_0 + dC55

    # 未松弛刚度：取复刚度实部
    C11 = np.real(C11_c)
    C13 = np.real(C13_c)
    C33 = np.real(C33_c)
    C55 = np.real(C55_c)

    # 原始 Q 值
    eps = 1e-18
    Qp_raw = np.real(C33_c) / max(abs(np.imag(C33_c)), eps)
    Qs_raw = np.real(C55_c) / max(abs(np.imag(C55_c)), eps)

    # 应用增强因子
    Qp = Qp_raw * Qp_boost
    Qs = Qs_raw * Qs_boost

    # SLS 松弛参数
    tau_sigma = 1.0 / (2.0 * np.pi * f0)
    tau_p = 2.0 / (Qp - 1.0) if Qp > 1.0 else 0.0
    tau_s = 2.0 / (Qs - 1.0) if Qs > 1.0 else 0.0

    return {
        'C11': C11, 'C13': C13, 'C33': C33, 'C55': C55,
        'tau_p': tau_p, 'tau_s': tau_s, 'tau_sigma': tau_sigma,
        'Qp': Qp, 'Qs': Qs
    }


# ============================================================
# 示例
# ============================================================
if __name__ == "__main__":
    params = hudson_crack_sls_parameters(
        Vp0=4500.0, Vs0=2200.0, rho=2500.0,
        epsilon=-0.02, delta=-0.05, gamma=0.08,
        lambda_fluid=1.44e9, eta=0.05, f0=30.0,
        crack_density=0.05, a=0.1, c=0.0002,
    )

    print("===== 时域 SLS 方程输入参数（均为实数） =====")
    print(f"C11 = {params['C11']:.6e} Pa")
    print(f"C13 = {params['C13']:.6e} Pa")
    print(f"C33 = {params['C33']:.6e} Pa")
    print(f"C55 = {params['C55']:.6e} Pa")
    print(f"τ_σ = {params['tau_sigma']:.6e} s")
    print(f"τ_p = {params['tau_p']:.6f}")
    print(f"τ_s = {params['tau_s']:.6f}")
    print(f"Qp  = {params['Qp']:.3f}")
    print(f"Qs  = {params['Qs']:.3f}")