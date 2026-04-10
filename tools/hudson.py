import numpy as np

def thomsen_to_stiffness(vp, vs, rho, epsilon, delta, gamma):
    """
    将Thomsen参数转换为VTI刚度矩阵 (Voigt notation 6x6)。
    """
    # 垂直波速对应的模量
    c33 = rho * vp**2
    c44 = rho * vs**2
    c66 = c44 * (1 + 2 * gamma)
    c11 = c33 * (1 + 2 * epsilon)
    
    # 利用delta计算c13
    # delta = ((c13 + c44)^2 - (c33 - c44)^2) / (2 * c33 * (c33 - c44))
    # 推导 c13
    term = np.sqrt(2 * delta * c33 * (c33 - c44) + (c33 - c44)**2) - c44
    c13 = term
    
    # 构建刚度矩阵
    C0 = np.zeros((6, 6))
    C0[0, 0] = c11; C0[1, 1] = c11; C0[2, 2] = c33
    C0[0, 1] = c11 - 2 * c66; C0[1, 0] = C0[0, 1]
    C0[0, 2] = c13; C0[2, 0] = c13
    C0[1, 2] = c13; C0[2, 1] = c13
    C0[3, 3] = c44
    C0[4, 4] = c44
    C0[5, 5] = c66
    
    return C0, c33, c44

def compute_hudson_c1(c33_bg, c44_bg, rho_bg, fluid_eta, omega, crack_density, aspect_ratio):
    """
    计算Hudson一阶扰动项 C1。
    针对水平裂缝 (n = [0,0,1])。
    """
    # 裂缝密度 e (通常定义为 N*a^3 或类似的无量纲量)
    # 这里假设输入的 crack_density 即为 Hudson 定义的 e
    e = crack_density
    
    # 背景拉梅常数 (近似取垂直方向的等效值)
    mu = c44_bg
    lam = c33_bg - 2 * c44_bg
    
    # 裂缝填充物性质 (粘性流体)
    # 流体模量 K' = -i * omega * eta (忽略流体体积模量，仅考虑粘性)
    # 或者更严谨的复模量 M_fluid = K_fluid - i*omega*eta。
    # 此处假设主要由粘度控制:
    mu_fluid = 0.0
    k_fluid = -1j * omega * fluid_eta
    
    # --- 计算 U1 (切向柔度相关) ---
    # 对于粘性流体，切向应力传递受限。
    # 简化处理：如果裂缝很薄且充满流体，U1主要受流体剪切阻抗影响。
    # 但在Hudson理论中，通常假设流体不能承受剪切 (mu'=0)，此时U1仅取决于基质。
    # 如果考虑流体的粘性剪切耦合，公式会非常复杂。
    # 此处采用 Hudson (1981) 标准公式 (mu'=0):
    U1 = 16 / (3 * (3 - 2 * mu / (lam + 2 * mu))) # 简化形式，实际上分母是 (3-2*nu)/(1-nu) 之类
    # 修正 U1 表达式以匹配标准 Hudson 公式 (各向同性基质近似):
    nu = lam / (2 * (lam + mu)) # 泊松比近似
    U1 = 16 * (1 - nu) / (3 * (2 - nu))
    
    # --- 计算 U3 (法向柔度相关) ---
    # 包含流体压缩性/粘性影响
    # M = (lambda + 2mu) / pi * (aspect_ratio) ... 这里的公式依赖于具体近似
    # 使用更通用的形式:
    kappa = lam + 2*mu/3 # 体模量
    # 刚度比
    # 这里的 M 是裂缝面法向刚度参数
    # 对于 penny-shaped crack:
    term_M = (lam + 2*mu) / (np.pi * aspect_ratio)
    
    # 修正项 K' (流体模量)
    # 如果流体是粘性的: K_fluid_eff = -1j * omega * eta
    # 注意：Hudson公式中的K'通常指流体体模量。
    # 这里我们将粘性项直接代入复模量计算。
    
    # 复数形式的法向柔度参数
    # 这是一个复数，导致 C1 为复数
    denominator = 1 + (k_fluid / (term_M))
    U3 = 4 / (3 * denominator) # 简化示意，实际公式涉及更多项
    # 准确的 U3 公式 (Hudson 1981, Eq 2-12 变形):
    # U3 = 4/3 * 1 / (1 + M/K')  <-- 这种形式常见
    # 让我们使用更精确的各向同性近似公式代入 VTI 垂直模量:
    M_param = (lam + 2*mu) / (np.pi * aspect_ratio)
    U3 = 4 / (3 * (1 + M_param / (k_fluid + 4*mu/3))) # 近似
    
    # --- 构建 C1 矩阵 (Voigt 6x6) ---
    # 水平裂缝只影响垂直传播波的性质 (33, 44, 13, etc.)
    # 参考 Hudson (1981) 或 Crampin (1984) 的水平裂缝形式
    C1 = np.zeros((6, 6), dtype=np.complex128)
    
    factor = -e / mu # Hudson 公式前的系数通常是 -e/mu 或类似，取决于 e 的定义
    
    # 根据图片中的公式 (2-7) 形式映射到 Voigt 标记:
    # 11->1, 22->2, 33->3, 23->4, 13->5, 12->6
    # 注意：图片公式是针对各向同性背景推导的。
    # 我们将其中的 lambda, mu 替换为 VTI 的垂直等效值。
    
    C1[2, 2] = (lam + 2*mu)**2 * U3  # 33
    C1[0, 2] = lam * (lam + 2*mu) * U3 # 13
    C1[1, 2] = lam * (lam + 2*mu) * U3 # 23
    C1[2, 0] = C1[0, 2]
    C1[2, 1] = C1[1, 2]
    
    C1[0, 0] = lam**2 * U3      # 11 (弱影响)
    C1[1, 1] = lam**2 * U3      # 22
    C1[0, 1] = lam**2 * U3      # 12
    C1[1, 0] = C1[0, 1]
    
    # 剪切分量 (主要受 U1 控制)
    # 对于水平裂缝，SH波 (C66) 和 SV波 (C44) 受影响不同
    # 图片公式中 44, 55 位置对应 U1
    C1[3, 3] = 2 * mu**2 * U1   # 44 (SV波衰减关键)
    C1[4, 4] = 2 * mu**2 * U1   # 55
    C1[5, 5] = 2 * mu**2 * U1   # 66 (SH波)
    
    C1 *= factor # 应用系数
    
    return C1, U1, U3

def map_to_sls_params(C0_real, C1_complex, f0):
    """
    将复刚度映射到 SLS 参数 (tau_sigma, tau_epsilon, Q).
    这里我们主要针对受影响最大的分量（例如 C33 和 C44）进行拟合。
    为了简化，我们计算一个“视”Q值，并假设 tau_sigma 全局一致。
    """
    omega0 = 2 * np.pi * f0
    
    # 1. 获取总复刚度
    C_total_complex = C0_real + C1_complex
    
    # 2. 提取关键分量 (垂直传播主要看 C33 和 C44)
    # 注意：C0 是实数，C1 是复数
    M_eff = C_total_complex[2, 2] # C33
    Mu_eff = C_total_complex[3, 3] # C44
    
    # 3. 计算品质因子 Q (定义: Q = Re(M) / Im(M))
    # 注意：Hudson 的 C1 虚部通常是负的 (物理衰减)，所以 Im(M) < 0
    # 这里取绝对值
    Qp_eff = np.real(M_eff) / np.abs(np.imag(M_eff))
    Qs_eff = np.real(Mu_eff) / np.abs(np.imag(Mu_eff))
    
    # 4. 计算 SLS 参数
    # 关系式: Q = (1 + omega^2 * tau_sigma * tau_epsilon) / (omega * tau_sigma * (tau_epsilon - tau_sigma))
    # 简化关系 (中心频率处): tau_epsilon = tau_sigma * (1 + 1/Q^2 + 1/Q) ... 近似
    # 更常用的关系 (Liu et al., 1976):
    # tau_sigma = 1 / (omega0 * Q)
    # tau_epsilon = tau_sigma * (1 + 1/Q)  <-- 这是一个近似，精确解需解二次方程
    
    # 我们使用更稳健的近似 (适用于 Q > 10):
    # tau_sigma = 1 / (omega0 * Q)
    # tau_epsilon = tau_sigma * (1 + 1/Q)
    
    tau_sigma_p = 1.0 / (omega0 * Qp_eff)
    tau_epsilon_p = tau_sigma_p * (1.0 + 1.0/Qp_eff)
    
    tau_sigma_s = 1.0 / (omega0 * Qs_eff)
    tau_epsilon_s = tau_sigma_s * (1.0 + 1.0/Qs_eff)
    
    # 为了代码整洁，我们返回 P波和 S波分开的参数
    # 在实际方程中，通常取平均值或分别赋值给对应的应力分量
    
    sls_params = {
        'tau_sigma_p': tau_sigma_p,
        'tau_epsilon_p': tau_epsilon_p,
        'Qp_eff': Qp_eff,
        'tau_sigma_s': tau_sigma_s,
        'tau_epsilon_s': tau_epsilon_s,
        'Qs_eff': Qs_eff
    }
    
    # 返回更新后的实部刚度矩阵 (用于方程中的 M_L 或 M_U)
    # 注意：SLS方程通常写成 dSigma/dt = M_U * dEps/dt - 1/tau * R
    # 这里的 M_U 应该是 C_total 的实部 (高频极限刚度)
    C_final_real = np.real(C_total_complex)
    
    return C_final_real, sls_params

def get_hudson_sls_parameters(params):
    """
    主接口函数。
    
    输入 params (dict):
        - vp, vs, rho: 背景垂直波速和密度
        - epsilon, delta, gamma: Thomsen参数
        - fluid_eta: 裂缝流体粘度 (Pa.s)
        - f0: 震源主频 (Hz)
        - crack_density: 裂缝密度 e
        - aspect_ratio: 裂缝纵横比 alpha
        
    返回:
        - C_final: 最终用于差分方程的实刚度矩阵 (6x6)
        - sls_params: 包含 tau_sigma, tau_epsilon 的字典
    """
    # 1. 背景介质
    C0, c33_bg, c44_bg = thomsen_to_stiffness(
        params['vp'], params['vs'], params['rho'],
        params['epsilon'], params['delta'], params['gamma']
    )
    
    # 2. 频率与粘度
    omega = 2 * np.pi * params['f0']
    
    # 3. 计算 C1 (复数)
    # 注意：这里假设裂缝是水平的，所以使用 c33_bg, c44_bg 作为背景参数
    C1, _, _ = compute_hudson_c1(
        c33_bg, c44_bg, params['rho'],
        params['fluid_eta'], omega,
        params['crack_density'], params['aspect_ratio']
    )
    
    # 4. 映射到 SLS
    C_final, sls_params = map_to_sls_params(C0, C1, params['f0'])
    
    return C_final, sls_params

# ==========================================
# 使用示例 (在你的主程序中)
# ==========================================
"""
params = {
    'vp': 3000.0,
    'vs': 1732.0,
    'rho': 2500.0,
    'epsilon': 0.1,
    'delta': 0.05,
    'gamma': 0.1,
    'fluid_eta': 0.01,      # 例如 0.01 Pa.s (水)
    'f0': 25.0,
    'crack_density': 0.02,  # 稀疏裂缝
    'aspect_ratio': 0.001   # 扁平裂缝
}

C_matrix, sls_p = get_hudson_sls_parameters(params)

print("最终刚度矩阵 C (用于方程):")
print(C_matrix)
print("\nSLS 参数 (用于方程):")
print(f"tau_sigma_p (用于 Eq 3,4,6,7): {sls_p['tau_sigma_p']:.4e}")
print(f"tau_epsilon_p: {sls_p['tau_epsilon_p']:.4e}")
print(f"tau_sigma_s (用于 Eq 5,8): {sls_p['tau_sigma_s']:.4e}")
"""

