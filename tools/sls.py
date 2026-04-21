import numpy as np
# Blanch et al. 1995 tau method
QFIXFACTOR = 0.96

def getI0(tau_sig: float, f1: float, f2: float) -> float:
    omega1 = 2 * np.pi * f1
    omega2 = 2 * np.pi * f2
    val2 = np.log(1 + omega2**2 * tau_sig**2)
    val1 = np.log(1 + omega1**2 * tau_sig**2)
    return 1.0 / (2.0 * tau_sig) * (val2 - val1)

def getI1(tau_sig: float, f1: float, f2: float) -> float:
    omega1 = 2 * np.pi * f1
    omega2 = 2 * np.pi * f2
    def F(omega, t):
        return np.arctan(omega * t) - (omega * t) / (1 + (omega * t)**2)
    val2 = F(omega2, tau_sig)
    val1 = F(omega1, tau_sig)
    return 1.0 / (2.0 * tau_sig) * (val2 - val1)

def getI2(tsl: float, tsk: float, f1: float, f2: float) -> float:
    omega1 = 2 * np.pi * f1
    omega2 = 2 * np.pi * f2
    denom = tsk**2 - tsl**2
    def term(omega, t):
        return np.arctan(omega * t) / t
    val2 = term(omega2, tsl) - term(omega2, tsk)
    val1 = term(omega1, tsl) - term(omega1, tsk)
    return (tsl * tsk / denom) * (val2 - val1)

# ==================== Blanch tau method ====================
def compute_tau(Q_target: float, tau_sigmas: np.ndarray, f_min: float, f_max: float) -> float:
    I0 = np.array([getI0(ts, f_min, f_max) for ts in tau_sigmas])
    I1 = np.array([getI1(ts, f_min, f_max) for ts in tau_sigmas])
    sum_I0 = np.sum(I0)
    sum_I1 = np.sum(I1)

    sum_I2 = 0.0
    L = len(tau_sigmas)
    for l in range(L - 1):
        for k in range(l + 1, L):
            sum_I2 += getI2(tau_sigmas[l], tau_sigmas[k], f_min, f_max)

    # Blanch
    Q_input = QFIXFACTOR * Q_target
    tau = (1.0 / Q_input) * (sum_I0 / (sum_I1 + 2.0 * sum_I2))
    return tau

# ==================== LIU Xue-feng, FAN You-hua, CHANG Dong-mei 2017 ====================
def generate_tau_sigmas(L: int, f_min: float, f_max: float) -> np.ndarray:
    """
    1/tau_sigma 的对数均匀分布在 [f_min/2, 2*f_max]
    """
    f_start = f_min / 2
    f_end   = f_max * 2
    f_centers = np.logspace(np.log10(f_start), np.log10(f_end), L)
    tau_sigmas = 1.0 / (2 * np.pi * f_centers)
    return tau_sigmas

def get_sls_parameters(Qp: float, Qs: float, L: int, f_min: float, f_max: float) -> dict:
    tau_sigmas = generate_tau_sigmas(L, f_min, f_max)

    taup = compute_tau(Qp, tau_sigmas, f_min, f_max)
    taus = compute_tau(Qs, tau_sigmas, f_min, f_max)

    return {
        'taup': taup,
        'taus': taus,
        'tau_sigmas': tau_sigmas,          # 单位：秒
    }
