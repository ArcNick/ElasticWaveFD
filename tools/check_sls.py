import numpy as np
import matplotlib.pyplot as plt
import sls

# 论文式(14)计算τ方法Q值
def calc_tau_Q(omega, tau_sigmas, tau):
    Q_inv = np.zeros_like(omega)
    for ts in tau_sigmas:
        Q_inv += (omega * ts * tau) / (1 + (omega * ts)**2)
    return 1.0 / Q_inv

# 图2：Q=20, L=2
def plot_fig2():
    Q_target = 30
    L = 2
    f_min, f_max = 2, 100
    params = sls.get_sls_parameters(Q_target, Q_target, L, f_min, f_max)
    f = np.linspace(f_min, f_max, 200)
    omega = 2 * np.pi * f
    Q_tau = calc_tau_Q(omega, params['tau_sigmas'], params['taup'])
    
    plt.figure(figsize=(10,6))
    plt.plot(f, np.full_like(f, Q_target), 'k-', lw=2, label='Target Q')
    plt.plot(f, Q_tau, 'r--', lw=2, label=rf'$\tau$-method ($L={L}$)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Quality factor Q')
    plt.title(f'Constant Q = {Q_target} (2–25 Hz), {L} mechanisms')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim(10, 50)
    plt.show()

# 图3：Q=20, L=5
def plot_fig3():
    Q_target = 50
    L = 3
    f_min, f_max = 2, 100
    params = sls.get_sls_parameters(Q_target, Q_target, L, f_min, f_max)
    f = np.linspace(f_min, f_max, 200)
    omega = 2 * np.pi * f
    Q_tau = calc_tau_Q(omega, params['tau_sigmas'], params['taup'])
    
    plt.figure(figsize=(10,6))
    plt.plot(f, np.full_like(f, Q_target), 'k-', lw=2, label='Target Q')
    plt.plot(f, Q_tau, 'r--', lw=2, label=rf'$\tau$-method ($L={L}$)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Quality factor Q')
    plt.title(f'Constant Q = {Q_target} (2–25 Hz), {L} mechanisms')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim(20, 70)
    plt.show()

# 图4：Q=100, L=2
def plot_fig4():
    Q_target = 20
    L = 5
    f_min, f_max = 2, 100
    params = sls.get_sls_parameters(Q_target, Q_target, L, f_min, f_max)
    f = np.linspace(f_min, f_max, 200)
    omega = 2 * np.pi * f
    Q_tau = calc_tau_Q(omega, params['tau_sigmas'], params['taup'])
    
    plt.figure(figsize=(10,6))
    plt.plot(f, np.full_like(f, Q_target), 'k-', lw=2, label='Target Q')
    plt.plot(f, Q_tau, 'r--', lw=2, label=rf'$\tau$-method ($L={L}$)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Quality factor Q')
    plt.title(f'Constant Q = {Q_target} (2–25 Hz), {L} mechanisms')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim(10, 50)
    plt.show()

if __name__ == '__main__':
    plot_fig2()
    plot_fig3()
    plot_fig4()