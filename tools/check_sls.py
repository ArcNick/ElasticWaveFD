import numpy as np
import matplotlib.pyplot as plt
import sls

def calc_tau_Q(omega, tau_sigmas, tau):
    Q_inv = np.zeros_like(omega)
    for ts in tau_sigmas:
        Q_inv += (omega * ts * tau) / (1 + (omega * ts)**2)
    return 1.0 / Q_inv

def plot_combined():
    cases = [
        {'Q': 200, 'L': 3, 'ylim': (100, 300), 'color': 'b'},
        {'Q': 50,  'L': 3, 'ylim': (25,  75),  'color': 'g'},
        {'Q': 20,  'L': 3, 'ylim': (10,  30),  'color': 'r'}
    ]
    f_min, f_max = 2, 50
    f = np.linspace(f_min, f_max * 2, 200)
    omega = 2 * np.pi * f

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, case in enumerate(cases):
        Q_target = case['Q']
        L = case['L']
        params = sls.get_sls_parameters(Q_target, Q_target, L, f_min, f_max)
        Q_tau = calc_tau_Q(omega, params['tau_sigmas'], params['taup'])
        
        ax = axes[i]
        ax.plot(f, np.full_like(f, Q_target), 'k-', lw=2, label='Target Q')
        ax.plot(f, Q_tau, 'r--', lw=2, label=rf'$\tau$-method ($L={L}$)')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Quality factor Q')
        ax.set_title(f'Q = {Q_target}, L = {L}')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(case['ylim'])
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_combined()