import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import stft

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

def stran(x: np.ndarray, t: np.ndarray, f: np.ndarray) -> np.ndarray:
    """S 变换"""
    x = x.ravel()
    t = t.ravel()
    f = f.ravel()
    Nt = len(t)
    Nf = len(f)
    dt = t[1] - t[0]
    STx = np.zeros((Nf, Nt), dtype=np.complex128)
    
    f_col = f[:, None]
    t_row = t[None, :]
    f_abs = np.abs(f_col)
    
    for i in range(Nt):
        tau = t[i]
        td = t_row - tau
        gauss = (f_abs / np.sqrt(2*np.pi)) * np.exp(-0.5 * (f_col**2) * (td**2))
        phase = np.exp(-2j * np.pi * f_col * t_row)
        STx[:, i] = np.sum(x * gauss * phase, axis=1) * dt
    return STx

def calculate_and_plot_rms(data: np.ndarray, dt: float, time_window: tuple):
    """
    计算所有道在特定时间窗口内的 RMS 能量并绘图
    data: 形状为 (nt, nx) 的地震记录
    time_window: (t1, t2) 秒
    """
    t_start, t_end = time_window
    idx_start = int(np.round(t_start / dt))
    idx_end = int(np.round(t_end / dt))
    
    # 确保索引不越界
    idx_start = max(0, idx_start)
    idx_end = min(data.shape[0], idx_end)
    
    # 截取窗口内的数据 (nt_win, nx)
    win_data = data[idx_start:idx_end, :]
    
    # 计算每一道的 RMS: sqrt(mean(v^2))
    # axis=0 表示对时间轴求均值
    rms_values = np.sqrt(np.mean(np.square(win_data), axis=0))
    
    # 生成距离轴 (假设粗网格间距为 1.5m，根据你之前说的)
    dx = 1.5
    dist_x = np.arange(len(rms_values)) * dx
    
    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(dist_x, rms_values, 'r-', lw=2, label=f'RMS Amplitude ({t_start}-{t_end}s)')
    plt.title("Surface RMS Amplitude Profile (Vz)")
    plt.xlabel("Distance x (m)")
    plt.ylabel("RMS Amplitude")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()


# ===================== 用户可选的时窗参数 =====================
TIME_WINDOW = (0.29, 1)          # 例如: (0.3, 0.8) 秒
# ==============================================================

# ===================== 数据读取 =====================
FILE_DIR = "./output/record/record_vz.bin"
PARAMS_DIR = "./models/params.json"
MODEL_DIR = "./models/models.json"

with open(PARAMS_DIR, 'r') as f: params_data = json.load(f)
with open(MODEL_DIR, 'r') as f: model_data = json.load(f)

dt = 0.001
print(f"采样间隔 dt = {dt} s，采样率 = {1/dt:.1f} Hz")

nx = model_data["coarse"]["nx"]
nt = params_data["base"]["nt"]

# 读取数据并裁剪边界（去掉 PML / halo）
file = np.fromfile(FILE_DIR, dtype=np.float32).reshape((-1, nx))
file = file[:, 3:nx-4]                     # 裁剪 PML/halo
trace_orig = file[:, params_data["base"]["posx"]-3].astype(np.float64) * 1e6   # 缩放
# trace_orig[0:150] = trace_orig[150]         # 消除直流跳变

# 完整时间轴
t_full = np.arange(len(trace_orig)) * dt

# ---------- 根据时窗截取信号 ----------
if TIME_WINDOW is not None:
    t_start, t_end = TIME_WINDOW
    idx_start = int(np.round(t_start / dt))
    idx_end = int(np.round(t_end / dt))
    idx_start = max(0, idx_start)
    idx_end = min(len(trace_orig), idx_end)
    trace = trace_orig[idx_start:idx_end].copy()
    t = t_full[idx_start:idx_end].copy()          # 保留原始时间坐标，不从0开始
else:
    trace = trace_orig.copy()
    t = t_full.copy()

fs = 1 / dt

# ===================== 频域分析 =====================
# FFT
freq_fft = fftfreq(len(trace), dt)
amp_fft = np.abs(fft(trace))
freq_pos = freq_fft[freq_fft >= 0]
amp_pos = amp_fft[freq_fft >= 0]

# STFT
f_stft, t_stft_raw, Z_stft = stft(
    trace, 
    fs=fs, 
    nperseg=64, 
    noverlap=60, 
    nfft=128
)
# 将STFT的相对时间转换为绝对时间
t_stft = t_stft_raw + t[0]

stft_amp = np.abs(Z_stft)
f_max_plot = 150
f_mask_stft = f_stft <= f_max_plot
f_stft_plot = f_stft[f_mask_stft]
stft_amp_plot = stft_amp[f_mask_stft, :]

# S 变换
f_s = np.linspace(0, fs/2, len(trace)//2)
STx = stran(trace, t, f_s)                 # 这里使用原始时间轴 t
s_amp = np.abs(STx)
f_mask_s = f_s <= f_max_plot
f_s_plot = f_s[f_mask_s]
s_amp_plot = s_amp[f_mask_s, :]

# ===================== 绘图 =====================
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# 1. 时域波形
axes[0, 0].plot(t, trace, 'k', lw=0.8)
window_str = f"  [{t[0]:.3f}s - {t[-1]:.3f}s]" if TIME_WINDOW is not None else ""
axes[0, 0].set_title("Time Domain (No Filter)" + window_str)
axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].set_xlim(t.min(), t.max())
axes[0, 0].grid(True)

# 2. 频谱
axes[0, 1].plot(freq_pos, amp_pos, 'b', lw=1)
axes[0, 1].set_xlim(0, f_max_plot)
axes[0, 1].set_ylim(0, amp_pos.max() * 1.1 if amp_pos.max() > 0 else 1)
axes[0, 1].set_title("Frequency Spectrum")
axes[0, 1].set_xlabel("Frequency (Hz)")
axes[0, 1].grid(True)

calculate_and_plot_rms(file.astype(np.float64), dt, TIME_WINDOW)

# 4. S-Transform
im2 = axes[1, 1].imshow(
    s_amp_plot,
    cmap='jet',
    aspect='auto',
    origin='lower',
    extent=[t.min(), t.max(), f_s_plot.min(), f_s_plot.max()]
)
axes[1, 1].set_title("S-Transform")
axes[1, 1].set_xlabel("Time (s)")
axes[1, 1].set_ylabel("Frequency (Hz)")
axes[1, 1].set_ylim(0, f_max_plot)
fig.colorbar(im2, ax=axes[1, 1])

plt.tight_layout()
plt.show()