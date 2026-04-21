import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, stft

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

def stran(x: np.ndarray, t: np.ndarray, f: np.ndarray) -> np.ndarray:
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

# ===================== 数据读取 =====================
FILE_DIR = "./output/record/record_vz.bin"
PARAMS_DIR = "./models/params.json"
MODEL_DIR = "./models/models.json"

with open(PARAMS_DIR, 'r') as f: params_data = json.load(f)
with open(MODEL_DIR, 'r') as f: model_data = json.load(f)

dt = params_data["base"]["dt"]
nx = model_data["coarse"]["nx"]
nt = params_data["base"]["nt"]
target_dt = 0.001

# ===================== 降采样 =====================
k = int(round(target_dt / dt))
file = np.fromfile(FILE_DIR, dtype=np.float32).reshape((nt, nx))
file = file[:, 3:nx-4]
trace = file[:, params_data["base"]["posx"]-3].astype(np.float64) * 1e6

# 低通滤波参数
def lowpass_filter(data: np.ndarray, dt: float, cutoff=500, order=4) -> np.ndarray:
    fs = 1/dt
    nyq = 0.5*fs
    b, a = butter(order, cutoff/nyq, 'low')
    return filtfilt(b, a, data)

trace_filt = lowpass_filter(trace, dt)
trace_down = trace_filt[::k]
trace_down[0:200] = trace_down[200]
dt_down = dt * k
fs_down = 1/dt_down
t_total = np.arange(len(trace_down)) * dt_down
f_max_plot = 150  # 统一绘制 0~150Hz

freq_fft = fftfreq(len(trace_down), dt_down)
amp_fft = np.abs(fft(trace_down))
freq_pos = freq_fft[freq_fft >= 0]
amp_pos = amp_fft[freq_fft >= 0]

f_stft, t_stft, Z_stft = stft(
    trace_down, 
    fs=fs_down, 
    nperseg=64, 
    noverlap=60, 
    nfft=128
)
stft_amp = np.abs(Z_stft)

f_mask_stft = f_stft <= f_max_plot
f_stft_plot = f_stft[f_mask_stft]
stft_amp_plot = stft_amp[f_mask_stft, :]

f_s = np.linspace(0, fs_down/2, len(trace_down)//2)
STx = stran(trace_down, t_total, f_s)
s_amp = np.abs(STx)

f_mask_s = f_s <= f_max_plot
f_s_plot = f_s[f_mask_s]
s_amp_plot = s_amp[f_mask_s, :]

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# 1. 时域
axes[0, 0].plot(t_total, trace_down, 'k', lw=0.8)
axes[0, 0].set_title("Time Domain")
axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].set_xlim(t_total.min(), t_total.max())
axes[0, 0].grid(True)

# 2. 频谱
axes[0, 1].plot(freq_pos, amp_pos, 'b', lw=1)
axes[0, 1].set_xlim(0, f_max_plot)
axes[0, 1].set_title("Frequency Spectrum")
axes[0, 1].set_xlabel("Frequency (Hz)")
axes[0, 1].grid(True)

im1 = axes[1, 0].imshow(
    stft_amp_plot,
    cmap='jet',
    aspect='auto',
    origin='lower',
    extent=[t_stft.min(), t_stft.max(), f_stft_plot.min(), f_stft_plot.max()]
)
axes[1, 0].set_title("STFT")
axes[1, 0].set_xlabel("Time (s)")
axes[1, 0].set_ylabel("Frequency (Hz)")
axes[1, 0].set_ylim(0, f_max_plot)
fig.colorbar(im1, ax=axes[1, 0])

im2 = axes[1, 1].imshow(
    s_amp_plot,
    cmap='jet',
    aspect='auto',
    origin='lower',
    extent=[t_total.min(), t_total.max(), f_s_plot.min(), f_s_plot.max()]
)
axes[1, 1].set_title("S-Transform")
axes[1, 1].set_xlabel("Time (s)")
axes[1, 1].set_ylabel("Frequency (Hz)")
axes[1, 1].set_ylim(0, f_max_plot)
fig.colorbar(im2, ax=axes[1, 1])

plt.tight_layout()
plt.show()