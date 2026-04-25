import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import stft

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

# 硬编码采样间隔（降采样后为 0.001 s，未滤波）
dt = 0.001
print(f"采样间隔 dt = {dt} s，采样率 = {1/dt:.1f} Hz")

nx = model_data["coarse"]["nx"]
nt = params_data["base"]["nt"]

# 读取数据并裁剪边界（去掉 PML / halo）
file = np.fromfile(FILE_DIR, dtype=np.float32).reshape((-1, nx))
file = file[:, 3:nx-4]                     # 裁剪 PML/halo
trace = file[:, params_data["base"]["posx"]-3].astype(np.float64) * 1e6   # 提取一道并缩放

# ===================== 不滤波，直接使用原始 trace =====================
trace_raw = trace   # 无滤波
trace_raw[0:150] = trace_raw[150]
# 时间轴
t_total = np.arange(len(trace_raw)) * dt
fs = 1/dt

# 频谱
freq_fft = fftfreq(len(trace_raw), dt)
amp_fft = np.abs(fft(trace_raw))
freq_pos = freq_fft[freq_fft >= 0]
amp_pos = amp_fft[freq_fft >= 0]

# STFT
f_stft, t_stft, Z_stft = stft(
    trace_raw, 
    fs=fs, 
    nperseg=64, 
    noverlap=60, 
    nfft=128
)
stft_amp = np.abs(Z_stft)
f_max_plot = 150
f_mask_stft = f_stft <= f_max_plot
f_stft_plot = f_stft[f_mask_stft]
stft_amp_plot = stft_amp[f_mask_stft, :]

# S 变换
f_s = np.linspace(0, fs/2, len(trace_raw)//2)
STx = stran(trace_raw, t_total, f_s)
s_amp = np.abs(STx)
f_mask_s = f_s <= f_max_plot
f_s_plot = f_s[f_mask_s]
s_amp_plot = s_amp[f_mask_s, :]

# 绘图
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# 1. 时域
axes[0, 0].plot(t_total, trace_raw, 'k', lw=0.8)
axes[0, 0].set_title("Time Domain (No Filter)")
axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].set_xlim(t_total.min(), t_total.max())
axes[0, 0].set_ylim(-0.0025, 0.0025)
axes[0, 0].grid(True)

# 2. 频谱
axes[0, 1].plot(freq_pos, amp_pos, 'b', lw=1)
axes[0, 1].set_xlim(0, f_max_plot)
axes[0, 1].set_ylim(0, 0.03)
axes[0, 1].set_title("Frequency Spectrum")
axes[0, 1].set_xlabel("Frequency (Hz)")
axes[0, 1].grid(True)

# 3. STFT
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

# 4. S-Transform
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