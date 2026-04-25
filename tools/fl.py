import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# ===================== Data Loading =====================
FILE1_DIR = "./Q200.bin"
FILE2_DIR = "./Q50.bin"
FILE3_DIR = "./Q20.bin"
PARAMS_DIR = "./models/params.json"
MODEL_DIR = "./models/models.json"

with open(PARAMS_DIR, 'r') as f: params_data = json.load(f)
with open(MODEL_DIR, 'r') as f: model_data = json.load(f)

# Hardcoded sampling interval (0.001 s, no filtering)
dt = 0.001
nx = model_data["coarse"]["nx"]

# Read data and crop boundaries (remove PML / halo)
file1 = np.fromfile(FILE1_DIR, dtype=np.float32).reshape((-1, nx))
file2 = np.fromfile(FILE2_DIR, dtype=np.float32).reshape((-1, nx))
file3 = np.fromfile(FILE3_DIR, dtype=np.float32).reshape((-1, nx))
file1 = file1[:, 3:nx-4]
trace1 = file1[:, params_data["base"]["posx"]-3].astype(np.float64) * 1e6
file2 = file2[:, 3:nx-4]
trace2 = file2[:, params_data["base"]["posx"]-3].astype(np.float64) * 1e6
file3 = file3[:, 3:nx-4]
trace3 = file3[:, params_data["base"]["posx"]-3].astype(np.float64) * 1e6

# Time axis
t_total = np.arange(len(trace1)) * dt
fs = 1 / dt

# Compute frequency spectra
freq_fft = fftfreq(len(trace1), dt)
amp1_fft = np.abs(fft(trace1))
amp2_fft = np.abs(fft(trace2))
amp3_fft = np.abs(fft(trace3))
freq_pos = freq_fft[freq_fft >= 0]
amp1_pos = amp1_fft[freq_fft >= 0]
amp2_pos = amp2_fft[freq_fft >= 0]
amp3_pos = amp3_fft[freq_fft >= 0]

# ===================== Calculate dominant frequency =====================
def find_dominant_frequency(freq, amplitude):
    dominant_idx = np.argmax(amplitude)
    dominant_freq = freq[dominant_idx]
    return dominant_freq

dom1 = find_dominant_frequency(freq_pos, amp1_pos)
dom2 = find_dominant_frequency(freq_pos, amp2_pos)
dom3 = find_dominant_frequency(freq_pos, amp3_pos)

print("=" * 60)
print("Frequency Analysis Results:")
print("-" * 60)
print(f"Q=200: Dominant Frequency = {dom1:.2f} Hz")
print(f"Q=50:  Dominant Frequency = {dom2:.2f} Hz")
print(f"Q=20:  Dominant Frequency = {dom3:.2f} Hz")
print("=" * 60)

freq_mask = freq_pos <= 120
freq_limited = freq_pos[freq_mask]
dom1_lim = find_dominant_frequency(freq_limited, amp1_pos[freq_mask])
dom2_lim = find_dominant_frequency(freq_limited, amp2_pos[freq_mask])
dom3_lim = find_dominant_frequency(freq_limited, amp3_pos[freq_mask])

print("\nFrequency analysis limited to 0-120 Hz:")
print("-" * 60)
print(f"Q=200: Dominant Frequency = {dom1_lim:.2f} Hz")
print(f"Q=50:  Dominant Frequency = {dom2_lim:.2f} Hz")
print(f"Q=20:  Dominant Frequency = {dom3_lim:.2f} Hz")
print("=" * 60)

ylim_time = (-0.1, 0.1)

# ===================== Figure 1: Time + Raw Spectrum =====================
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
axes1[0].plot(t_total, trace1, 'b', lw=0.8, label='Q=200')
axes1[0].plot(t_total, trace2, 'r', lw=0.8, label='Q=50')
axes1[0].plot(t_total, trace3, 'g', lw=0.8, label='Q=20')
axes1[0].set_title("Time Domain Comparison")
axes1[0].set_xlabel("Time (s)")
axes1[0].set_ylabel("Amplitude")
axes1[0].set_xlim(t_total.min(), t_total.max() / 4)
axes1[0].set_ylim(ylim_time)
axes1[0].grid(True)
axes1[0].legend()

axes1[1].plot(freq_pos, amp1_pos, 'b', lw=1, label=f'Q=200 (dom={dom1:.1f}Hz)')
axes1[1].plot(freq_pos, amp2_pos, 'r', lw=1, label=f'Q=50 (dom={dom2:.1f}Hz)')
axes1[1].plot(freq_pos, amp3_pos, 'g', lw=1, label=f'Q=20 (dom={dom3:.1f}Hz)')
axes1[1].set_xlim(0, 120)
axes1[1].set_title("Frequency Spectrum (Raw Amplitude)")
axes1[1].set_xlabel("Frequency (Hz)")
axes1[1].set_ylabel("Amplitude")
axes1[1].grid(True)
axes1[1].legend(loc='upper right')
plt.tight_layout()
plt.show()

# ===================== Amplitude Ratio Preparation =====================
L = 800.0    # 传播距离 (m)
Vp = 4000.0  # P波速度 (m/s)

eps = 1e-10
ratio_50_200_measured = amp2_pos / (amp1_pos + eps)
ratio_20_200_measured = amp3_pos / (amp1_pos + eps)
ratio_20_50_measured = amp3_pos / (amp2_pos + eps)   # 新增：20 vs 50

theory_50_200 = np.exp(-np.pi * freq_pos * L / Vp * (1/50 - 1/200))
theory_20_200 = np.exp(-np.pi * freq_pos * L / Vp * (1/20 - 1/200))
theory_20_50  = np.exp(-np.pi * freq_pos * L / Vp * (1/20 - 1/50))    # 新增理论

# ===================== Figure 2: Improved Time + Spectrum + Ratios =====================
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

# ---- Time domain (auto-scale x to capture main arrival) ----
# 自动寻找第一个明显波峰的位置，使窗口聚焦在信号段
peak_idx = np.argmax(np.abs(trace1))
t_center = t_total[peak_idx]
t_width = 0.1  # 窗口半宽 0.05s，总宽 0.1s
axes2[0].plot(t_total, trace1, 'b', lw=0.8, label='Q=200')
axes2[0].plot(t_total, trace2, 'r', lw=0.8, label='Q=50')
axes2[0].plot(t_total, trace3, 'g', lw=0.8, label='Q=20')
axes2[0].set_title("Time Domain")
axes2[0].set_xlabel("Time (s)")
axes2[0].set_ylabel("Amplitude")
axes2[0].set_xlim(t_center - t_width, t_center + t_width)
# 自动 y 范围（稍加边距）
y_max = max(np.max(np.abs(trace1)), np.max(np.abs(trace2)), np.max(np.abs(trace3)))
axes2[0].set_ylim(-y_max*1.1, y_max*1.1)
axes2[0].grid(True)
axes2[0].legend()

# ---- Frequency spectrum (raw) ----
axes2[1].plot(freq_pos, amp1_pos, 'b', lw=1, label='Q=200')
axes2[1].plot(freq_pos, amp2_pos, 'r', lw=1, label='Q=50')
axes2[1].plot(freq_pos, amp3_pos, 'g', lw=1, label='Q=20')
axes2[1].set_xlim(0, 120)
axes2[1].set_title("Frequency Spectrum")
axes2[1].set_xlabel("Frequency (Hz)")
axes2[1].set_ylabel("Amplitude")
axes2[1].grid(True)
axes2[1].legend()

# ---- Amplitude ratios (including 20 vs 50) ----
axes2[2].plot(freq_pos, ratio_50_200_measured, 'r.', markersize=2, label='Measured 50/200')
axes2[2].plot(freq_pos, theory_50_200, 'r--', lw=1.5, label='Theory 50/200')
axes2[2].plot(freq_pos, ratio_20_200_measured, 'g.', markersize=2, label='Measured 20/200')
axes2[2].plot(freq_pos, theory_20_200, 'g--', lw=1.5, label='Theory 20/200')
axes2[2].plot(freq_pos, ratio_20_50_measured, 'm.', markersize=2, label='Measured 20/50')
axes2[2].plot(freq_pos, theory_20_50, 'm--', lw=1.5, label='Theory 20/50')
axes2[2].set_xlim(5, 100)
axes2[2].set_ylim(0, 1.1)
axes2[2].set_title("Amplitude Ratio Verification")
axes2[2].set_xlabel("Frequency (Hz)")
axes2[2].set_ylabel("Amplitude Ratio")
axes2[2].legend(loc='upper right', fontsize=9)
axes2[2].grid(True)

plt.tight_layout()
plt.show()