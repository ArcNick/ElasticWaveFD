import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# ===================== 1. Data Loading & Parameters =====================
FILE1_DIR = "./Q200_f.bin"
FILE2_DIR = "./Q50_f.bin"
FILE3_DIR = "./Q100_f.bin"
PARAMS_DIR = "./models/params.json"
MODEL_DIR = "./models/models.json"

with open(PARAMS_DIR, 'r') as f: params_data = json.load(f)
with open(MODEL_DIR, 'r') as f: model_data = json.load(f)

# 严格锁定指定的物理参数
Vp = 1300.0     # P波速度 (m/s)
f0 = 30.0       # 震源主频 f0 (Hz)
L = 260.0       # 传播距离 (m)
dt = 0.001      # 时间采样间隔 (s)

nx = model_data["coarse"]["nx"]

# 读取并裁剪边界
file1 = np.fromfile(FILE1_DIR, dtype=np.float32).reshape((-1, nx))
file2 = np.fromfile(FILE2_DIR, dtype=np.float32).reshape((-1, nx))
file3 = np.fromfile(FILE3_DIR, dtype=np.float32).reshape((-1, nx))

# 提取相同位置的单道信号 (已经对齐你的网格裁剪逻辑)
posx_idx = params_data["base"]["posx"] - 3
trace1 = file1[:, 3:nx-4][:, posx_idx].astype(np.float64) * 1e6  # Q200
trace2 = file2[:, 3:nx-4][:, posx_idx].astype(np.float64) * 1e6  # Q50
trace3 = file3[:, 3:nx-4][:, posx_idx].astype(np.float64) * 1e6  # Q100

t_total = np.arange(len(trace1)) * dt

# ===================== 2. 原生有限点 FFT (绝不插值) =====================
freq_fft = fftfreq(len(trace1), dt)

amp1_fft = np.abs(fft(trace1))
amp2_fft = np.abs(fft(trace2))
amp3_fft = np.abs(fft(trace3))

freq_pos = freq_fft[freq_fft >= 0]
amp1_pos = amp1_fft[freq_fft >= 0]
amp2_pos = amp2_fft[freq_fft >= 0]
amp3_pos = amp3_fft[freq_fft >= 0]

points_in_100hz = np.sum(freq_pos <= 100)
print("=" * 60)
print(f"信号真实总点数 N = {len(trace1)}")
print(f"在 0-100 Hz 范围内的原生离散计算点数: {points_in_100hz} 个")
print("=" * 60)

# ===================== 3. 计算主频 =====================
def find_dominant_frequency(freq, amplitude):
    dominant_idx = np.argmax(amplitude)
    return freq[dominant_idx]

dom1 = find_dominant_frequency(freq_pos, amp1_pos)
dom2 = find_dominant_frequency(freq_pos, amp2_pos)
dom3 = find_dominant_frequency(freq_pos, amp3_pos)

print(f"Q=200: Dominant Frequency = {dom1:.2f} Hz")
print(f"Q=100: Dominant Frequency = {dom3:.2f} Hz")
print(f"Q=50:  Dominant Frequency = {dom2:.2f} Hz")
print("=" * 60)

# ===================== 4. 流体幅比理论曲线 (新组合: 50/200, 50/100, 100/200) =====================
eps = 1e-10
ratio_50_200_measured  = amp2_pos / (amp1_pos + eps)
ratio_100_200_measured = amp3_pos / (amp1_pos + eps)
ratio_50_100_measured  = amp2_pos / (amp3_pos + eps)

theory_50_200  = np.exp(-np.pi * L / (Vp * f0) * (1/50 - 1/200)  * (freq_pos**2))
theory_100_200 = np.exp(-np.pi * L / (Vp * f0) * (1/100 - 1/200) * (freq_pos**2))
theory_50_100  = np.exp(-np.pi * L / (Vp * f0) * (1/50 - 1/100)  * (freq_pos**2))

# ===================== 5. 绘图部分 (严格限定 0-100Hz) =====================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ---- Panel 1: 时域信号 (全时间显示) ----
axes[0].plot(t_total, trace1, 'b', lw=0.8, label='Q=200')
axes[0].plot(t_total, trace3, 'g', lw=0.8, label='Q=100')
axes[0].plot(t_total, trace2, 'r', lw=0.8, label='Q=50')
y_max = max(np.max(np.abs(trace1)), np.max(np.abs(trace2)), np.max(np.abs(trace3)))
axes[0].set_ylim(-y_max * 1.1, y_max * 1.1)
axes[0].set_xlim(0.15, 0.35)
axes[0].set_title("Full Time Domain")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True)
axes[0].legend()

# ---- Panel 2: 频谱曲线 ----
axes[1].plot(freq_pos, amp1_pos, 'b-', lw=1.2, label=f'Q=200 (dom={dom1:.1f}Hz)')
axes[1].plot(freq_pos, amp3_pos, 'g-', lw=1.2, label=f'Q=100 (dom={dom3:.1f}Hz)')
axes[1].plot(freq_pos, amp2_pos, 'r-', lw=1.2, label=f'Q=50 (dom={dom2:.1f}Hz)')
axes[1].set_xlim(0, 100)  
axes[1].set_title("Frequency Spectrum")
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Amplitude")
axes[1].grid(True)
axes[1].legend()

# ---- Panel 3: 幅比验证 (散点放大且压在理论线上方) ----
# 红色：50 对比 200
axes[2].plot(freq_pos, ratio_50_200_measured, 'ro', markersize=5, zorder=3, label='Measured 50/200')
axes[2].plot(freq_pos, theory_50_200, 'k--', lw=1.0, zorder=2, label='Theory 50/200')

# 绿色：100 对比 200
axes[2].plot(freq_pos, ratio_100_200_measured, 'go', markersize=5, zorder=3, label='Measured 100/200')
axes[2].plot(freq_pos, theory_100_200, 'b--', lw=1.0, zorder=2, label='Theory 100/200')

# 品红：50 对比 100
axes[2].plot(freq_pos, ratio_50_100_measured, 'mo', markersize=5, zorder=3, label='Measured 50/100')
axes[2].plot(freq_pos, theory_50_100, 'y--', lw=1.0, zorder=2, label='Theory 50/100')

axes[2].set_xlim(0, 100)  
axes[2].set_ylim(0, 1.1)
axes[2].set_title("Fluid Verification")
axes[2].set_xlabel("Frequency (Hz)")
axes[2].set_ylabel("Amplitude Ratio")
axes[2].grid(True)
axes[2].legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.show()