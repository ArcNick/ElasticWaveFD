import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# ==================== 全局配置 ====================
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

TW_RMS = (0.2, 0.6)       # RMS 剖面使用的时窗
TW_FFT = (0.2, 0.6)       # 时域波形与 FFT 频谱使用的时窗
F_MAX_PLOT = 100.0         # 频谱显示的最大频率 (Hz)
DT = 0.001
FS = 1 / DT

# 🔥【新增强大功能：自由配置频段】
# 你可以随意修改、增加或减少频段。格式为 (下限, 上限)
# 代码会自动统计这些频段在“总考核范围（所有频段的最小下限到最大上限）”内的相对含量。
FREQ_BANDS = [
    (0, 45),
    (46, 90)
]

# 【支持任意数量文件】只需在此处添加或减少元素，后续绘图会自动适配数量
FILE_DIR = ["./0.bin", "./5.bin", "./10.bin", "./10+sch.bin", "./15.bin", "./20.bin"]
LABELS = ["0%", "5%", "10%", "10+sch", "15%", "20%"]  

PARAMS_DIR = [
    {"nx": 501, "nz": 551, "dh": 1.5},
    {"nx": 501, "nz": 551, "dh": 1.5},
    {"nx": 501, "nz": 551, "dh": 1.5},
    {"nx": 501, "nz": 551, "dh": 1.5},
    {"nx": 501, "nz": 551, "dh": 1.5},
    {"nx": 501, "nz": 551, "dh": 1.5}
]
POSX = [250, 250, 250, 250, 250, 250]


# ==================== 工具函数 ====================
def slice_by_window(trace_orig: np.ndarray, t_full: np.ndarray, window: tuple, dt: float):
    if window is not None:
        t_start, t_end = window
        idx_start = max(0, int(np.round(t_start / dt)))
        idx_end = min(len(trace_orig), int(np.round(t_end / dt)))
        return trace_orig[idx_start:idx_end].copy(), t_full[idx_start:idx_end].copy()
    return trace_orig.copy(), t_full.copy()

def calculate_rms_profile(data: np.ndarray, dt: float, dx: float, time_window: tuple):
    if time_window is not None:
        t_start, t_end = time_window
        idx_start = max(0, int(np.round(t_start / dt)))
        idx_end = min(data.shape[0], int(np.round(t_end / dt)))
        win_data = data[idx_start:idx_end, :]
    else:
        win_data = data
    rms_values = np.sqrt(np.mean(np.square(win_data), axis=0))
    dist_x = np.arange(len(rms_values)) * dx
    return dist_x, rms_values


# ==================== 主程序 ====================
def main():
    num_files = len(FILE_DIR)
    
    # 根据配置的 FREQ_BANDS 自动计算总能量范围的上下限（用于分母归一化）
    all_freqs = [f for band in FREQ_BANDS for f in band]
    f_min_total = min(all_freqs)
    f_max_total = max(all_freqs)
    
    # 自动生成条形图的 X 轴标签（例如 '0-30 Hz'）
    bands_labels = [f"{b[0]}-{b[1]} Hz" for b in FREQ_BANDS]
    
    # 用于集中存储所有文件处理结果的容器
    data_records = []

    # 1. 数据读取与核心计算
    for i in range(num_files):
        nx_val = PARAMS_DIR[i]["nx"]
        dh_val = PARAMS_DIR[i]["dh"]
        cpml = 20
        
        # 读取二进制数据
        temp = np.fromfile(FILE_DIR[i], dtype=np.float32).reshape((-1, nx_val))
        temp = temp[:, 3+cpml:nx_val-4-cpml]
        
        # 提取目标位置处的地震道
        trace_orig = temp[:, POSX[i]-3-cpml].copy().astype(np.float64) * 1e7
        t_full = np.arange(len(trace_orig)) * DT
        
        # 计算时窗内的地震道段
        trace_fft, t_fft = slice_by_window(trace_orig, t_full, TW_FFT, DT)
        
        # 计算傅里叶变换 (FFT)
        freq_fft = fftfreq(len(trace_fft), DT)
        amp_fft = np.abs(fft(trace_fft))
        freq_pos = freq_fft[freq_fft >= 0]
        amp_pos = amp_fft[freq_fft >= 0]
        
        # 计算总范围（例如 0 - 90Hz）内的振幅积分作为分母
        mask_total = (freq_pos >= f_min_total) & (freq_pos <= f_max_total)
        total_amp = np.sum(amp_pos[mask_total])
        
        # 动态计算每个自定义频段的相对含量
        rel_content = []
        if total_amp > 0:
            # ✨ 修正这里的语法：直接用 low, high 接收频段的上下限
            for low, high in FREQ_BANDS:
                # 包含下限，小于等于上限
                mask_band = (freq_pos >= low) & (freq_pos <= high)
                band_amp = np.sum(amp_pos[mask_band])
                rel_content.append(band_amp / total_amp)
        else:
            rel_content = [0.0] * len(FREQ_BANDS)
            
        # 计算 RMS 剖面
        dist_x, rms_values = calculate_rms_profile(temp.astype(np.float64), DT, dh_val, TW_RMS)
        
        # 暂存所有绘图需要的数据
        data_records.append({
            'label': LABELS[i],
            't_fft': t_fft,
            'trace_fft': trace_fft,
            'freq_pos': freq_pos,
            'amp_pos': amp_pos,
            'rel_content': rel_content,
            'dist_x': dist_x,
            'rms_values': rms_values
        })

    # ==================== 开始绘图 ====================

    # ---- Figure 1: 叠放显示（两行一列：上为时间波形，下为振幅谱） ----
    fig1, axes1 = plt.subplots(2, 1, figsize=(10, 8))
    for rec in data_records:
        axes1[0].plot(rec['t_fft'], rec['trace_fft'], lw=1, label=rec['label'])
        axes1[1].plot(rec['freq_pos'], rec['amp_pos'], lw=1.2, label=rec['label'])
    
    axes1[0].set_title("Combined Time Domain Waveforms")
    axes1[0].set_xlabel("Time (s)")
    axes1[0].set_ylabel("Amplitude")
    axes1[0].grid(True, linestyle='--', alpha=0.5)
    axes1[0].legend()
    
    axes1[1].set_title("Combined Amplitude Spectra")
    axes1[1].set_xlabel("Frequency (Hz)")
    axes1[1].set_ylabel("Amplitude")
    axes1[1].set_xlim(0, F_MAX_PLOT)
    axes1[1].grid(True, linestyle='--', alpha=0.5)
    axes1[1].legend()
    fig1.tight_layout()


    # ---- Figure 2: 单独放地震道（横向平铺子图，共享纵轴范围） ----
    fig2, axes2 = plt.subplots(1, num_files, figsize=(4 * num_files, 4), sharey=True, squeeze=False)
    axes2 = axes2.flatten()
    for idx, rec in enumerate(data_records):
        axes2[idx].plot(rec['t_fft'], rec['trace_fft'], lw=1, color='C'+str(idx))
        axes2[idx].set_title(f"Trace: {rec['label']}")
        axes2[idx].set_xlabel("Time (s)")
        if idx == 0:
            axes2[idx].set_ylabel("Amplitude")
        axes2[idx].grid(True, linestyle='--', alpha=0.5)
    fig2.tight_layout()


    # ---- Figure 3: 单独放振幅谱（横向平铺子图，共享纵轴范围） ----
    fig3, axes3 = plt.subplots(1, num_files, figsize=(4 * num_files, 4), sharey=True, squeeze=False)
    axes3 = axes3.flatten()
    for idx, rec in enumerate(data_records):
        axes3[idx].plot(rec['freq_pos'], rec['amp_pos'], lw=1.2, color='C'+str(idx))
        axes3[idx].set_title(f"Spectrum: {rec['label']}")
        axes3[idx].set_xlabel("Frequency (Hz)")
        axes3[idx].set_xlim(0, F_MAX_PLOT)
        if idx == 0:
            axes3[idx].set_ylabel("Amplitude")
        axes3[idx].grid(True, linestyle='--', alpha=0.5)
    fig3.tight_layout()


    # ---- Figure 4: 动态频段相对含量多组条形统计图 ----
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    x_indexes = np.arange(len(FREQ_BANDS))  # 基础组位置
    
    # 动态计算柱子总宽度与单根宽度
    total_width = 0.7
    bar_width = total_width / num_files
    
    for idx, rec in enumerate(data_records):
        # 计算当前文件柱子的横坐标偏移
        current_x = x_indexes - (total_width / 2) + (idx + 0.5) * bar_width
        ax4.bar(current_x, rec['rel_content'], width=bar_width, label=rec['label'])
        
    ax4.set_title(f"Relative Frequency Band Content Comparison ({f_min_total}-{f_max_total} Hz)")
    ax4.set_xticks(x_indexes)
    ax4.set_xticklabels(bands_labels)
    ax4.set_xlabel("Frequency Bands")
    ax4.set_ylabel("Relative Content (Ratio)")
    ax4.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax4.legend()
    fig4.tight_layout()


    # ---- Figure 5: RMS 剖面全员叠放 ----
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    for rec in data_records:
        ax5.plot(rec['dist_x'], rec['rms_values'], lw=2, label=rec['label'])
        
    ax5.set_title(f"Surface RMS Amplitude Profile Comparison [{TW_RMS[0]}s - {TW_RMS[1]}s]")
    ax5.set_xlabel("Distance x (m)")
    ax5.set_ylabel("RMS Amplitude")
    ax5.grid(True, linestyle='--', alpha=0.7)
    ax5.legend()
    fig5.tight_layout()

    plt.show()

if __name__ == '__main__':
    main()