import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

FONT_LABEL = 12
FONT_TITLE = 14
FONT_TICK = 10
FONT_LEGEND = 10

TW_RMS = (0.2, 0.6)
TW_FFT = (0.2, 0.6)
TW_ST = (0.2, 0.6)
F_MAX_PLOT = 100.0

DT = 0.001
FS = 1.0 / DT

FILE_DIR = [
    "./crack/crack0.bin",
    "./crack/crack10.bin",
    "./crack/crack20.bin"
]

LABELS = [
    "0",
    "10",
    "20"
]

PARAMS_DIR = [
    {"nx": 501, "nz": 551, "dh": 1.5},
    {"nx": 501, "nz": 551, "dh": 1.5},
    {"nx": 501, "nz": 551, "dh": 1.5}
]

POSX = [250, 250, 250]

FREQ_BANDS = [
    (0, 40),
    (41, 100)
]

def slice_by_window(trace_orig, t_full, window, dt):
    if window is not None:
        t_start, t_end = window
        idx_start = max(0, int(np.round(t_start / dt)))
        idx_end = min(len(trace_orig), int(np.round(t_end / dt)))
        return (
            trace_orig[idx_start:idx_end].copy(),
            t_full[idx_start:idx_end].copy()
        )
    return trace_orig.copy(), t_full.copy()

def calculate_rms_profile(data, dt, dx, time_window):
    if time_window is not None:
        t_start, t_end = time_window
        idx_start = max(0, int(np.round(t_start / dt)))
        idx_end = min(data.shape[0], int(np.round(t_end / dt)))
        win_data = data[idx_start:idx_end, :]
    else:
        win_data = data
    rms_values = np.sqrt(np.mean(np.square(win_data), axis=0))
    dist_x = (np.arange(len(rms_values)) - len(rms_values) // 2) * dx
    return dist_x, rms_values

def spectral_centroid(freq, amp):
    numerator = np.sum(freq * amp)
    denominator = np.sum(amp)
    if denominator <= 1e-14:
        return 0.0
    return numerator / denominator

def stran(x, t, f):
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
        gauss = (f_abs / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (f_col ** 2) * (td ** 2))
        phase = np.exp(-2j * np.pi * f_col * t_row)
        STx[:, i] = np.sum(x * gauss * phase, axis=1) * dt
    return STx

def main():
    num_files = len(FILE_DIR)
    data_records = []
    centroid_list = []
    st_results = []
    global_st_max = 0.0

    for i in range(num_files):
        nx_val = PARAMS_DIR[i]["nx"]
        dh_val = PARAMS_DIR[i]["dh"]
        cpml = 20
        temp = np.fromfile(FILE_DIR[i], dtype=np.float32).reshape((-1, nx_val))
        temp = temp[:, 3 + cpml:nx_val - 4 - cpml]
        trace_orig = (temp[:, POSX[i] - 3 - cpml].copy().astype(np.float64) * 1e7)
        t_full = np.arange(len(trace_orig)) * DT

        trace_fft, t_fft = slice_by_window(trace_orig, t_full, TW_FFT, DT)
        freq_fft = fftfreq(len(trace_fft), DT)
        amp_fft = np.abs(fft(trace_fft))
        freq_pos = freq_fft[freq_fft >= 0]
        amp_pos = amp_fft[freq_fft >= 0]
        mask_plot = ((freq_pos >= 0) & (freq_pos <= F_MAX_PLOT))
        freq_plot = freq_pos[mask_plot]
        amp_plot = amp_pos[mask_plot]

        centroid = spectral_centroid(freq_plot, amp_plot)
        centroid_list.append(centroid)

        dist_x, rms_values = calculate_rms_profile(temp.astype(np.float64), DT, dh_val, TW_RMS)

        trace_st, t_st = slice_by_window(trace_orig, t_full, TW_ST, DT)
        f_s = np.linspace(0, FS / 2, len(trace_st) // 2)
        STx = stran(trace_st, t_st, f_s)
        s_amp = np.abs(STx)
        f_mask_s = (f_s <= F_MAX_PLOT)
        f_s_plot = f_s[f_mask_s]
        s_amp_plot = s_amp[f_mask_s, :]
        global_st_max = max(global_st_max, np.max(s_amp_plot))
        st_results.append({
            "t": t_st,
            "f": f_s_plot,
            "amp": s_amp_plot
        })

        data_records.append({
            "label": LABELS[i],
            "t_fft": t_fft,
            "trace_fft": trace_fft,
            "freq_pos": freq_plot,
            "amp_pos": amp_plot,
            "dist_x": dist_x,
            "rms_values": rms_values
        })

    # 全模型：时域 + 频谱
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    for rec in data_records:
        axes1[0].plot(rec['t_fft'], rec['trace_fft'], linewidth=1.5, label=rec['label'])
    axes1[0].set_title("Time Domain Waveforms", fontsize=FONT_TITLE)
    axes1[0].set_xlabel("Time (s)", fontsize=FONT_LABEL)
    axes1[0].set_ylabel("Amplitude", fontsize=FONT_LABEL)
    axes1[0].tick_params(labelsize=FONT_TICK)
    axes1[0].grid(True, linestyle='--', alpha=0.5)
    axes1[0].legend(fontsize=FONT_LEGEND)
    axes1[0].text(0.5, -0.18, '(a)', transform=axes1[0].transAxes, ha='center', va='top', fontsize=14)
    for rec in data_records:
        axes1[1].plot(rec['freq_pos'], rec['amp_pos'], linewidth=1.5, label=rec['label'])
    axes1[1].set_title("Amplitude Spectra", fontsize=FONT_TITLE)
    axes1[1].set_xlabel("Frequency (Hz)", fontsize=FONT_LABEL)
    axes1[1].set_ylabel("Amplitude", fontsize=FONT_LABEL)
    axes1[1].set_xlim(0, F_MAX_PLOT)
    axes1[1].tick_params(labelsize=FONT_TICK)
    axes1[1].grid(True, linestyle='--', alpha=0.5)
    axes1[1].legend(fontsize=FONT_LEGEND)
    axes1[1].text(0.5, -0.18, '(b)', transform=axes1[1].transAxes, ha='center', va='top', fontsize=14)
    fig1.tight_layout()

    # 全模型 RMS
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for rec in data_records:
        ax2.plot(rec['dist_x'], rec['rms_values'], linewidth=2, label=rec['label'])
    ax2.set_title("Surface RMS Amplitude", fontsize=FONT_TITLE)
    ax2.set_xlabel("Offset (m)", fontsize=FONT_LABEL)
    ax2.set_ylabel("RMS Amplitude", fontsize=FONT_LABEL)
    ax2.tick_params(labelsize=FONT_TICK)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=FONT_LEGEND)
    fig2.tight_layout()

    # 全模型 S变换
    fig3, axes3 = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    axes3 = axes3.flatten()
    for i in range(num_files):
        ax = axes3[i]
        st_data = st_results[i]
        im = ax.imshow(
            st_data["amp"],
            cmap='turbo',
            aspect='auto',
            origin='lower',
            extent=[st_data["t"].min(), st_data["t"].max(), st_data["f"].min(), st_data["f"].max()],
            vmin=0, vmax=global_st_max
        )
        ax.set_title(f"N_cracks={LABELS[i]}", fontsize=FONT_TITLE)
        ax.set_xlabel("Time (s)", fontsize=FONT_LABEL)
        ax.set_ylabel("Frequency (Hz)", fontsize=FONT_LABEL)
        ax.tick_params(labelsize=FONT_TICK, labelleft=True, labelbottom=True)
        ax.text(0.5, -0.18, f'({chr(97+i)})', transform=ax.transAxes, ha='center', va='top', fontsize=14)
        cbar = fig3.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
        cbar.ax.tick_params(labelsize=9)
    fig3.tight_layout()

    # 谱重心
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.plot(LABELS, centroid_list, marker='o', linewidth=2)
    ax4.set_title("Spectral Centroid", fontsize=FONT_TITLE)
    ax4.set_ylabel("Centroid Frequency (Hz)", fontsize=FONT_LABEL)
    ax4.tick_params(labelsize=FONT_TICK)
    ax4.grid(True, linestyle='--', alpha=0.5)
    fig4.tight_layout()

    # 频谱比
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    ref_amp = data_records[1]["amp_pos"]
    ref_freq = data_records[1]["freq_pos"]
    for i in range(2, num_files):
        amp_pos = data_records[i]["amp_pos"]
        ratio = (amp_pos + 1e-14) / (ref_amp + 1e-14)
        ax5.plot(ref_freq, ratio, linewidth=2, label=f"{LABELS[i]} / 0%")
    ax5.set_title("Linear Spectral Ratio", fontsize=FONT_TITLE)
    ax5.set_xlabel("Frequency (Hz)", fontsize=FONT_LABEL)
    ax5.set_ylabel("Amplitude Ratio", fontsize=FONT_LABEL)
    ax5.set_xlim(0, F_MAX_PLOT)
    ax5.tick_params(labelsize=FONT_TICK)
    ax5.grid(True, linestyle='--', alpha=0.5)
    ax5.legend(fontsize=FONT_LEGEND)
    fig5.tight_layout()

    # 频段占比
    bands_labels = [f"{b[0]}-{b[1]} Hz" for b in FREQ_BANDS]
    relative_contents = []
    for rec in data_records:
        freq = rec["freq_pos"]
        amp = rec["amp_pos"]
        mask_total = ((freq >= 0) & (freq <= 100))
        total_amp = np.sum(amp[mask_total])
        rel = []
        for low, high in FREQ_BANDS:
            mask_band = ((freq >= low) & (freq <= high))
            band_amp = np.sum(amp[mask_band])
            rel.append(band_amp / total_amp)
        relative_contents.append(rel)
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    x_indexes = np.arange(len(FREQ_BANDS))
    total_width = 0.7
    bar_width = total_width / num_files
    for idx in range(num_files):
        current_x = (x_indexes - total_width / 2 + (idx + 0.5) * bar_width)
        ax6.bar(current_x, relative_contents[idx], width=bar_width, label=LABELS[idx])
    ax6.set_title("Relative Frequency Band Content", fontsize=FONT_TITLE)
    ax6.set_xticks(x_indexes)
    ax6.set_xticklabels(bands_labels, fontsize=FONT_TICK)
    ax6.set_ylabel("Relative Content", fontsize=FONT_LABEL)
    ax6.tick_params(labelsize=FONT_TICK)
    ax6.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax6.legend(fontsize=FONT_LEGEND)
    fig6.tight_layout()

    plt.show()

if __name__ == '__main__':
    main()