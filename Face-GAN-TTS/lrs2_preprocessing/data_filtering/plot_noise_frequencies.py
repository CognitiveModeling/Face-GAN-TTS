import os
import sys
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

# -------------------- Thesis-kompatibler Plotstil --------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "cmr10", "DejaVu Serif", "serif"],
    "mathtext.fontset": "cm",
    "font.size": 24,
    "axes.titlesize": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "axes.unicode_minus": False,
    "axes.formatter.use_mathtext": True,
})

# --- IMPORT CONFIG ---
sys.path.append("/mnt/lustre/work/butz/bst080/faceGANtts")
from config import ex

@ex.automain
def main(_config):
    # --- SINGLE FILE ---
    wav_path = "/mnt/lustre/work/butz/bst080/data/mvlrs_v1/lrs2_splitted/wav/test/spk41/00006.wav"
    output_single = "/mnt/lustre/work/butz/bst080/faceGANtts/lrs2_preprocessing/data_filtering/filter_plots/00006_spectrogram.pdf"
    os.makedirs(os.path.dirname(output_single), exist_ok=True)

    waveform, sr = torchaudio.load(wav_path)
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    window = torch.hann_window(_config["win_len"], device=waveform.device)
    spec_complex = torch.stft(
        input=waveform,
        n_fft=_config["n_fft"],
        hop_length=_config["hop_len"],
        win_length=_config["win_len"],
        window=window,
        return_complex=True
    )
    spec_db = 20.0 * torch.log10(spec_complex.abs() + 1e-8)

    plt.figure(figsize=(10, 4), constrained_layout=True)
    plt.imshow(spec_db[0].numpy(), origin="lower", aspect="auto", cmap="magma")
    plt.title("Spectrogram in dB (00006.wav)")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency Bins")
    cbar = plt.colorbar(label="Amplitude (dB)")
    cbar.ax.tick_params(labelsize=20)
    plt.savefig(output_single, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Single plot saved at: {output_single}")

    # --- ALL FILES ---
    base_dir = "/mnt/lustre/work/butz/bst080/data/mvlrs_v1/lrs2_splitted/wav/trainval"
    output_mean = "/mnt/lustre/work/butz/bst080/faceGANtts/lrs2_preprocessing/data_filtering/filter_plots/mean_spectrogram_trainval.pdf"
    output_median = "/mnt/lustre/work/butz/bst080/faceGANtts/lrs2_preprocessing/data_filtering/filter_plots/median_spectrogram_trainval.pdf"
    os.makedirs(os.path.dirname(output_mean), exist_ok=True)

    wav_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".wav"):
                wav_files.append(os.path.join(root, f))
    print(f"Found WAV files: {len(wav_files)} (using max. 2000)")
    wav_files = wav_files[:2000]

    n_fft = _config["n_fft"]
    hop_length = _config["hop_len"]
    win_length = _config["win_len"]

    all_specs = []
    for path in tqdm(wav_files, desc="Processing audio files"):
        try:
            wf, _ = torchaudio.load(path)
            waveform = torch.mean(wf, dim=0, keepdim=True) if wf.size(0) > 1 else wf
            window = torch.hann_window(win_length, device=waveform.device)
            spec_complex = torch.stft(
                input=waveform,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                return_complex=True
            )
            spec_db = 20.0 * torch.log10(spec_complex.abs() + 1e-8)[0]
            all_specs.append(spec_db.numpy())
        except Exception as e:
            print(f"Error in {path}: {e}")

    min_time = min(s.shape[1] for s in all_specs)
    all_specs = np.stack([s[:, :min_time] for s in all_specs], axis=0)
    mean_spec = np.mean(all_specs, axis=0)
    median_spec = np.median(all_specs, axis=0)

    for data, title, out_path in [
        (mean_spec, "Mean Spectrogram", output_mean),
        (median_spec, "Median Spectrogram", output_median)
    ]:
        plt.figure(figsize=(10, 4), constrained_layout=True)
        plt.imshow(data, origin="lower", aspect="auto", cmap="magma")
        plt.title(f"{title} (trainval)")
        plt.xlabel("Time Frames")
        plt.ylabel("Frequency Bins")
        cbar = plt.colorbar(label="Amplitude (dB)")
        cbar.ax.tick_params(labelsize=20)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"{title} saved at: {out_path}")

    # --- PEAK & Q-VALUE ANALYSIS ---
    mean_energy_per_bin = np.mean(mean_spec, axis=1)
    peak_bin = int(np.argmax(mean_energy_per_bin))
    bin_hz = (sr // 2) / mean_spec.shape[0]
    peak_freq_hz = round(peak_bin * bin_hz, 2)
    print(f"Dominant frequency (Peak): Bin {peak_bin} → {peak_freq_hz} Hz")
    peak_db = mean_energy_per_bin[peak_bin]
    threshold = peak_db - 3.0
    lower_bin, upper_bin = peak_bin, peak_bin
    while lower_bin > 0 and mean_energy_per_bin[lower_bin] >= threshold:
        lower_bin -= 1
    while upper_bin < len(mean_energy_per_bin)-1 and mean_energy_per_bin[upper_bin] >= threshold:
        upper_bin += 1
    bandwidth_hz = (upper_bin - lower_bin) * bin_hz
    q_value = round(peak_freq_hz / bandwidth_hz, 2) if bandwidth_hz else 1.0
    print(f"Q-Value: {q_value}")

    # --- BEFORE-AFTER COMPARISON ---
    wav_before = os.path.join(base_dir, "spk09", "00009.wav")
    test_dir = "/mnt/lustre/work/butz/bst080/faceGANtts/lrs2_preprocessing/data_filtering/test_preprocessed_wavs"
    wav_after = os.path.join(test_dir, "spk09", "00009.wav")
    os.makedirs(os.path.dirname(wav_after), exist_ok=True)
    compare_out = "/mnt/lustre/work/butz/bst080/faceGANtts/lrs2_preprocessing/data_filtering/filter_plots/compare_spectrograms_00009.pdf"
    os.makedirs(os.path.dirname(compare_out), exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True, constrained_layout=True)
    for i, (wav_file, title) in enumerate([(wav_before, "Before Processing (Original)"), (wav_after, "After Processing")]):
        waveform, _ = torchaudio.load(wav_file)
        waveform = torch.mean(waveform, dim=0, keepdim=True) if waveform.size(0) > 1 else waveform
        window = torch.hann_window(win_length, device=waveform.device)
        spec_complex = torch.stft(
            input=waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True
        )
        spec_db = 20.0 * torch.log10(spec_complex.abs() + 1e-8)[0].numpy()
        im = axs[i].imshow(spec_db, origin="lower", aspect="auto", cmap="magma")
        axs[i].set(title=title, xlabel="Time Frames")
        if i == 0:
            axs[i].set_ylabel("Frequency Bins")
    cbar = fig.colorbar(im, ax=axs, label="Amplitude (dB)")
    cbar.ax.tick_params(labelsize=20)
    plt.savefig(compare_out, dpi=300, bbox_inches="tight")
    plt.close()

    # --- DIFFERENCE SPECTROGRAM ---
    try:
        waveform_before, _ = torchaudio.load(wav_before)
        waveform_after, _ = torchaudio.load(wav_after)
        waveform_before = torch.mean(waveform_before, dim=0, keepdim=True) if waveform_before.size(0)>1 else waveform_before
        waveform_after = torch.mean(waveform_after, dim=0, keepdim=True) if waveform_after.size(0)>1 else waveform_after
        spec_before = 20.0 * torch.log10(torch.stft(
            input=waveform_before,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length),
            return_complex=True
        ).abs() + 1e-8)[0].numpy()
        spec_after = 20.0 * torch.log10(torch.stft(
            input=waveform_after,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length),
            return_complex=True
        ).abs() + 1e-8)[0].numpy()
        min_time = min(spec_before.shape[1], spec_after.shape[1])
        diff_spec = spec_after[:,:min_time] - spec_before[:,:min_time]
        plt.figure(figsize=(14,5), constrained_layout=True)
        plt.imshow(diff_spec, origin="lower", aspect="auto", cmap="seismic", vmin=-10, vmax=10)
        plt.title("Difference Spectrogram (Original - After Filter)")
        plt.xlabel("Time Frames")
        plt.ylabel("Frequency Bins")
        cbar = plt.colorbar(label="Δ Amplitude (dB)")
        cbar.ax.tick_params(labelsize=20)
        plt.savefig("/mnt/lustre/work/butz/bst080/faceGANtts/lrs2_preprocessing/data_filtering/filter_plots/diff_spectrogram_00009.pdf", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Error in diff spectrogram: {e}")

    # --- FRAME-WISE ENERGY ---
    energies = {}                         # Sammel-Dict anlegen

    for version_name, base_dir in [("original", "/mnt/lustre/work/butz/bst080/data/mvlrs_v1/lrs2_splitted/wav/trainval"),
                                   ("denoised", test_dir)]:
        print(f"Processing version: {version_name}")
        wav_files = []
        for root, _, files in os.walk(base_dir):
            for f in files:
                if f.endswith(".wav"):
                    wav_files.append(os.path.join(root, f))
        wav_files = wav_files[:2000]
        all_specs = []
        for path in tqdm(wav_files, desc=f"Loading {version_name}"):
            try:
                wf, _ = torchaudio.load(path)
                waveform = torch.mean(wf, dim=0, keepdim=True) if wf.size(0)>1 else wf
                spec_complex = torch.stft(
                    input=waveform,
                    n_fft=_config["n_fft"],
                    hop_length=_config["hop_len"],
                    win_length=_config["win_len"],
                    window=torch.hann_window(_config["win_len"]),
                    return_complex=True
                )
                spec_db = 20.0 * torch.log10(spec_complex.abs() + 1e-8)[0].numpy()
                all_specs.append(spec_db)
            except Exception as e:
                print(f"Error in {path}: {e}")
        min_time = min(s.shape[1] for s in all_specs)
        all_specs = np.stack([s[:, :min_time] for s in all_specs], axis=0)
        frame_energy = np.mean(all_specs, axis=1)
        mean_frame_energy = np.mean(frame_energy, axis=0)
        energies[version_name.capitalize()] = mean_frame_energy
    fig, axes = plt.subplots(2, 1, figsize=(10, 6),
                         sharex=True, constrained_layout=True)

    # Original –
    axes[0].plot(energies['Original'], label='Original')
    axes[0].axvline(len(energies['Original'])-1, linestyle='--', label='Last frame')
    axes[0].set_ylabel('Mean dB Energy')        # y-Label nur hier
    axes[0].set_title('Frame-wise Energy')
    axes[0].grid(True)
    axes[0].legend(loc='upper right')

    # Denoised –
    axes[1].plot(energies['Denoised'], label='Denoised', color='green')
    axes[1].axvline(len(energies['Denoised'])-1, linestyle='--')
    axes[1].set_xlabel('Time Frame')            # gemeinsame x-Achse
    axes[1].grid(True)
    axes[1].legend(loc='upper right')

    plt.savefig("/mnt/lustre/work/butz/bst080/faceGANtts/lrs2_preprocessing/data_filtering/"
            "filter_plots/framewise_energy_subplots.pdf",
            dpi=300, bbox_inches="tight")
    plt.close()