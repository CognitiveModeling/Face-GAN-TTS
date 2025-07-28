import os
import sys
import torch
import torchaudio
import torchaudio.functional as AF
import noisereduce as nr
from glob import glob
from tqdm import tqdm
import numpy as np 

sys.path.append("/mnt/lustre/work/butz/bst080/faceGANtts")
from config import ex

@ex.automain
def main(_config):
    input_dir = "/mnt/lustre/work/butz/bst080/data/mvlrs_v1/lrs2_splitted/wav/train"
    output_dir = "./test_preprocessed_wavs"
    os.makedirs(output_dir, exist_ok=True)

    wav_paths = glob(os.path.join(input_dir, "**/*.wav"), recursive=True)
    wav_paths = wav_paths[:2000]  # Nur 10 WAVs zum Testen

    print(f"Test: processed {len(wav_paths)} WAV-data preprocessed...")

    for path in tqdm(wav_paths):
        try:
            waveform, sr = torchaudio.load(path)

            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            n_fft = _config["n_fft"]
            hop_len = _config["hop_len"]
            win_len = _config["win_len"]
            # Schritt 1: Denoising
            waveform_np = waveform.numpy()
            # waveform_denoised = nr.reduce_noise(y=waveform_np, sr=sr, prop_decrease=_config["denoise_factor"])
            # waveform = torch.tensor(waveform_denoised)

            # pad_len = int(0.1 * sr)
            # padded = np.pad(waveform_np, ((0, 0), (0, pad_len)), mode='reflect')
            # denoised = nr.reduce_noise(y=padded, sr=sr, prop_decrease=_config["denoise_factor"])
            # waveform = torch.tensor(denoised[:, :-pad_len])  # Padding entfernen
            waveform_denoised = nr.reduce_noise(
                y=waveform_np,
                sr=sr,
                stationary=True,
                prop_decrease=_config["denoise_factor"],
                # n_std_thresh_stationary=1.5,
                # freq_mask_smooth_hz=400,
                # time_mask_smooth_ms=80,
                n_fft=n_fft,
                win_length=win_len,
                hop_length=hop_len
            )

            waveform = torch.tensor(waveform_denoised)

            # Schritt 1.5: Adaptive Bandstop unter 300 Hz
            if _config["use_bandstop_filter"]:
                with torch.no_grad():
                    window = torch.hann_window(win_len)
                    spec = torch.stft(
                        input=waveform,
                        n_fft=win_len,
                        hop_length=hop_len,
                        win_length=win_len,
                        window=window,
                        return_complex=True
                    )
                    magnitude = spec.abs()[0]  # [Freq, Time]
                    mean_energy = magnitude[:, :].mean(dim=1).numpy()

                    # Nur unter 300 Hz betrachten
                    max_bin = int((300 / (sr / 2)) * (magnitude.size(0)))
                    peak_bin = np.argmax(mean_energy[:max_bin])
                    peak_freq = round((sr / 2) / magnitude.size(0) * peak_bin, 2)

                    # Optional: harmonische auch ignorieren
                    print(f"→ Adaptive Bandstop gesetzt bei {peak_freq} Hz")
                    q_value = _config["bandstop_q_value"]
                    waveform = AF.bandreject_biquad(waveform, sr, central_freq=peak_freq, Q=q_value)


            # Schritt 2: Highpass (optional)
            if _config["use_highpass_filter"]:
                cutoff = _config["highpass_cutoff"]
                waveform = AF.highpass_biquad(waveform, sr, cutoff_freq=cutoff)

            # # Schritt 3: Bandstop (optional)
            # if _config["use_bandstop_filter"]:
            #     center_freq = _config["bandstop_center_freq"]
            #     q_value = _config["bandstop_q_value"]
            #     waveform = AF.bandreject_biquad(waveform, sr, central_freq=center_freq, Q=q_value)

            # Schritt 4: Lowpass (optional)
            if _config["use_lowpass_filter"]:
                waveform = AF.lowpass_biquad(waveform, sr, cutoff_freq=_config["lowpass_cutoff"])

            # Schritt 5: Fadeout 
            fade_len = int(0.05 * sr)  # letzte 50 ms
            fade = torch.linspace(1, 0, fade_len).unsqueeze(0)
            waveform[:, -fade_len:] *= fade

            # Speichern
            rel_path = os.path.relpath(path, input_dir)
            out_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torchaudio.save(out_path, waveform, sr)

        except Exception as e:
            print(f"Error {path}: {e}")

    print(f"Ready! {len(wav_paths)} files saved in: {output_dir}")