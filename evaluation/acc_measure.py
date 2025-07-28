import os

from config import ex
import torch
import torchaudio as ta
import pytorch_lightning as pl
import torchaudio.functional as F   
import os, torchaudio as ta
from model.syncnet_hifigan import SyncNet
from utils.mel_spectrogram import mel_spectrogram

from tqdm import tqdm
import cv2
import numpy as np
import random

N = 5
FOLDER = "./test/results_facegantts/"

@ex.automain
def main(_config):

    pl.seed_everything(_config["seed"])
    files = [file for file in os.listdir(FOLDER) if file.endswith(".png")]

    speakers = {}
    for f in files:
        spk = f.split("_")[0]
        if spk not in speakers.keys():
            speakers[spk] = []
        speakers[spk].append(f)

    spk_list = list(speakers.keys())
    model = SyncNet(_config).eval().cuda()

    accs = []
    with torch.no_grad():
        for _ in tqdm(range(100)):
            idxs = random.sample(range(len(speakers)), N)
            v, a = [], []

            for i in idxs:
                spk = spk_list[i]
                f   = random.choice(speakers[spk])

                # ---------- Bild ----------
                img = cv2.imread(os.path.join(FOLDER, f))
                img = cv2.resize(img, (224, 224))                
                img = torch.FloatTensor(img.transpose(2,0,1))    # CHW

                # ---------- Audio ----------
                wav  = f.replace("_face.png", ".wav")
                aud, _ = ta.load(os.path.join(FOLDER, wav))
                # How does facetts if we filter the noise afterwards? 
                aud = F.highpass_biquad(aud, 16_000, 300)      #  << 300-Hz-HP
                aud = F.lowpass_biquad (aud, 16_000, 4000)     #  << 4-kHz-LP
                # --------------------------------------------------------------------------

                aud = mel_spectrogram(
                        aud, 1024, 128, 16000, 160, 1024,
                        0.0, 8000.0, center=False).squeeze()     # (128,T)

                # ---------- Embeddings ----------
                zv, za = model(img[None].cuda(), aud[None,None].cuda())
                zv = zv.squeeze(0).squeeze(-1)   # (512,)
                za = za.mean(dim=2).squeeze(0)   # (512,)

                v.append(zv)
                a.append(za)

            # -------- nach INNERER Schleife --------
            v = torch.stack(v, 0).cuda()          # (N,512)
            a = torch.stack(a, 0).cuda()          # (N,512)

            sim    = torch.matmul(v, a.t())       # (N,N)
            labels = torch.arange(N, device=sim.device)
            acc    = (sim.argmax(dim=1) == labels).float().mean()

            accs.append(acc.item())

    print(f"######## Done measurement.")
    print(f"######## Dir: ", FOLDER)
    acc = np.mean(accs)
    print(f"######## {N}-way ACC: ", acc)

    #Sanity Check for noise recognition instead of speaker
    OUT_DIR  = os.path.join(FOLDER, "bandpass_300-4000")
    os.makedirs(OUT_DIR, exist_ok=True)

    for fname in os.listdir(FOLDER):
        if fname.endswith(".wav"):
            wav, sr = ta.load(os.path.join(FOLDER, fname))
            if sr != 16_000:
                wav = ta.functional.resample(wav, sr, 16_000)
                sr  = 16_000
            wav = F.highpass_biquad(wav, sr, 300)
            wav = F.lowpass_biquad (wav, sr, 4000)
            ta.save(os.path.join(OUT_DIR, fname.replace(".wav", "_bp.wav")), wav, sr)