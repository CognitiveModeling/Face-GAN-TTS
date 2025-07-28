import torch
import torch.nn as nn
import pytorch_lightning as pl
import os
import copy
import cv2
import numpy as np
import sys
from tqdm import tqdm
from config import ex
from model.face_tts import FaceTTS
from data.lrs2_dataset import lrs2Dataset
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils.tts_util import intersperse
from scipy.io.wavfile import write
from model.face_tts_w_discriminator import FaceTTSWithDiscriminator

# Path to the LRS2 test dataset
LRS2_TEST_DIR = "/mnt/lustre/work/butz/bst080/data/mvlrs_v1/lrs2_splitted/test"

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])
    torch.cuda.empty_cache()  # Free GPU memory before starting

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("######## Initializing TTS model")

    # Decide whether to use the GAN-based version
    use_gan = bool(int(os.environ.get("USE_GAN", _config.get("use_gan", 1))))

    # Load checkpoint path from environment or config
    ckpt_env = os.getenv("resume_from_checkpoint")
    print(f"[INFO] resume_from_checkpoint ENV = {ckpt_env}")
    if ckpt_env:
        checkpoint_path = ckpt_env
    else:
        checkpoint_path = _config.get("infr_resume_from_gan") if use_gan else _config.get("infr_resume_from_orig")

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"######## Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Initialize the model
    model = FaceTTSWithDiscriminator(_config).to(device) if use_gan else FaceTTS(_config).to(device)


    # Check whether discriminator weights are present and load only the generator part if needed
    has_discriminator = any(k.startswith("discriminator") for k in checkpoint['state_dict'].keys())
    if has_discriminator:
        print("######## GAN checkpoint detected. Loading only generator weights.")
        generator_state_dict = {
            k: v for k, v in checkpoint["state_dict"].items()
            if not k.startswith("discriminator") and not k.startswith("feature_extractor")
        }
        model.load_state_dict(generator_state_dict, strict=False)

        model.discriminator = None
        model.feature_extractor = None

    else:
        print("######## Standard FaceTTS checkpoint detected.")
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    model.zero_grad()

    for p in model.parameters():
        p.requires_grad = False
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")


    print("######## Initializing HiFi-GAN vocoder")
    vocoder = torch.hub.load('bshall/hifigan:main', 'hifigan').eval().to(device)

    # Load CMU pronunciation dictionary
    cmu = cmudict.CMUDict(_config['cmudict_path'])

    # Set output directory depending on whether GAN is used
    output_dir = _config.get("output_dir_gan") if use_gan else _config.get("output_dir_orig")

    # --- Speaker image loading logic ---
    if _config['use_custom'] == 1:
        # Use user-provided face image
        spk = cv2.imread(_config['test_faceimg'])
        spk = cv2.resize(spk, (224, 224))
        spk = np.transpose(spk, (2, 0, 1))
        spk = torch.FloatTensor(spk).unsqueeze(0).to(device)

    # --- Batch inference over dataset only when use_custom == 2 ---
    elif _config['use_custom'] == 2:
        spk = cv2.imread(_config['test_faceimg'])
        spk = cv2.resize(spk, (224, 224))
        spk = np.transpose(spk, (2, 0, 1))
        spk = torch.FloatTensor(spk).unsqueeze(0).to(device)
        with torch.no_grad():
            speakers = sorted(os.listdir(LRS2_TEST_DIR))[:5]  # nur die ersten 5 Speakerordner
            for speaker in speakers:
                speaker_dir = os.path.join(LRS2_TEST_DIR, speaker)
                for filename in os.listdir(speaker_dir):
                    if filename.endswith(".mp4"):
                        video_path = os.path.join(speaker_dir, filename)
                        text_path = video_path.replace(".mp4", ".txt")

                        if not os.path.exists(text_path):
                            print(f"Warning: Missing transcript for {video_path}")
                            continue

                        # Read transcription
                        with open(text_path, "r", encoding="utf-8") as f:
                            text = f.readline().strip()

                        # Convert text to sequence
                        x = torch.LongTensor(
                            intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))
                        ).to(device)[None]
                        x_len = torch.LongTensor([x.size(-1)]).to(device)

                        # Run inference
                        y_enc, y_dec, attn = model.forward(
                            x,
                            x_len,
                            n_timesteps=_config["timesteps"],
                            temperature=1.5,
                            stoc=False,
                            spk=spk,
                            length_scale=0.91,
                        )

                        # Convert mel-spectrogram to waveform
                        audio = (
                            vocoder.forward(y_dec[-1]).cpu().squeeze().clamp(-1, 1).numpy()
                            * 32768
                        ).astype(np.int16)

                        # Save generated audio
                        output_speaker_dir = os.path.join(output_dir, speaker)
                        os.makedirs(output_speaker_dir, exist_ok=True)
                        output_path = os.path.join(output_speaker_dir, f"{filename.replace('.mp4', '.wav')}")
                        write(output_path, _config["sample_rate"], audio)
    else:
        # use LRS3 image 
        print(f"######## Load speaker from dataset: {_config['dataset']}")
        dataset = lrs2Dataset(split="test", config=_config)
        speaker_folder = os.path.join(LRS2_TEST_DIR, os.listdir(LRS2_TEST_DIR)[0])
        video_files = [f for f in os.listdir(speaker_folder) if f.endswith('.mp4')]
        if not video_files:
            raise FileNotFoundError(f"No .mp4 files found in {speaker_folder}")
        sample_video = video_files[0]
        spk = dataset.load_random_frame(speaker_folder, sample_video)
        if spk is None:
            raise ValueError(f"Failed to load speaker face frame from {sample_path}")
        spk = torch.FloatTensor(spk[0]).unsqueeze(0).to(device)

        print(f"######## Done inference. Check '{output_dir}' folder")

    # --- Inference for test_txt ---
    with open(_config['test_txt'], 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines()]

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, text in enumerate(texts):
            x = torch.LongTensor(
                intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))
            ).to(device)[None]
            x_len = torch.LongTensor([x.size(-1)]).to(device)

            y_enc, y_dec, _ = model.forward(
                x, x_len, n_timesteps=_config["timesteps"], temperature=1.5,
                stoc=False, spk=spk, length_scale=0.91
            )

            audio = (vocoder(y_dec[-1]).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)

            face_tag = os.environ.get("FACE_TAG", "face")
            out_path = os.path.join(output_dir, f"{face_tag}_sample_{i}.wav")
            write(out_path, _config["sample_rate"], audio)
            print(f"Saved  →  {out_path}")
