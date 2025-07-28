
import os
import shutil
from pathlib import Path
from save_face_pdf import save_random_frame_as_png  

AUDIO_LIST = [
    "/mnt/lustre/work/butz/bst080/faceGANtts/test/inference_outputs/FACETTS_finetuned_lr1e-8_gamma0.02_denoised/spk2565/00030.wav",
    "/mnt/lustre/work/butz/bst080/faceGANtts/test/inference_outputs/FACETTS_finetuned_lr1e-8_gamma0.02_denoised/spk5934/00020.wav",
    "/mnt/lustre/work/butz/bst080/faceGANtts/test/inference_outputs/FACETTS_finetuned_lr1e-8_gamma0.02_denoised/spk9201/00007.wav",
    "/mnt/lustre/work/butz/bst080/faceGANtts/test/inference_outputs/FACETTS_finetuned_lr1e-8_gamma0.02_denoised/spk2077/00054.wav",
    "/mnt/lustre/work/butz/bst080/faceGANtts/test/inference_outputs/FACETTS_finetuned_lr1e-8_gamma0.02_denoised/spk4763/00015.wav"
]

# AUDIO_LIST = [
#     "/mnt/lustre/work/butz/bst080/faceGANtts/test/inference_outputs/v1538404_hinge/spk2565/00030.wav",
#     "/mnt/lustre/work/butz/bst080/faceGANtts/test/inference_outputs/v1538404_hinge/spk5934/00020.wav",
#     "/mnt/lustre/work/butz/bst080/faceGANtts/test/inference_outputs/v1538404_hinge/spk9201/00007.wav",
#     "/mnt/lustre/work/butz/bst080/faceGANtts/test/inference_outputs/v1538404_hinge/spk2077/00054.wav",
#     "/mnt/lustre/work/butz/bst080/faceGANtts/test/inference_outputs/v1538404_hinge/spk4763/00015.wav"
# ]

VIDEO_ROOT = "/mnt/lustre/work/butz/bst080/data/mvlrs_v1/lrs2_splitted/test"
TARGET_DIR = "./test/results_facetts" #"./test/results_facetts"
os.makedirs(TARGET_DIR, exist_ok=True)

for wav_path in AUDIO_LIST:
    speaker = Path(wav_path).parts[-2]
    file_id = Path(wav_path).stem
    video_file = f"{file_id}.mp4"
    video_path = os.path.join(VIDEO_ROOT, speaker, video_file)

    # Extract frame as PNG
    out_name = f"{speaker}_{file_id}_face.png"
    try:
        out_path = save_random_frame_as_png(
            datadir=os.path.join(VIDEO_ROOT, speaker),
            filename=video_file,
            outdir=TARGET_DIR
        )
        print("Saved:", out_path)
    except Exception as e:
        print("ERROR extracting frame:", video_path, "->", e)
        continue

    # Copy audio to same folder
    new_audio_path = os.path.join(TARGET_DIR, f"{speaker}_{file_id}.wav")
    shutil.copy(wav_path, new_audio_path)
    print("Copied:", new_audio_path)