import os, random, time
import cv2, numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from pathlib import Path

def save_random_frame_as_png(datadir, filename, outdir):
    cap = cv2.VideoCapture(os.path.join(datadir, filename))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {filename}")

    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ridx = random.randint(2, nframes - 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, ridx)

    ret, img = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to read frame")

    # ----------------------------------------------------------
    # auf 224 × 224 hochskalieren
    # (oder größer, aber 224 ist Standard für SyncNet)
    # ----------------------------------------------------------
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)

    os.makedirs(outdir, exist_ok=True)          # Ordner anlegen, falls nötig
    outname = f"{Path(datadir).name}_{Path(filename).stem}_face.png"
    outpath = os.path.join(outdir, outname)
    cv2.imwrite(outpath, img)
    return outpath



def save_random_frame_as_pdf(datadir, filename, pdf_path=None, len_frame=1, max_attempts=5):
    """
    Wählt wie load_random_frame() einen Zufalls-Frame,
    speichert ihn – ohne Farbverfälschung – als 1-seitige PDF.

    Rückgabewert: Pfad der erstellten PDF-Datei
    """
    attempt = 0
    while attempt < max_attempts:
        try:
            cap = cv2.VideoCapture(os.path.join(datadir, filename))
            if not cap.isOpened():
                raise FileNotFoundError(f"Unable to open video file: {filename}")

            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if nframes <= 2:
                raise ValueError(f"Video file {filename} does not have enough frames.")

            # Zufälliger Start-Index wie gehabt
            ridx  = random.randint(2, nframes - len_frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, ridx)

            ret, img = cap.read()
            if not ret:
                raise ValueError(f"Failed to read frame {ridx} from video {filename}")
            cap.release()

            # OpenCV liefert BGR – für Matplotlib in RGB umwandeln
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

            # Ziel-PDF-Pfad
            if pdf_path is None:
                stem      = os.path.splitext(os.path.basename(filename))[0]
                pdf_path  = os.path.join(datadir, f"{stem}_random-frame.pdf")

            # 1-seitige PDF schreiben
            with PdfPages(pdf_path) as pdf:               
                plt.figure(figsize=(8, 8))
                plt.axis("off")
                plt.imshow(img)
                pdf.savefig(bbox_inches="tight", pad_inches=0)
                plt.close()

            return pdf_path

        except (FileNotFoundError, ValueError) as e:
            print(f"Attempt {attempt + 1} failed for {filename}: {e}")
            attempt += 1
            if attempt < max_attempts:
                print("Retrying...")
                time.sleep(10)
            else:
                raise FileNotFoundError(f"Failed to process {filename} after {max_attempts} attempts.")
pdf_file = save_random_frame_as_pdf(
    datadir="/mnt/lustre/work/butz/bst080/data/mvlrs_v1/lrs2_splitted/test/spk1019",
    filename="00014.mp4"
)
print("PDF gespeichert:", pdf_file)