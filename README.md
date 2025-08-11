# [Face-GAN-TTS] An Adversarial-Diffusion Framework for Generating High-Quality Voices from Faces

<a href="https://degiaaaa.github.io/Demo_Face-GAN-TTS/"><img src="https://img.shields.io/badge/Project%20Page-online-brightgreen"></a>

---
## Installation

1. Creating Package Environments
```
conda env create lrs2_preprocessing/environment_label.yml
conda activate label_env

```

```
conda env create -f environment_train_emv.yaml
conda activate train_env
```

2. Build monotonic align module
```
cd model/monotonic_align; python setup.py build_ext --inplace; cd ../..
```

---
## Preparation

1. Checkpoint handling

    1.1 For Face-GAN-TTS transfer LRS2 trained model weights from the USB-Stick

    1.2 For FACE-TTS download LRS3 trained model weights from [here](https://drive.google.com/file/d/18ERr-91Z1Mnc2Aq9n1nBPijzb5gSymLq/view?usp=sharing)

    1.3 Store the checkpoints here: `'.\ckpts\'`

    1.4 Adjust `'resume_from'`, `'use_gan'`, `'infr_resume_from_orig'` or `'infr_resume_gan'` in `config.py`


2. Download <a href="https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html">LRS2</a> into `'data/lrs2/'` 


3. Extract and save audio as '*.wav' files in `'data/lrs2/wav'`
  ```
   conda activate label_env 
   ```
   ```
   python data/lrs2_preprocessing/lrs2_split/extract_audio.py
   ```

4. Data Labeling and Preprocessing
   ```
   conda activate label_env 
   ```
   ```
   python data/lrs2_preprocessing/labeling.py
   ```

:exclamation: Faces in video files should be cropped and aligned for LRS2 distribution. You can use <a href="https://github.com/joonson/syncnet_python/tree/master/detectors">'syncnet_python/detectors'</a>. 

---
## Test

1. Prepare text description in txt file.
```
echo "This is test" > test/text.txt
```


2. Inference Face-TTS.
```
python inference.py
```

3. Result will be saved in `'test/'`. 

:zap: To make MOS test set, we use the LRS2 test set and the <a href="https://www.chicagofaces.org/">CFD corpus</a> to randomly select faces.

--- 
## Training

1. Check config.py 

2. Run
```
python train.py
```

---
## Reference
This repo is based on 
<a href="https://github.com/naver-ai/facetts">FACE-TTS</a>, 
<a href="https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS">Grad-TTS</a>, 
<a href="https://github.com/bshall/hifigan">HiFi-GAN-16k</a>, 
<a href="https://github.com/joonson/syncnet_trainer">SyncNet</a>.  Thanks!


---
## License

```
Face-GAN-TTS
Copyright (c) 2025-present Cognitive Modeling Group University of Tübingen

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```




