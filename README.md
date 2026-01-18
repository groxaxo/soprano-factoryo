<div align="center">
  
  # Soprano-Factory: Train your own 2000x realtime text-to-speech model

  [![Alt Text](https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface)](https://huggingface.co/ekwek/Soprano-80M)
  [![Alt Text](https://img.shields.io/badge/Github-Repo-black?logo=github)](https://github.com/ekwek1/soprano)
  [![Alt Text](https://img.shields.io/badge/HuggingFace-Demo-yellow?logo=huggingface)](https://huggingface.co/spaces/ekwek/Soprano-TTS)

  <img width="640" height="320" alt="soprano-github" src="https://github.com/user-attachments/assets/f929e484-7a1e-43ea-ba75-83bc79b559de" />
</div>

### 📰 News
**2026.01.13 - [Soprano-Factory](https://github.com/ekwek1/soprano-factory) released! You can now train/fine-tune your own Soprano models.**

---

## Overview

**Soprano-Factory** is a 600-LOC training script allowing anyone to train/fine-tune Soprano models with their **own data**, on their **own hardware**. You can use Soprano-Factory to add new voices, styles, and languages to the original Soprano model.

## About Soprano

**[Soprano](https://github.com/ekwek1/soprano)** is an ultra‑lightweight, on-device text‑to‑speech (TTS) model designed for expressive, high‑fidelity speech synthesis at unprecedented speed. Soprano was designed with the following features:
- Up to **20x** real-time generation on CPU and **2000x** real-time on GPU
- **Lossless streaming** with **<250 ms** latency on CPU, **<15 ms** on GPU
- **<1 GB** memory usage with a compact 80M parameter architecture
- **Infinite generation length** with automatic text splitting
- Highly expressive, crystal clear audio generation at **32kHz**
- Widespread support for CUDA, CPU, and MPS devices on Windows, Linux, and Mac
- Supports WebUI, CLI, and OpenAI-compatible endpoint for easy and production-ready inference

---

## Installation

```bash
git clone https://github.com/ekwek1/soprano-factory.git
cd soprano-factory
pip install -r requirements.txt
```

If using Windows you may need to reinstall Pytorch to have CUDA support.

---

## Usage

### 1. Prepare Dataset

Soprano-Factory expects input data in LJSpeech format. Please see `example_dataset` for how to structure your dataset. The wav files can be in any sample rate; they will be automatically resampled to 32 kHz.

### 2. Dataset Preprocessing

```
python generate_dataset.py --input-dir path/to/files

Args:
  --input-dir:  Path to directory of LJSpeech-style dataset. If none is provided this defaults to the provided example dataset.
```

### 3. Model Training

#### Basic Training (LLM only)

```
python train.py --input-dir path/to/files --save-dir path/to/weights

Args:
  --input-dir:  Path to directory of LJSpeech-style dataset. If none is provided this defaults to the provided example dataset.
  --save-dir:   Path to directory to save weights
```

#### Joint Decoder Training (LLM + Decoder)

The decoder is a Vocos-based model that converts LLM hidden states to waveforms. By default, the decoder is frozen during training. To enable joint training of both the LLM and decoder:

```
python train.py --input-dir path/to/files --save-dir path/to/weights --train-decoder

Args:
  --train-decoder:              Enable decoder training with waveform loss
  --decoder-lr-mult:            Learning rate multiplier for decoder (default: 0.1)
  --decoder-loss-weight:        Weight for decoder waveform loss (default: 0.05)
  --freeze-decoder-steps:       Steps before unfreezing decoder (default: 800)
  --decoder-step-freq:          Train decoder every N steps (default: 4)
  --base-model:                 Base model to load decoder from (default: ekwek/Soprano-80M)
```

**Note**: Decoder training uses utterance-level batching (not packed sequences) to maintain waveform alignment. The decoder is trained every N steps (default: 4) with a combined loss of mel-spectrogram and multi-resolution STFT.

### 4. Inference
Once the model has been trained, you can then use the custom weights in the [Soprano repository](https://github.com/ekwek1/soprano)! Just pass your save directory into model_path.

---

## Disclaimer
I did not originally design Soprano with finetuning in mind. As a result, I cannot guarantee that you will see good results after training.

---

## License

This project is licensed under the **Apache-2.0** license. See `LICENSE` for details.
