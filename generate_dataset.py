"""
Converts a dataset in LJSpeech format into audio tokens that can be used to train/fine-tune Soprano.
This script creates two JSON files for train and test splits in the provided directory.

Usage:
python generate_dataset.py --input-dir path/to/files

Args:
--input-dir: Path to directory of LJSpeech-style dataset. If none is provided this defaults to the provided example dataset.
"""
import argparse
import pathlib
import random
import json

from scipy.io import wavfile
import torchaudio
import torch
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from encoder.codec import Encoder
from text_normalizer import normalize_text


SAMPLE_RATE = 32000
SEED = 42
VAL_PROP = 0.1
VAL_MAX = 512

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",
        required=False,
        default="./example_dataset",
        type=pathlib.Path
    )
    return parser.parse_args()

def main():
    args = get_args()
    input_dir = args.input_dir

    print("Loading model.")
    encoder = Encoder()
    encoder_path = hf_hub_download(repo_id='ekwek/Soprano-Encoder', filename='encoder.pth')
    encoder.load_state_dict(torch.load(encoder_path))
    print("Model loaded.")


    print("Reading metadata.")
    files = []
    with open(f'{input_dir}/metadata.txt', encoding='utf-8') as f:
        data = f.read().split('\n')
        for line in data:
            if not line.strip():
                continue  # Skip empty lines
            filename, transcript = line.split('|', maxsplit=1)
            # Apply text normalization
            transcript = normalize_text(transcript)
            files.append((filename, transcript))
    print(f'{len(files)} samples located in directory.')

    print("Encoding audio.")
    dataset = []
    for sample in tqdm(files):
        filename, transcript = sample
        # Use absolute path to ensure wav can be loaded from any working directory
        wav_path = f'{input_dir}/wavs/{filename}.wav'
        sr, audio = wavfile.read(wav_path)
        audio = torch.from_numpy(audio)
        if sr != SAMPLE_RATE:
            audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
        audio = audio.unsqueeze(0)
        with torch.no_grad():
            audio_tokens = encoder(audio)
        # Now store as dict with wav_path for decoder training
        dataset.append({
            "text": transcript,
            "tokens": audio_tokens.squeeze(0).tolist(),
            "wav_path": wav_path
        })

    print("Generating train/test splits.")
    random.seed(SEED)
    random.shuffle(dataset)
    num_val = min(int(VAL_PROP * len(dataset)) + 1, VAL_MAX)
    train_dataset = dataset[num_val:]
    val_dataset = dataset[:num_val]
    print(f'# train samples: {len(train_dataset)}')
    print(f'# val samples: {len(val_dataset)}')

    print("Saving datasets.")
    with open(f'{input_dir}/train.json', 'w', encoding='utf-8') as f:
        json.dump(train_dataset, f, indent=2)
    with open(f'{input_dir}/val.json', 'w', encoding='utf-8') as f:
        json.dump(val_dataset, f, indent=2)
    print("Datasets saved.")


if __name__ == '__main__':
    main()
