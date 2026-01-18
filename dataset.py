import json
import torch
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, path):
        with open(path, encoding='utf-8') as f:
            self.dataset = json.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Handle both old format [text, audio] and new format {"text": ..., "tokens": ..., "wav_path": ...}
        if isinstance(item, list):
            # Old format: [text, audio_tokens]
            text, audio = item
            wav_path = None
            tokens = audio
        else:
            # New format: dict with text, tokens, wav_path
            text = item["text"]
            tokens = item["tokens"]
            wav_path = item.get("wav_path", None)
        
        # Format: [STOP][TEXT]<text prompt>[START]<audio tokens>[STOP]
        formatted_text = f"[STOP][TEXT]{text}[START]{''.join(list(map(lambda x: f'[{x}]', tokens)))}[STOP]"
        
        # Return dict with both text and wav_path for flexible collation
        return {
            "text": formatted_text,
            "wav_path": wav_path,
            "tokens": tokens
        }
