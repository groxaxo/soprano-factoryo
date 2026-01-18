"""
Utterance-level collator for decoder training.
Keeps one utterance per sequence to maintain waveform alignment.
"""
import torch


def collate_utterance(batch, tokenizer, seq_len):
    """
    Collate function that keeps utterances separate (no packing).
    
    Args:
        batch: list of dicts with keys:
          - "text": str (formatted text)
          - "wav_path": str
          - "tokens": list of audio token IDs
        tokenizer: HuggingFace tokenizer
        seq_len: maximum sequence length
    
    Returns:
        x: input token IDs [B, T]
        y: target token IDs [B, T]
        wav_paths: list of wav file paths [B]
    """
    texts = [item["text"] for item in batch]
    wav_paths = [item["wav_path"] for item in batch]
    
    # Tokenize each text
    tokens_batch = tokenizer(texts, padding=False, truncation=False)
    
    # Truncate to seq_len + 1 and collect
    tokens = []
    for i in range(len(texts)):
        t = torch.tensor(tokens_batch['input_ids'][i], dtype=torch.long)
        t = t[: seq_len + 1]
        tokens.append(t)
    
    # Pad to max len in batch
    max_t = max(t.numel() for t in tokens)
    pad_id = tokenizer.pad_token_id
    tokens_padded = []
    for t in tokens:
        if t.numel() < max_t:
            t = torch.cat([t, torch.full((max_t - t.numel(),), pad_id, dtype=torch.long)])
        tokens_padded.append(t)
    
    tokens_padded = torch.stack(tokens_padded)  # [B, T]
    x = tokens_padded[:, :-1]
    y = tokens_padded[:, 1:]
    
    return x, y, wav_paths
