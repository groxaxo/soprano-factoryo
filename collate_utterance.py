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
    
    # Pad to max len in batch using pad_sequence for efficiency
    from torch.nn.utils.rnn import pad_sequence
    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=tokenizer.pad_token_id)
    
    x = tokens_padded[:, :-1]
    y = tokens_padded[:, 1:]
    
    return x, y, wav_paths
