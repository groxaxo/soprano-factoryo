"""
Training script for Soprano.

Usage:
python train.py --input-dir path/to/files --save-dir path/to/weights

Args:
--input-dir: Path to directory of LJSpeech-style dataset. If none is provided this defaults to the provided example dataset.
--save-dir: Path to directory to save weights
--train-decoder: Enable decoder training with waveform loss
--decoder-lr-mult: Learning rate multiplier for decoder (default: 0.1)
--decoder-loss-weight: Weight for decoder loss (default: 0.05)
--freeze-decoder-steps: Steps before unfreezing decoder (default: 800)

Adapted from https://github.com/karpathy/nanoGPT
"""
import argparse
import os
import pathlib
import random
import time

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import AudioDataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",
        required=False,
        default="./example_dataset",
        type=pathlib.Path
    )
    parser.add_argument("--save-dir",
        required=True,
        type=pathlib.Path
    )
    # Decoder training flags
    parser.add_argument("--train-decoder",
        action="store_true",
        help="Enable decoder training with waveform loss"
    )
    parser.add_argument("--decoder-lr-mult",
        type=float,
        default=0.1,
        help="Learning rate multiplier for decoder parameters (default: 0.1)"
    )
    parser.add_argument("--decoder-loss-weight",
        type=float,
        default=0.05,
        help="Weight for decoder waveform loss (default: 0.05)"
    )
    parser.add_argument("--freeze-decoder-steps",
        type=int,
        default=800,
        help="Number of steps to keep decoder frozen before training (default: 800)"
    )
    parser.add_argument("--decoder-step-freq",
        type=int,
        default=4,
        help="Train decoder every N steps (default: 4)"
    )
    parser.add_argument("--base-model",
        type=str,
        default="ekwek/Soprano-80M",
        help="Base model to load decoder from (default: ekwek/Soprano-80M)"
    )
    return parser.parse_args()

args = get_args()

# training hyperparameters
device = 'cuda:0'
seed = 1337
base_lr = 1e-5  # base learning rate for most weights
text_lr = 5e-5  # higher LR for text, cross-attn, speaker/adapters
warmup_steps_target = 1500  # warmup steps
min_lr = 2e-6  # minimum learning rate for cosine decay
batch_size = 4
grad_accum_steps = 4  # effective batch size: 4 * 4 = 16
seq_len = 1024
val_freq = 250
max_steps = 15000  # 12k-20k range, using 15k as middle ground
betas = (0.9, 0.95)
weight_decay = 0.01
grad_clip = 1.0
train_dataset_path = f'{args.input_dir}/train.json'
val_dataset_path = f'{args.input_dir}/val.json'
save_path = args.save_dir

def worker_seed_init(_):
    worker_seed = torch.initial_seed() % (2**32-1)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_lr(it):
    """Cosine learning rate schedule with warmup."""
    if it < warmup_steps_target:
        # Linear warmup
        return base_lr * (it + 1) / warmup_steps_target
    # Cosine decay
    progress = (it - warmup_steps_target) / (max_steps - warmup_steps_target)
    return min_lr + (base_lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

def get_text_factor(it):
    """Ramped text_factor schedule to prevent conditioning collapse."""
    if it < 500:
        return 0.00
    elif it < 2000:
        return 0.05
    elif it < 8000:
        return 0.10
    else:
        return 0.15  # cap at 0.15

def collate_pack(batch):
    # Extract text from dict items
    texts = [item["text"] for item in batch]
    tokens_batch = tokenizer(texts, padding=False, truncation=False)
    packed_batch = []
    cur_sample, cur_size = [], 0
    for i in range(len(texts)):
        tokens = torch.tensor(tokens_batch['input_ids'][i][:-1], dtype=torch.long)
        cur_size += tokens.size(0)
        cur_sample.append(tokens)
        if cur_size >= seq_len + 1:
            packed_batch.append(torch.cat(cur_sample)[: seq_len + 1])
            cur_sample, cur_size = [], 0
            if len(packed_batch) == batch_size:
                break
    if cur_sample and not packed_batch: # add partial sample if there isn't enough data
        packed_batch.append(torch.cat(cur_sample + [torch.zeros(seq_len, dtype=torch.long)])[: seq_len + 1])
    if len(packed_batch) < batch_size:
        # pad up to batch_size for consistency
        pad = packed_batch[-1]
        while len(packed_batch) < batch_size:
            packed_batch.append(pad)
    packed_batch = torch.stack(packed_batch)
    x = packed_batch[:, :-1]
    y = packed_batch[:, 1:]
    return x, y

def compute_loss(logits, y, num_steps):
    pred = logits.view(-1, logits.size(-1))
    labels = y.reshape(-1)
    loss = torch.nn.functional.cross_entropy(pred, labels, reduction='none')
    audio_mask = torch.logical_and(y>=3, y<=8003).view(-1)
    audio_loss = loss[audio_mask].mean()
    text_loss = loss[~audio_mask].mean()
    predictions = logits.argmax(dim=-1).view(-1)
    audio_acc = (predictions == labels)[audio_mask].to(torch.float32).mean()
    text_acc = (predictions == labels)[~audio_mask].to(torch.float32).mean()
    audio_loss = audio_loss / num_steps
    text_loss = text_loss / num_steps
    audio_acc = audio_acc / num_steps
    text_acc = text_acc / num_steps
    return audio_loss, text_loss, audio_acc, text_acc

def evaluate(val_dataloader):
    model.eval()
    val_dataloader_it = iter(val_dataloader)
    with torch.no_grad():
        val_audio_loss_accum = torch.tensor(0.0).to(device)
        val_text_loss_accum = torch.tensor(0.0).to(device)
        val_audio_acc_accum = torch.tensor(0.0).to(device)
        val_text_acc_accum = torch.tensor(0.0).to(device)
        val_loss_steps = 1
        for _ in range(val_loss_steps):
            x, y = next(val_dataloader_it)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits = model(x).logits
                audio_loss, text_loss, audio_acc, text_acc = compute_loss(logits, y, val_loss_steps)
            val_audio_loss_accum += audio_loss.detach()
            val_text_loss_accum += text_loss.detach()
            val_audio_acc_accum += audio_acc.detach()
            val_text_acc_accum += text_acc.detach()
        print(f"validation | txt_L: {val_text_loss_accum.item():.4f} | aud_L: {val_audio_loss_accum.item():.4f} | txt_acc: {val_text_acc_accum.item():.2%} | aud_acc: {val_audio_acc_accum.item():.2%}")
    model.train()


if __name__ == '__main__':
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_float32_matmul_precision('high')
    print(f"Save Path: {save_path}")
    
    # Decoder training setup
    if args.train_decoder:
        print("Decoder training enabled")
        print(f"  Decoder LR multiplier: {args.decoder_lr_mult}")
        print(f"  Decoder loss weight: {args.decoder_loss_weight}")
        print(f"  Freeze decoder steps: {args.freeze_decoder_steps}")
        print(f"  Decoder step frequency: every {args.decoder_step_freq} steps")
        
        from decoder import SopranoDecoder
        from loss_audio import audio_decoder_loss
        from collate_utterance import collate_utterance
        from functools import partial
        
        # Constants for audio token range
        AUDIO_MIN = 3
        AUDIO_MAX = 8003

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # model
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    model.to(torch.bfloat16).to(device)
    model.train()
    
    # Get LLM hidden dimension for decoder
    llm_hidden_dim = model.config.hidden_size
    
    # Load or initialize decoder
    decoder = None
    if args.train_decoder:
        try:
            # Try to load existing decoder
            decoder = SopranoDecoder.from_pretrained(args.base_model, device=device)
            print(f"Loaded decoder from {args.base_model}")
        except FileNotFoundError:
            # Initialize new decoder
            print(f"Initializing new decoder (hidden_dim={llm_hidden_dim})")
            decoder = SopranoDecoder(
                input_dim=llm_hidden_dim,
                hidden_dim=512,
                num_layers=4,
            )
        decoder.to(torch.bfloat16).to(device)
        decoder.train()

    # dataset
    dataset = AudioDataset(train_dataset_path)
    # we need batch_size * 16 to have enough tokens after packing
    dataloader = DataLoader(dataset,
        batch_size=batch_size * 16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_seed_init,
        collate_fn=collate_pack,
    )
    dataloader_it = iter(dataloader)
    
    # Decoder dataloader (for utterance-level batching)
    decoder_dataloader = None
    decoder_dataloader_it = None
    if args.train_decoder:
        decoder_collate_fn = partial(collate_utterance, tokenizer=tokenizer, seq_len=seq_len)
        decoder_dataloader = DataLoader(dataset,
            batch_size=batch_size,  # smaller batch for decoder
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=worker_seed_init,
            collate_fn=decoder_collate_fn,
        )
        decoder_dataloader_it = iter(decoder_dataloader)
    
    val_dataset = AudioDataset(val_dataset_path)
    val_dataloader = DataLoader(val_dataset,
        batch_size=batch_size * 16,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_seed_init,
        collate_fn=collate_pack,
    )

    # optimizer with parameter groups
    # Higher LR for text embeddings and conditioning pathway
    text_params = []
    base_params = []
    lr_ratio = text_lr / base_lr  # 5x ratio for text pathway
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Text token embeddings (wte), cross-attention, speaker/adapters get higher LR
        # Use specific patterns to avoid false positives
        if any(keyword in name for keyword in ['wte', 'cross_attn', 'speaker', 'adapter']) or \
           name.endswith('token_embeddings') or 'token_embed' in name:
            text_params.append(param)
        else:
            base_params.append(param)
    
    param_groups = [
        {'params': base_params, 'lr': base_lr, 'group_name': 'base'},
        {'params': text_params, 'lr': text_lr, 'group_name': 'text'},
    ]
    
    # Add decoder parameters if decoder training enabled
    if args.train_decoder:
        decoder_params = list(decoder.parameters())
        param_groups.append({
            'params': decoder_params,
            'lr': base_lr * args.decoder_lr_mult,
            'weight_decay': 0.0,
            'group_name': 'decoder'
        })
    
    opt = torch.optim.AdamW(param_groups, betas=betas, weight_decay=weight_decay, fused=True)

    pbar = tqdm(range(0, max_steps), ncols=200, dynamic_ncols=True)
    for step in pbar:
        start = time.time()
        if val_freq>0 and (step % val_freq == 0 or step==max_steps-1):
            evaluate(val_dataloader)
        
        # Decoder freeze/unfreeze schedule
        if args.train_decoder:
            if step < args.freeze_decoder_steps:
                decoder.eval()
                for p in decoder.parameters():
                    p.requires_grad = False
            else:
                decoder.train()
                for p in decoder.parameters():
                    p.requires_grad = True

        # Decide if this is a decoder training step
        is_decoder_step = args.train_decoder and step >= args.freeze_decoder_steps and (step % args.decoder_step_freq == 0)

        opt.zero_grad()
        audio_loss_accum = 0.0
        text_loss_accum = 0.0
        audio_acc_accum = 0.0
        text_acc_accum = 0.0
        decoder_loss_accum = 0.0
        
        if is_decoder_step:
            # Decoder training step with utterance batches
            try:
                x, y, wav_paths = next(decoder_dataloader_it)
            except:
                decoder_dataloader_it = iter(decoder_dataloader)
                x, y, wav_paths = next(decoder_dataloader_it)
            x, y = x.to(device), y.to(device)
            
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                # Get hidden states from LLM
                out = model(x, output_hidden_states=True)
                logits = out.logits
                hidden_states = out.hidden_states[-1]  # [B, T, D]
                
                # Compute LM loss
                audio_loss, text_loss, audio_acc, text_acc = compute_loss(logits, y, 1)
                audio_loss_accum += audio_loss.detach()
                text_loss_accum += text_loss.detach()
                audio_acc_accum += audio_acc.detach()
                text_acc_accum += text_acc.detach()
                
                text_factor = get_text_factor(step)
                lm_loss = audio_loss + text_factor * text_loss
                
                # Extract hidden states for audio tokens
                # y contains the target tokens, identify audio tokens
                is_audio = (y >= AUDIO_MIN) & (y <= AUDIO_MAX)  # [B, T]
                
                # Align hidden states with y (shift by 1)
                h_for_y = hidden_states[:, 1:, :]  # Shift to align with y positions
                
                # Process each sample in batch
                total_decoder_loss = 0.0
                num_decoder_samples = 0
                
                for b in range(y.size(0)):
                    if wav_paths[b] is None:
                        continue
                        
                    # Get audio token positions for this sample
                    mask_b = is_audio[b]  # [T]
                    if mask_b.sum() == 0:
                        continue
                    
                    # Extract hidden states for audio tokens
                    h_audio = h_for_y[b, mask_b, :]  # [T_audio, D]
                    
                    # Load ground truth waveform
                    try:
                        gt_wav, sr = torchaudio.load(wav_paths[b])
                        if sr != 32000:
                            gt_wav = torchaudio.functional.resample(gt_wav, sr, 32000)
                        gt_wav = gt_wav.to(device)
                        
                        # Generate predicted waveform from decoder
                        pred_wav = decoder(h_audio.unsqueeze(0))  # [1, 1, L]
                        
                        # Match lengths by truncating or padding
                        pred_len = pred_wav.size(-1)
                        gt_len = gt_wav.size(-1)
                        min_len = min(pred_len, gt_len)
                        
                        pred_wav = pred_wav[..., :min_len]
                        gt_wav = gt_wav[..., :min_len].unsqueeze(0)  # [1, 1, L]
                        
                        # Compute decoder loss
                        dec_loss, _ = audio_decoder_loss(pred_wav, gt_wav)
                        total_decoder_loss += dec_loss
                        num_decoder_samples += 1
                        
                    except Exception as e:
                        print(f"\nWarning: Failed to load/process {wav_paths[b]}: {e}")
                        continue
                
                # Average decoder loss
                if num_decoder_samples > 0:
                    avg_decoder_loss = total_decoder_loss / num_decoder_samples
                    decoder_loss_accum = avg_decoder_loss.detach()
                    
                    # Combined loss
                    total_loss = lm_loss + args.decoder_loss_weight * avg_decoder_loss
                else:
                    total_loss = lm_loss
                
                total_loss.backward()
        else:
            # Regular LM training step with packed batches
            for micro_step in range(grad_accum_steps):
                try:
                    x, y = next(dataloader_it)
                except:
                    dataloader_it = iter(dataloader)
                    x, y = next(dataloader_it)
                x, y = x.to(device), y.to(device)

                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits = model(x).logits
                    audio_loss, text_loss, audio_acc, text_acc = compute_loss(logits, y, grad_accum_steps)
                audio_loss_accum += audio_loss.detach()
                text_loss_accum += text_loss.detach()
                audio_acc_accum += audio_acc.detach()
                text_acc_accum += text_acc.detach()
                text_factor = get_text_factor(step)
                total_loss = audio_loss + text_factor * text_loss
                total_loss.backward()

        # Gradient clipping and optimizer step
        if args.train_decoder:
            # Clip gradients for both model and decoder
            norm_model = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            norm_decoder = torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
            norm = max(norm_model, norm_decoder)
        else:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        base_lr_current = get_lr(step)
        # Apply different LRs to parameter groups
        opt.param_groups[0]['lr'] = base_lr_current
        opt.param_groups[1]['lr'] = base_lr_current * lr_ratio
        if args.train_decoder:
            opt.param_groups[2]['lr'] = base_lr_current * args.decoder_lr_mult
        
        opt.step()
        torch.cuda.synchronize()
        total_tokens = step * batch_size*seq_len*grad_accum_steps
        end = time.time()
        dt = (end-start)*1000
        tokens_per_second = (batch_size*seq_len*grad_accum_steps) / (end-start)
        
        # Update progress bar
        if args.train_decoder and is_decoder_step:
            tqdm_log = f'txt_L: {text_loss_accum.item():.3f} | aud_L: {audio_loss_accum.item():.3f} | dec_L: {decoder_loss_accum.item():.3f} | txt_acc: {text_acc_accum.item():.2%} | aud_acc: {audio_acc_accum.item():.2%} | txt_f: {text_factor:.2f} | lr: {base_lr_current:.2e} | norm: {norm:.3f} | {dt:.0f}ms | [DEC]'
        else:
            tqdm_log = f'txt_L: {text_loss_accum.item():.3f} | aud_L: {audio_loss_accum.item():.3f} | txt_acc: {text_acc_accum.item():.2%} | aud_acc: {audio_acc_accum.item():.2%} | txt_f: {text_factor:.2f} | lr: {base_lr_current:.2e} | norm: {norm:.3f} | {dt:.0f}ms | {tokens_per_second:.0f}t/s'
        pbar.set_description(tqdm_log)

    print(f"Training complete. Saving model at {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Save decoder if training was enabled
    if args.train_decoder:
        os.makedirs(save_path, exist_ok=True)
        decoder_path = os.path.join(save_path, "decoder.pth")
        torch.save(decoder.state_dict(), decoder_path)
        print(f"Saved decoder to {decoder_path}")
    
    print("Saving done.")
