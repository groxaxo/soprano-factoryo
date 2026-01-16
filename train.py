"""
Training script for Soprano.

Usage:
python train.py --input-dir path/to/files --save-dir path/to/weights

Args:
--input-dir: Path to directory of LJSpeech-style dataset. If none is provided this defaults to the provided example dataset.
--save-dir: Path to directory to save weights

Adapted from https://github.com/karpathy/nanoGPT
"""
import argparse
import pathlib
import random
import time

import numpy as np
import torch
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
    return parser.parse_args()

args = get_args()

# training hyperparameters
device = 'cuda:0'
seed = 1337
max_lr = 2e-5  # reduced from 5e-4 to prevent catastrophic forgetting
warmup_ratio = 0.1
cooldown_ratio = 0.3  # increased for WSD schedule
min_lr = 0.1 * max_lr
batch_size = 4
grad_accum_steps = 1
seq_len = 1024
val_freq = 250
text_factor = 0.1  # trains on both text and audio tokens (prevents modality collapse)
max_steps = 10000
betas = (0.9, 0.95)
weight_decay = 0.01  # reduced from 0.1
train_dataset_path = f'{args.input_dir}/train.json'
val_dataset_path = f'{args.input_dir}/val.json'
save_path = args.save_dir

def worker_seed_init(_):
    worker_seed = torch.initial_seed() % (2**32-1)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_lr(it): # WSD schedule
    if it<warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it<max_steps-cooldown_steps:
        return max_lr
    return min_lr + (max_lr-min_lr) * ((max_steps-it) / cooldown_steps)

def collate_pack(texts):
    tokens_batch = tokenizer(texts, padding=False, truncation=False)
    batch = []
    cur_sample, cur_size = [], 0
    for i in range(len(texts)):
        tokens = torch.tensor(tokens_batch['input_ids'][i][:-1], dtype=torch.long)
        cur_size += tokens.size(0)
        cur_sample.append(tokens)
        if cur_size >= seq_len + 1:
            batch.append(torch.cat(cur_sample)[: seq_len + 1])
            cur_sample, cur_size = [], 0
            if len(batch) == batch_size:
                break
    if cur_sample and not batch: # add partial sample if there isn't enough data
        batch.append(torch.cat(cur_sample + [torch.zeros(seq_len, dtype=torch.long)])[: seq_len + 1])
    if len(batch) < batch_size:
        # pad up to batch_size for consistency
        pad = batch[-1]
        while len(batch) < batch_size:
            batch.append(pad)
    batch = torch.stack(batch)
    x = batch[:, :-1]
    y = batch[:, 1:]
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


tokenizer = AutoTokenizer.from_pretrained('ekwek/Soprano-80M')
if __name__ == '__main__':
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_float32_matmul_precision('high')
    print(f"Save Path: {save_path}")

    # lr schedule
    warmup_steps = int(max_steps * warmup_ratio)
    cooldown_steps = int(max_steps * cooldown_ratio)

    # model
    model = AutoModelForCausalLM.from_pretrained('ekwek/Soprano-80M')
    model.to(torch.bfloat16).to(device)
    model.train()

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

    # optimizer
    opt = torch.optim.AdamW(model.parameters(), max_lr, betas=betas, weight_decay=weight_decay, fused=True)

    pbar = tqdm(range(0, max_steps), ncols=200, dynamic_ncols=True)
    for step in pbar:
        start = time.time()
        if val_freq>0 and (step % val_freq == 0 or step==max_steps-1):
            evaluate(val_dataloader)

        opt.zero_grad()
        audio_loss_accum = 0.0
        text_loss_accum = 0.0
        audio_acc_accum = 0.0
        text_acc_accum = 0.0
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
            total_loss = audio_loss + text_factor*text_loss
            total_loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in opt.param_groups:
            param_group['lr'] = lr
        opt.step()
        torch.cuda.synchronize()
        total_tokens = step * batch_size*seq_len*grad_accum_steps
        end = time.time()
        dt = (end-start)*1000
        tokens_per_second = (batch_size*seq_len*grad_accum_steps) / (end-start)
        tqdm_log = f'txt_L: {text_loss_accum.item():.3f} | aud_L: {audio_loss_accum.item():.3f} | txt_acc: {text_acc_accum.item():.2%} | aud_acc: {audio_acc_accum.item():.2%} | lr: {lr:.2e} | norm: {norm:.3f} | {dt:.0f}ms | {tokens_per_second:.0f}t/s'
        pbar.set_description(tqdm_log)

    print(f"Training complete. Saving model at {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Saving done.")
