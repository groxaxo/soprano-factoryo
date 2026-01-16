"""
Safer model patcher for Soprano.

This patcher fixes tokenizer issues without corrupting token IDs.
Unlike add_tokens() which appends to the end, this version:
- Wires eos_token to [STOP] if it already exists in vocab
- Adds a distinct [PAD] only if missing, and resizes embeddings

Usage:
python patch_model.py --base path/to/base/model --output path/to/output/model
"""
import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModelForCausalLM


def patch_model(base_path, output_path):
    print(f"Loading base from {base_path}...")
    tok = AutoTokenizer.from_pretrained(base_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(base_path)

    print(f"Vocab size (before): {len(tok)}")

    # 1) EOS: prefer existing [STOP] without assuming its ID
    stop_id = tok.convert_tokens_to_ids("[STOP]")
    if stop_id != tok.unk_token_id:  # [STOP] exists in vocab
        if tok.eos_token != "[STOP]" or tok.eos_token_id != stop_id:
            print(f"Setting EOS to [STOP] (id={stop_id})")
            tok.eos_token = "[STOP]"
            tok.eos_token_id = stop_id
    else:
        print("WARNING: [STOP] not found in vocab. Not changing EOS. "
              "If your pipeline requires [STOP], you must use the correct tokenizer files (don't add_tokens).")

    # 2) PAD: ensure pad exists and does NOT collide with EOS
    if tok.pad_token is None or tok.pad_token_id is None:
        print("Adding [PAD] special token...")
        tok.add_special_tokens({"pad_token": "[PAD]"})
    elif tok.pad_token_id == tok.eos_token_id:
        print("WARNING: pad_token_id == eos_token_id. This can break stopping.")
        # Prefer adding a distinct PAD if possible
        if tok.convert_tokens_to_ids("[PAD]") == tok.unk_token_id:
            tok.add_special_tokens({"pad_token": "[PAD]"})
        tok.pad_token = "[PAD]"
        tok.pad_token_id = tok.convert_tokens_to_ids("[PAD]")

    print(f"EOS: {tok.eos_token} id={tok.eos_token_id}")
    print(f"PAD: {tok.pad_token} id={tok.pad_token_id}")

    # 3) Resize embeddings if tokenizer grew
    model.resize_token_embeddings(len(tok))

    # Stamp normalizer version if available
    try:
        from text_normalizer import NORMALIZER_VERSION
        model.config.soprano_text_normalizer_version = NORMALIZER_VERSION
        print(f"Stamped normalizer version: {NORMALIZER_VERSION}")
    except Exception as e:
        print(f"NOTE: normalizer version not stamped: {e}")

    print(f"Saving to {output_path}...")
    model.save_pretrained(output_path)
    tok.save_pretrained(output_path)

    # Copy/download decoder.pth
    base_decoder = os.path.join(base_path, "decoder.pth")
    out_decoder = os.path.join(output_path, "decoder.pth")
    if os.path.exists(base_decoder):
        import shutil
        shutil.copy(base_decoder, out_decoder)
        print("Copied decoder.pth")
    else:
        try:
            from huggingface_hub import hf_hub_download
            print("Downloading decoder.pth...")
            hf_hub_download(repo_id=base_path, filename="decoder.pth", local_dir=output_path)
        except Exception as e:
            print(f"Could not download decoder.pth: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="./bases/Soprano-1.1-80M")
    ap.add_argument("--output", default="./bases/Soprano-1.1-80M_tokfix_safe")
    args = ap.parse_args()
    patch_model(args.base, args.output)
