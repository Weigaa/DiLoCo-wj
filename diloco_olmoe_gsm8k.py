"""
Stream-DiLoCo-style Distributed Data Parallel training for GSM8K with OLMoE-1B-7B
Single-node, 4-GPU DDP. Local optimizer: AdamW every step. Cross-node Nesterov sync every H steps.
After training, evaluate on GSM8K test set by generation and compute accuracy.

Usage (example):
    torchrun --nproc_per_node=4 diloco_olmoe_gsm8k.py \
        --model_name allenai/OLMoE-1B-7B \
        --per_device_batch 2 \
        --grad_accum 2 \
        --epochs 1 \
        --lr 5e-5 \
        --weight_decay 0.01 \
        --H 20 \
        --mu 0.9 \
        --lr_nes 1.0 \
        --max_train_samples 2000 \
        --max_eval_samples 500 \
        --max_new_tokens 256

Notes:
- The exact Hugging Face model ID for "olmoe-1b-7b" may differ. Replace --model_name accordingly.
- This script aims for clarity over peak performance. Consider fusing ops, using Flash-Attn, and Deepspeed for larger runs.
- Mixed precision (bf16 if available, else fp16) is used.

#not avaliable now
"""
from __future__ import annotations
import argparse
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset

# ------------------------------ Utils ------------------------------
def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0

def setup_distributed():
    if dist.is_available() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)


def cleanup_distributed():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


@dataclass
class Example:
    q: str
    a: str

class GSM8KTrainDataset(Dataset):
    def __init__(self, hf_split, tokenizer, max_length: int = 2048):
        self.data = [
            Example(x["question"].strip(), x["answer"].strip()) for x in hf_split
        ]
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        # Simple supervised formatting: include rationale; train to emit final answer
        prompt = (
            "You are a helpful math tutor.\n"
            f"Question: {ex.q}\n"
            "Answer step by step and finish with '#### <final_number>'.\n"
            "Answer:"
        )
        target = ex.a  # contains rationale and final '####'
        text = prompt + "\n" + target
        # We train on the whole concatenated text; mask the prompt tokens from loss if desired.
        inputs = self.tok(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        # Create labels identical to input_ids
        labels = inputs.input_ids.clone()
        # Optionally, ignore loss on the prompt portion
        with self.tok.as_target_tokenizer():
            prompt_tok = self.tok(
                prompt,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                return_tensors="pt",
            )
        ignore_len = prompt_tok.input_ids.size(1)
        labels[0, :ignore_len] = -100
        return {
            "input_ids": inputs.input_ids[0],
            "attention_mask": inputs.attention_mask[0],
            "labels": labels[0],
        }

class GSM8KEvalDataset(Dataset):
    def __init__(self, hf_split, tokenizer, max_length: int = 2048):
        self.rows = [
            (x["question"].strip(), x["answer"].strip()) for x in hf_split
        ]
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        q, a = self.rows[idx]
        prompt = (
            "You are a helpful math tutor.\n"
            f"Question: {q}\n"
            "Answer step by step and finish with '#### <final_number>'.\n"
            "Answer:"
        )
        inputs = self.tok(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs.input_ids[0],
            "attention_mask": inputs.attention_mask[0],
            "gold_answer": a,
            "prompt": prompt,
        }

# ------------------------------ Nesterov DiLoCo core ------------------------------
class NesterovGlobalState:
    def __init__(self, model: torch.nn.Module):
        # Shadow params and velocity buffers mirror model parameters
        self.shadow: Dict[str, torch.Tensor] = {}
        self.v: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[name] = p.detach().clone()
            self.v[name] = torch.zeros_like(p, device=p.device)

    @torch.no_grad()
    def sync_step(self, model: torch.nn.Module, mu: float, lr_nes: float, world_size: int):
        """
        Stream-DiLoCo style: every H steps, compute average of current local parameters across ranks (p_avg),
        update a global shadow with Nesterov momentum, then broadcast the new shadow to all ranks.
        v = mu * v + (p_avg - shadow)
        shadow = shadow - lr_nes * (mu * v + (p_avg - shadow))    # Nesterov lookahead
        model.params <- shadow (overwrite)
        """
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # all-reduce to get sum, then average
            dist.all_reduce(p.data, op=dist.ReduceOp.SUM)
            p_avg = p.data / world_size
            # velocity and shadow update
            v = self.v[name]
            delta = p_avg - self.shadow[name]
            v.mul_(mu).add_(delta)
            # Nesterov update (lookahead)
            self.shadow[name].add_(-(lr_nes) * (mu * v + delta))
            # Overwrite model weights with updated shadow
            p.data.copy_(self.shadow[name])

        # Make sure all ranks have identical weights
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            dist.broadcast(p.data, src=0)

# ------------------------------ Accuracy helpers ------------------------------
NUM_RE = re.compile(r"####\s*([-+]?\d+(?:\.\d+)?)")
FALLBACK_NUM_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)")

def extract_final_number(text: str) -> Optional[str]:
    m = NUM_RE.search(text)
    if m:
        return m.group(1)
    # fallback: take last number in string
    ms = list(FALLBACK_NUM_RE.finditer(text))
    if ms:
        return ms[-1].group(1)
    return None

# ------------------------------ Main ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="allenai/OLMoE-1B-7B-0924")
    parser.add_argument("--per_device_batch", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--H", type=int, default=20, help="Sync period (steps)")
    parser.add_argument("--mu", type=float, default=0.9, help="Nesterov momentum")
    parser.add_argument("--lr_nes", type=float, default=1.0, help="Nesterov global step size")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    setup_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(device) if device.type == "cuda" else None

    if is_main_process():
        print(f"World size: {world_size}. Using model: {args.model_name}")

    # -------------------- Data --------------------
    if is_main_process():
        print("Loading GSM8K...")
    ds = load_dataset("gsm8k", "main")
    if args.max_train_samples > 0:
        ds["train"] = ds["train"].select(range(min(args.max_train_samples, len(ds["train"]))))
    if args.max_eval_samples > 0:
        ds["test"] = ds["test"].select(range(min(args.max_eval_samples, len(ds["test"]))))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # better for generation

    train_ds = GSM8KTrainDataset(ds["train"], tokenizer, max_length=args.max_length)
    eval_ds = GSM8KEvalDataset(ds["test"], tokenizer, max_length=args.max_length)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    train_loader = DataLoader(train_ds, batch_size=args.per_device_batch, sampler=train_sampler, num_workers=2, pin_memory=True, collate_fn=default_collate)

    # eval: run only on rank 0 later

    # -------------------- Model --------------------
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
    ).to(device)

    # Enable gradient checkpointing for large models if supported
    model.gradient_checkpointing_enable()

    ddp_model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None, output_device=device.index if device.type == "cuda" else None, find_unused_parameters=False)

    # Optimizer & scheduler
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    grouped_params = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped_params, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    total_train_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    warmup_steps = int(total_train_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_train_steps)

    # Nesterov global state
    global_state = NesterovGlobalState(model)

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype==torch.float16))

    # -------------------- Train --------------------
    if is_main_process():
        print(f"Start training for {args.epochs} epochs, H={args.H}, mu={args.mu}, lr_nes={args.lr_nes}")

    step_since_sync = 0
    global_step = 0
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        ddp_model.train()
        for it, batch in enumerate(train_loader):
            #控制DDP每步不更新的关键
            with ddp_model.no_sync():
                with torch.autocast(device_type="cuda" if device.type=="cuda" else "cpu", dtype=dtype):
                    outputs = ddp_model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                        labels=batch["labels"].to(device),
                    )
                    loss = outputs.loss / args.grad_accum

                if dtype == torch.float16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (it + 1) % args.grad_accum == 0:
                    if args.clip_grad is not None and args.clip_grad > 0:
                        if dtype == torch.float16:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), args.clip_grad)

                    if dtype == torch.float16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                    step_since_sync += 1
                    global_step += 1

                    if step_since_sync >= args.H:
                        # Periodic global Nesterov synchronization
                        ddp_model.eval()
                        torch.cuda.synchronize() if device.type == "cuda" else None
                        global_state.sync_step(ddp_model.module, mu=args.mu, lr_nes=args.lr_nes, world_size=world_size)
                        ddp_model.train()
                        step_since_sync = 0

            if is_main_process() and (global_step % 10 == 0) and ((it + 1) % args.grad_accum == 0):
                print(f"epoch {epoch} step {global_step} (local it {it}) loss={loss.item()*args.grad_accum:.4f}")

        # epoch boundary sync for good measure
        global_state.sync_step(ddp_model.module, mu=args.mu, lr_nes=args.lr_nes, world_size=world_size)

    # -------------------- Evaluate (rank 0) --------------------
    if rank == 0:
        ddp_model.module.eval()
        eval_subset = eval_ds
        if is_main_process():
            print("Evaluating on GSM8K test set...")
        correct = 0
        total = 0
        for idx in range(len(eval_subset)):
            batch = eval_subset[idx]
            input_ids = batch["input_ids"].unsqueeze(0).to(device)
            attention_mask = batch["attention_mask"].unsqueeze(0).to(device)
            with torch.no_grad(), torch.autocast(device_type="cuda" if device.type=="cuda" else "cpu", dtype=dtype):
                gen = ddp_model.module.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.eos_token_id,
                )
            full_text = tokenizer.decode(gen[0], skip_special_tokens=True)
            # Extract model's final number
            pred = extract_final_number(full_text)
            gold = extract_final_number(batch["gold_answer"])  # gold has '#### NNN'
            if pred is not None and gold is not None and pred.strip() == gold.strip():
                correct += 1
            total += 1
            if total % 50 == 0:
                print(f"Eval progress: {total} / {len(eval_subset)}; current acc={correct/total:.4f}")
        acc = correct / max(total, 1)
        print(f"GSM8K test accuracy: {acc:.4f} ({correct}/{total})")

    cleanup_distributed()

# ------------------------------ Collate ------------------------------
def pad_to_left(tokenizer, batch_tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    # left pad to max length in batch
    max_len = max(t.size(0) for t in batch_tensors)
    padded = []
    for t in batch_tensors:
        if t.size(0) < max_len:
            pad_len = max_len - t.size(0)
            pad_tensor = torch.full((pad_len,), tokenizer.pad_token_id, dtype=t.dtype)
            t = torch.cat([pad_tensor, t], dim=0)
        padded.append(t)
    return torch.stack(padded, dim=0)

def default_collate(features: Dict[str, torch.Tensor]):
    # features is a list[dict]; we implement custom collation for left padding
    if isinstance(features, dict):
        features = [features]
    keys = features[0].keys()
    tokenizer = default_collate.tokenizer  # type: ignore
    batch = {}
    for k in keys:
        if k in ("input_ids", "attention_mask", "labels"):
            batch[k] = pad_to_left(tokenizer, [f[k] for f in features])
        else:
            batch[k] = torch.stack([f[k] for f in features]) if torch.is_tensor(features[0][k]) else [f[k] for f in features]
    return batch

# attach tokenizer dynamically after creation

def _attach_tokenizer_to_collate(tokenizer):
    default_collate.tokenizer = tokenizer  # type: ignore

if __name__ == "__main__":
    # The tokenizer is needed inside collate; we set it after parsing.
    # But since argparse and model load are in main(), we call a tiny pre-parse here to get model name just for tokenizer.
    # Simpler approach: set a dummy, then main() will override.
    class DummyTok:
        pad_token_id = 0
    _attach_tokenizer_to_collate(DummyTok())
    main()
