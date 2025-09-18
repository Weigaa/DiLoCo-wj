#!/usr/bin/env python
# Minimal DiLoCo-style DDP (no AMP, no grad accumulation, no grad clipping)
# - Local AdamW every step; Nesterov sync every H steps

import argparse, os, re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def is_main(): return int(os.environ.get("RANK","0")) == 0
def setup_ddp():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
def cleanup_ddp():
    if dist.is_initialized():
        try: dist.barrier()
        except Exception: pass
        dist.destroy_process_group()

@dataclass
class Example: q:str; a:str

class GSM8KTrain(Dataset):
    def __init__(self, split, tok, max_len=2048):
        self.data=[Example(x["question"].strip(),x["answer"].strip()) for x in split]
        self.tok, self.max_len = tok, max_len
    def __len__(self): return len(self.data)
    def __getitem__(self,i):
        ex=self.data[i]
        prompt=("You are a helpful math tutor.\n"
                f"Question: {ex.q}\n"
                "Answer step by step and finish with '#### <final_number>'.\n"
                "Answer:")
        text=prompt+"\n"+ex.a
        inp=self.tok(text,max_length=self.max_len,truncation=True,padding=False,return_tensors="pt")
        labels=inp.input_ids.clone()
        with self.tok.as_target_tokenizer():
            p=self.tok(prompt,max_length=self.max_len,truncation=True,padding=False,return_tensors="pt")
        labels[0,:p.input_ids.size(1)]=-100
        return {"input_ids":inp.input_ids[0],"attention_mask":inp.attention_mask[0],"labels":labels[0]}

class GSM8KEval(Dataset):
    def __init__(self, split, tok, max_len=2048):
        self.rows=[(x["question"].strip(),x["answer"].strip()) for x in split]
        self.tok,self.max_len=tok,max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        q,a=self.rows[i]
        prompt=("You are a helpful math tutor.\n"
                f"Question: {q}\n"
                "Answer step by step and finish with '#### <final_number>'.\n"
                "Answer:")
        inp=self.tok(prompt,max_length=self.max_len,truncation=True,padding=False,return_tensors="pt")
        return {"input_ids":inp.input_ids[0],"attention_mask":inp.attention_mask[0],"gold_answer":a}

def pad_left(tok, seqs: Iterable[torch.Tensor]) -> torch.Tensor:
    L=max(t.size(0) for t in seqs); out=[]
    for t in seqs:
        if t.size(0)<L:
            pad=torch.full((L-t.size(0),),tok.pad_token_id,dtype=t.dtype)
            t=torch.cat([pad,t],0)
        out.append(t)
    return torch.stack(out,0)

def collate(batch: Dict[str,torch.Tensor]):
    if isinstance(batch,dict): batch=[batch]
    keys=batch[0].keys(); tok=collate.tok; out={}
    for k in keys:
        if k in ("input_ids","attention_mask","labels"):
            out[k]=pad_left(tok,[b[k] for b in batch])
        else:
            out[k]=[b[k] for b in batch]
    return out

class NesterovGlobalState:
    def __init__(self, model: torch.nn.Module):
        self.shadow, self.v = {}, {}
        for n,p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n]=p.detach().float().clone()
                self.v[n]=torch.zeros_like(p,dtype=torch.float32,device=p.device)
    @torch.no_grad()
    def sync_step(self, model: torch.nn.Module, mu: float, lr_nes: float, world: int):
        for n,p in model.named_parameters():
            if not p.requires_grad: continue
            dist.all_reduce(p.data, op=dist.ReduceOp.SUM)
            p_avg=(p.data/world).float()
            s,v=self.shadow[n],self.v[n]
            d=p_avg - s
            v.mul_(mu).add_(d)
            s.add_(-lr_nes*(mu*v + d))
            p.data.copy_(s.to(p.data.dtype))
        for _,p in model.named_parameters():
            if p.requires_grad: dist.broadcast(p.data, src=0)

NUM_RE=re.compile(r"####\s*([-+]?\d+(?:\.\d+)?)"); FALL=re.compile(r"([-+]?\d+(?:\.\d+)?)")
def extract_num(s:str)->Optional[str]:
    m=NUM_RE.search(s); 
    if m: return m.group(1)
    ms=list(FALL.finditer(s)); 
    return ms[-1].group(1) if ms else None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_name",type=str,default="allenai/OLMoE-1B-7B-0924") # 已存在于本机 HF 缓存
    ap.add_argument("--per_device_batch",type=int,default=1)
    ap.add_argument("--epochs",type=int,default=1)
    ap.add_argument("--lr",type=float,default=5e-5)
    ap.add_argument("--weight_decay",type=float,default=0.01)
    ap.add_argument("--H",type=int,default=20)
    ap.add_argument("--mu",type=float,default=0.9)
    ap.add_argument("--lr_nes",type=float,default=1.0)
    ap.add_argument("--max_length",type=int,default=1024)
    ap.add_argument("--max_new_tokens",type=int,default=128)
    ap.add_argument("--max_train_samples",type=int,default=-1)
    ap.add_argument("--max_eval_samples",type=int,default=-1)
    args=ap.parse_args()

    setup_ddp()
    local_rank=int(os.environ.get("LOCAL_RANK",0))
    rank=int(os.environ.get("RANK",0))
    world=int(os.environ.get("WORLD_SIZE",1))
    device=torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type=="cuda": torch.cuda.set_device(device)

    # 数据集与模型均从默认缓存读取（已下载）
    ds=load_dataset("gsm8k","main",download_mode="reuse_dataset_if_exists")
    if args.max_train_samples>0: ds["train"]=ds["train"].select(range(min(args.max_train_samples,len(ds["train"]))))
    if args.max_eval_samples>0:  ds["test"]= ds["test"]. select(range(min(args.max_eval_samples ,len(ds["test"]))))

    tok=AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    tok.padding_side="left"; collate.tok=tok

    model=AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.to(device)

    ddp=DDP(model,
            device_ids=[device.index] if device.type=="cuda" else None,
            output_device=device.index if device.type=="cuda" else None,
            find_unused_parameters=False,
            broadcast_buffers=False,
            gradient_as_bucket_view=True)

    train_ds=GSM8KTrain(ds["train"], tok, args.max_length)
    eval_ds =GSM8KEval (ds["test"],  tok, args.max_length)
    sampler =DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True)
    loader  =DataLoader(train_ds, batch_size=args.per_device_batch, sampler=sampler,
                        num_workers=2, pin_memory=True, collate_fn=collate)

    optim=torch.optim.AdamW([
        {"params":[p for n,p in model.named_parameters() if p.requires_grad and not any(k in n for k in ("bias","LayerNorm.weight","layer_norm.weight"))], "weight_decay":args.weight_decay},
        {"params":[p for n,p in model.named_parameters() if p.requires_grad and     any(k in n for k in ("bias","LayerNorm.weight","layer_norm.weight"))], "weight_decay":0.0},
    ], lr=args.lr, betas=(0.9,0.999), eps=1e-8)

    gstate=NesterovGlobalState(model)

    if is_main(): print(f"[Train] world={world} epochs={args.epochs} H={args.H} mu={args.mu} lr_nes={args.lr_nes}")
    step_since_sync=0
    for ep in range(args.epochs):
        sampler.set_epoch(ep); ddp.train()
        for it,b in enumerate(loader):
            print("process epoch", ep, "step", it)
            with ddp.no_sync():
                out=ddp(input_ids=b["input_ids"].to(device),
                        attention_mask=b["attention_mask"].to(device),
                        labels=b["labels"].to(device))
                loss=out.loss
                loss.backward()
            optim.step(); optim.zero_grad(set_to_none=True)
            step_since_sync+=1
            if step_since_sync>=args.H:
                ddp.eval()
                if device.type=="cuda": torch.cuda.synchronize()
                gstate.sync_step(ddp.module, mu=args.mu, lr_nes=args.lr_nes, world_size=world)
                ddp.train(); step_since_sync=0
            if is_main() and (it+1)%10==0:
                print(f"[ep {ep}] it {it+1} loss={loss.item():.4f}")
        gstate.sync_step(ddp.module, mu=args.mu, lr_nes=args.lr_nes, world_size=world)

    if rank==0:
        ddp.module.eval(); correct=0
        for i in range(len(eval_ds)):
            row=eval_ds[i]
            x=row["input_ids"].unsqueeze(0).to(device)
            m=row["attention_mask"].unsqueeze(0).to(device)
            with torch.no_grad():
                y=ddp.module.generate(input_ids=x, attention_mask=m, max_new_tokens=args.max_new_tokens,
                                      do_sample=False, num_beams=1, pad_token_id=tok.eos_token_id)
            pred=extract_num(tok.decode(y[0], skip_special_tokens=True))
            gold=extract_num(row["gold_answer"])
            if pred is not None and gold is not None and pred.strip()==gold.strip():
                correct+=1
            if (i+1)%50==0: print(f"[Eval] {i+1}/{len(eval_ds)} acc={correct/(i+1):.4f}")
        print(f"GSM8K test accuracy: {correct/len(eval_ds):.4f} ({correct}/{len(eval_ds)})")

    cleanup_ddp()

if __name__=="__main__":
    main()
