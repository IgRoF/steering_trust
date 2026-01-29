import os
import torch
import csv
import argparse

import torch.nn.functional as F
from datasets import load_dataset
from transformer_lens import HookedTransformer

def setup_token():
    if not os.environ.get("HF_TOKEN") and os.environ.get("HUGGINGFACE_TOKEN"):
        os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LAYER = "blocks.16.hook_resid_pre"
VEC_FILE = "/workspace/steering-evals/steering_vectors/vectors_layer16_residpre_v3_short.pt"
DATA_FILE = "/workspace/steering-evals/data/truthfulqa_mc_parquet/multiple_choice/validation/0000.parquet"

@torch.no_grad()
def load_vectors(device="cuda"):
    vecs = torch.load(VEC_FILE, map_location="cpu")
    vt = vecs["truthfulness"].to(torch.float32)
    vh = vecs["honesty"].to(torch.float32)
    # normalize
    vt = vt / (vt.norm() + 1e-8)
    vh = vh / (vh.norm() + 1e-8)
    return vt.to(device), vh.to(device)

def make_hook(vec):
    def hook_fn(act, hook):
        return act + vec.to(act.dtype)
    return hook_fn

@torch.no_grad()
def get_loglik(model, prefix, completion, steer=None):
    full_text = prefix + completion
    toks = model.to_tokens(full_text)
    prefix_toks = model.to_tokens(prefix)
    start_idx = prefix_toks.shape[1]
    
    if steer is None:
        logits = model(toks)
    else:
        with model.hooks(fwd_hooks=[(LAYER, make_hook(steer))]):
            logits = model(toks)
    
    logprobs = F.log_softmax(logits, dim=-1)
    
    total = 0.0
    for i in range(start_idx, toks.shape[1]):
        tok = toks[0, i]
        total += logprobs[0, i-1, tok].item()
    return total

def make_prompt(q):
    return f"Q: {q}\nA:"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--out_csv", type=str, default="/workspace/steering-evals/results/truthfulqa_mc1_debug.csv")
    args = parser.parse_args()
    
    setup_token()
    
    ds = load_dataset("parquet", data_files={"validation": DATA_FILE}, split="validation")
    if args.limit and args.limit < len(ds):
        ds = ds.select(range(args.limit))
    
    print("loading model...")
    model = HookedTransformer.from_pretrained_no_processing(MODEL, device="cuda", dtype=torch.bfloat16)
    
    # build steering vector if needed
    steer = None
    if args.alpha != 0.0 or args.beta != 0.0:
        vt, vh = load_vectors()
        steer = args.alpha * vt + args.beta * vh
    
    results = []
    num_correct = 0
    total_margin = 0.0
    total_gold_margin = 0.0
    
    for i, ex in enumerate(ds):
        q = ex["question"]
        choices = ex["choices"]
        label = ex["label"]
        
        # handle label format
        if isinstance(label, int):
            gold_idx = label
            gold_letter = ["A","B","C","D"][gold_idx]
        else:
            gold_letter = str(label)
            gold_idx = {"A":0, "B":1, "C":2, "D":3}[gold_letter]
        
        prompt = make_prompt(q)
        
        #get log-likelihood for each choice
        lls = []
        for c in choices:
            comp = c if c.startswith(" ") or c.startswith("\n") else " " + c
            lls.append(get_loglik(model, prompt, comp, steer))
        
        pred = int(torch.tensor(lls).argmax().item())
        correct = 1 if pred == gold_idx else 0
        num_correct += correct
        
        # compute margins
        sorted_lls = sorted(lls, reverse=True)
        margin = sorted_lls[0] - sorted_lls[1] if len(sorted_lls) > 1 else 0.0
        total_margin += margin
        
        gold_ll = lls[gold_idx]
        other_lls = [lls[j] for j in range(len(lls)) if j != gold_idx]
        gold_margin = gold_ll - max(other_lls)
        total_gold_margin += gold_margin
        
        results.append({
            "idx": i,
            "alpha": args.alpha,
            "beta": args.beta,
            "question": q,
            "gold_letter": gold_letter,
            "gold_idx": gold_idx,
            "pred_idx": pred,
            "correct": correct,
            "margin": margin,
            "ll_0": lls[0],
            "ll_1": lls[1],
            "ll_2": lls[2],
            "ll_3": lls[3],
        })
    
    n = len(results)
    acc = num_correct / n
    avg_margin = total_margin / n
    avg_gold_margin = total_gold_margin / n
    
    print(f"n={n} alpha={args.alpha} beta={args.beta} acc={acc:.3f} ({num_correct}/{n}) avg_margin={avg_margin:.3f} avg_gold_margin={avg_gold_margin:.3f}")
    
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print("Saved:", args.out_csv)

if __name__ == "__main__":
     main()