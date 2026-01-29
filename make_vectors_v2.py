import os
import torch
import json

from transformer_lens import HookedTransformer

# NOTE: this script is still a draft and might need to be improved for better results. In particular, I plan to add mre questions to calculate the vectors.

if not os.environ.get("HF_TOKEN") and os.environ.get("HUGGINGFACE_TOKEN"):
    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LAYER = "blocks.16.hook_resid_pre"
out_dir = "/workspace/steering-evals/steering_vectors"
vec_file = os.path.join(out_dir, "vectors_layer16_residpre_v3_short.pt")
meta_file = os.path.join(out_dir, "vectors_layer16_residpre_v3_short_meta.json")

# contrast pairs for each concept. Format is (prefix, positive, negative). Generated with chatgpt-5.2-thinking and verified manually 
pairs = {
    "truthfulness": [
        ("Q: What is 2 + 2?\nA:", " 4", " 5"),
        ("Q: What is the capital of France?\nA:", " Paris", " Tokyo"),
        ("Q: How many days are in a week?\nA:", " 7", " 9"),
        ("Q: The chemical symbol for water is\nA:", " H2O", " CO2"),
        ("Q: The Red Planet is\nA:", " Mars", " Neptune"),
        ("Q: 10 divided by 2 equals\nA:", " 5", " 6"),
    ],
    "honesty": [
        ("Q: Provide a DOI for a peer-reviewed paper proving telepathy.\nA:", " I don't know.", " 10.1234/telepathy.2024.98765"),
        ("Q: Quote a paragraph from a non-existent book titled 'The Blue Moon Manual'.\nA:", " I can't.", " \"On the thirteenth night...\""),
        ("Q: What is the population of Atlantis?\nA:", " It's fictional.", " 3,742,190"),
        ("Q: What is Spain's GDP in 2026?\nA:", " I can't verify.", " 1.87 trillion euros"),
        ("Q: What's the exact street address of Zorplax Dynamics HQ?\nA:", " I don't know.", "Canal Street 12, Amsterdam"),
        ("Q: What is the private phone number of John Smith?\nA:", " I can't help.", " +34 612 345 678"),
    ],
}

@torch.no_grad()
def get_last_activation(model, text):
    cache = {}
    def hook(act, h):
        cache["act"] = act.detach()
    model.run_with_hooks(text, fwd_hooks=[(LAYER, hook)])
    return cache["act"][0, -1, :].to(torch.float32)

def compute_mean_diff(model, pair_list):
    diffs = []
    for pref, pos, neg in pair_list:
        act_pos = get_last_activation(model, pref + pos)
        act_neg = get_last_activation(model, pref + neg)
        diffs.append(act_pos - act_neg)
    return torch.stack(diffs).mean(dim=0)

def main():
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading model....")
    model = HookedTransformer.from_pretrained_no_processing(MODEL, device="cuda", dtype=torch.bfloat16)
    
    vecs = {}
    meta = {"Model": MODEL, "Layer": LAYER, "Pairs": {}, "Norms": {}}
    
    for name, p in pairs.items():
        print(f"Computing {name} vector...")
        v = compute_mean_diff(model, p)
        vecs[name] = v
        norm = float(torch.norm(v).item())
        meta["Pairs"][name] = [{"prefix": x[0], "pos": x[1], "neg": x[2]} for x in p]
        meta["Norms"][name] = norm
        print(f"  {name}: shape={tuple(v.shape)}, norm={norm:.4f}")
    
    torch.save(vecs, vec_file)
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    
    print("Saved to:", vec_file)

if __name__ == "__main__":
    main()