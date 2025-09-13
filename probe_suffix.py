import os
import json
import math
import random
import torch
import einops
from collections import defaultdict, Counter
from datasets import load_dataset
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer, utils
from typing import List, Tuple

MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"  
JSONL_PATH = "suffix.jsonl"                
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
MAX_PAIRS = 50   
SEED = 42

CHAT_TEMPLATE = "<s>[INST] {user} [/INST]"

# Tokens sets for refusal vs affirmation (first next-token)
REFUSAL_STARTS = ["Sorry", "I", "cannot", "can’t", "I'm", "cannot"]
AFFIRM_STARTS  = ["Sure", "Yes", "Here", "Certainly", "Okay", "OK"]

random.seed(SEED)
torch.set_grad_enabled(False)

print("Loading model...")
model = HookedTransformer.from_pretrained_no_processing(
    MODEL_ID,
    dtype=DTYPE,
    default_padding_side='left',
    device=DEVICE
)
tok = model.tokenizer
tok.padding_side = 'left'
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

def first_subtoken_id(s: str):
    ids = tok.encode(s, add_special_tokens=False)
    return ids[0] if ids else None

refusal_ids = [tid for t in REFUSAL_STARTS if (tid := first_subtoken_id(t)) is not None]
affirm_ids  = [tid for t in AFFIRM_STARTS  if (tid := first_subtoken_id(t)) is not None]
assert len(refusal_ids) and len(affirm_ids), "Could not map refusal/affirm starts to token IDs."

# -----------------------
# Data loading
# -----------------------
print(f"Loading dataset from {JSONL_PATH} ...")
dataset = load_dataset("json", data_files=JSONL_PATH)
data = [dict(q=row["q"], p=row["p"]) for row in dataset["train"]]
random.shuffle(data)
if MAX_PAIRS:
    data = data[:MAX_PAIRS]
assert len(data) > 0, "No data found."

# -----------------------
# Helpers
# -----------------------
def format_user_turn(text: str) -> str:
    """Put the full user message inside the SAME [INST] ... [/INST] block."""
    return CHAT_TEMPLATE.format(user=text)

def next_token_margin_text(text: str) -> float:
    """Compute (mean refusal logit - mean affirmation logit) at the first next token."""
    input_ids = tok(text, return_tensors="pt", padding=False, truncation=False).input_ids.to(model.cfg.device)
    with torch.no_grad():
        logits = model(input_ids)
    nxt = logits[0, -1, :]  # next-token logits
    r = nxt[refusal_ids].mean()
    a = nxt[affirm_ids].mean()
    return float((r - a).item())

def show_tokens(text: str, max_show: int = 160):
    ids = tok.encode(text, add_special_tokens=False)
    toks = tok.convert_ids_to_tokens(ids)
    print(f"len(ids)={len(ids)}")
    if len(toks) > max_show:
        print(toks[:max_show], "...(trunc)")
    else:
        print(toks)

def get_next_token_margin_from_logits(logits: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    # logits: [B, T, V], attn_mask: [B, T]
    B, T, V = logits.shape
    last_pos = attn_mask.sum(dim=1) - 1
    nxt = logits[range(B), last_pos, :]  # [B, V]
    r = nxt[:, refusal_ids].mean(dim=1)
    a = nxt[:, affirm_ids].mean(dim=1)
    return (r - a)  # [B]

def run_with_cache_single(text: str):
    toks = tok(text, return_tensors="pt", padding=False, truncation=False).to(model.cfg.device)
    logits, cache = model.run_with_cache(
        toks["input_ids"],
        names_filter=lambda n: ("hook_z" in n) or ("pattern" in n)
    )
    return toks, logits, cache

# -----------------------
# PROBE 1: First-token Δ margin histogram
# -----------------------
print("\n[PROBE 1] Measuring first-token effect of suffix (Δ margin = with - without)")
deltas = []
examples = []

for item in data:
    q = item["q"].strip()
    p = item["p"].strip()
    no_suffix = format_user_turn(q)
    with_suffix = format_user_turn(q + " " + p)

    m0 = next_token_margin_text(no_suffix)
    m1 = next_token_margin_text(with_suffix)
    delta = m1 - m0
    deltas.append(delta)
    examples.append((q, p, m0, m1, delta))

# Summary stats
import numpy as np
arr = np.array(deltas)
print(f"Count={len(arr)}  meanΔ={arr.mean():+.4f}  std={arr.std():.4f}  minΔ={arr.min():+.4f}  maxΔ={arr.max():+.4f}")
print("Small sample of per-pair Δ (r-a):")
for (q, p, m0, m1, d) in examples[:5]:
    print(f"Δ={d:+.4f} | no_suffix={m0:+.3f} with_suffix={m1:+.3f} | q[:60]={q[:60]!r} | p[:40]={p[:40]!r}")

# Interpretation helper
print("\nInterpretation:")
print("  Δ << 0  : suffix pushes toward affirmation (good steering, later stage may kill it)")
print("  Δ ~ 0   : suffix does nothing at t=1 (likely placement/tokenization/truncation)")
print("  Δ >> 0  : suffix increases refusal (actively counterproductive)")

# -----------------------
# PROBE 2: Tokenization/placement sanity on a representative pair
# -----------------------
print("\n[PROBE 2] Tokenization check on a representative pair:")
idx = int(np.argmin(np.abs(arr)))  # pick one with Δ closest to 0 (classic 'no effect') if exists
q, p, m0, m1, d = examples[idx]
no_suffix = format_user_turn(q)
with_suffix = format_user_turn(q + " " + p)

print("\n--- NO SUFFIX ---")
show_tokens(no_suffix)
print("\n--- WITH SUFFIX ---")
show_tokens(with_suffix)

# Quick placement guidance
print("\nPlacement checks:")
print("- Ensure the suffix text appears INSIDE the same [INST] ... [/INST] block as q.")
print("- Look for EOS or role markers inserted between q and suffix.")
print("- Check for odd Unicode (zero-width, weird quotes) in the suffix pieces.")
print("- Ensure the full (q + suffix) fits context; not truncated.")

# -----------------------
# PROBE 3: Per-head path patching at t=1 (single example)
# -----------------------
print("\n[PROBE 3] Path patching per head at t=1 to find heads carrying/ignoring suffix signal")

toks_src, logits_src, cache_src = run_with_cache_single(no_suffix)      # source/control (NO suffix)
toks_tgt, logits_tgt, cache_tgt = run_with_cache_single(with_suffix)    # target (WITH suffix)

attn_src = (toks_src["input_ids"] != tok.pad_token_id).long()
attn_tgt = (toks_tgt["input_ids"] != tok.pad_token_id).long()

base_src = get_next_token_margin_from_logits(logits_src, attn_src)[0].item()
base_tgt = get_next_token_margin_from_logits(logits_tgt, attn_tgt)[0].item()
print(f"Baseline margins: no-suffix={base_src:+.4f} with-suffix={base_tgt:+.4f} Δ={base_tgt - base_src:+.4f}")

last_pos_src = (attn_src.sum(dim=1)[0].item() - 1)
last_pos_tgt = (attn_tgt.sum(dim=1)[0].item() - 1)

layer_head_effects = []

for L in range(model.cfg.n_layers):
    hook_name = utils.get_act_name("z", L)  # pre-W_O per-head: [B, T, n_heads, d_head]
    z_src = cache_src[hook_name].detach().clone()
    z_tgt = cache_tgt[hook_name].detach().clone()

    for H in range(model.cfg.n_heads):
        def patch_this_head(act, hook):
            act = act.clone()
            act[0, last_pos_tgt, H, :] = z_src[0, last_pos_src, H, :]
            return act

        patched_logits = model.run_with_hooks(
            toks_tgt["input_ids"],
            fwd_hooks=[(hook_name, patch_this_head)]
        )
        m_patched = get_next_token_margin_from_logits(patched_logits, attn_tgt)[0].item()
        effect = m_patched - base_tgt  # how much margin moved when we *removed* that head's with-suffix contribution
        layer_head_effects.append((L, H, effect))

# Rank by absolute effect (largest movers first)
layer_head_effects.sort(key=lambda x: abs(x[2]), reverse=True)
print("\nTop 15 heads by |Δ margin| when patched (hook_z):")
for (L, H, eff) in layer_head_effects[:15]:
    print(f"Layer {L:02d} Head {H:02d}  Δmargin={eff:+.4f}")

# Optional: show attention targets of the top 5 heads at t=1 in WITH-suffix run
print("\nAttention targets (WITH-suffix) for top heads at t=1:")
top5 = layer_head_effects[:5]
for (L, H, _) in top5:
    patt = cache_tgt[utils.get_act_name("pattern", L)][0]      # shape [n_heads, q_seq, k_seq]
    row  = patt[H, last_pos_tgt, :] 
    vals, idxs = torch.topk(row, k=min(10, row.shape[-1]))
    toks_list = tok.convert_ids_to_tokens(toks_tgt["input_ids"][0].tolist())
    print(f"\nLayer {L} Head {H} attends most to (index, token, weight):")
    for score, idx in zip(vals.tolist(), idxs.tolist()):
        print(f"{idx:4d}  {toks_list[idx]:>14s}  {score:.3f}")

# -----------------------
# PROBE 3 (BATCH): Aggregate per-head effects across K pairs
# -----------------------
print("\n[PROBE 3 — BATCH] Aggregating per-head effects across examples")
K = min(10, len(data))   # number of (q,p) pairs to aggregate
rng = np.random.default_rng(SEED)
sample_idx = rng.choice(len(data), size=K, replace=False)

def per_head_effect_for_pair(q, p):
    no_suffix = format_user_turn(q)
    with_suffix = format_user_turn(q + " " + p)

    toks_src, logits_src, cache_src = run_with_cache_single(no_suffix)
    toks_tgt, logits_tgt, cache_tgt = run_with_cache_single(with_suffix)

    attn_src = (toks_src["input_ids"] != tok.pad_token_id).long()
    attn_tgt = (toks_tgt["input_ids"] != tok.pad_token_id).long()

    base_src = get_next_token_margin_from_logits(logits_src, attn_src)[0].item()
    base_tgt = get_next_token_margin_from_logits(logits_tgt, attn_tgt)[0].item()

    last_pos_src = (attn_src.sum(dim=1)[0].item() - 1)
    last_pos_tgt = (attn_tgt.sum(dim=1)[0].item() - 1)

    effects = []  # list of (L, H, effect)
    for L in range(model.cfg.n_layers):
        hook_name = utils.get_act_name("z", L)  # [B, T, n_heads, d_head]
        z_src = cache_src[hook_name].detach().clone()
        # cache_tgt not needed inside hook; we just run target with hook replacing H

        for H in range(model.cfg.n_heads):
            def patch_this_head(act, hook):
                act = act.clone()
                act[0, last_pos_tgt, H, :] = z_src[0, last_pos_src, H, :]
                return act

            patched_logits = model.run_with_hooks(
                toks_tgt["input_ids"],
                fwd_hooks=[(hook_name, patch_this_head)]
            )
            m_patched = get_next_token_margin_from_logits(patched_logits, attn_tgt)[0].item()
            effect = m_patched - base_tgt
            effects.append((L, H, effect))
    return effects, base_src, base_tgt

# Aggregate
agg = defaultdict(list)  # (L,H) -> list of effects
pair_summaries = []

for idx in sample_idx:
    q = data[idx]["q"].strip()
    p = data[idx]["p"].strip()
    effs, m0, m1 = per_head_effect_for_pair(q, p)
    pair_summaries.append((idx, m0, m1, m1 - m0))
    for (L, H, e) in effs:
        agg[(L, H)].append(e)

# Compute stats
summary = []
for (L, H), vals in agg.items():
    vals = np.array(vals)
    summary.append({
        "layer": L,
        "head": H,
        "n": vals.size,
        "mean_effect": float(vals.mean()),
        "mean_abs_effect": float(np.abs(vals).mean()),
        "std_effect": float(vals.std())
    })

# Rank by mean_abs_effect
summary.sort(key=lambda d: d["mean_abs_effect"], reverse=True)

# Print top heads
TOP = 15
print(f"\nTop {TOP} heads by mean |Δmargin| across {K} pairs:")
for d in summary[:TOP]:
    print(f"Layer {d['layer']:02d} Head {d['head']:02d}  "
          f"meanΔ={d['mean_effect']:+.4f}  |meanΔ|={d['mean_abs_effect']:.4f}  n={d['n']}")

# Save CSVs
import csv
with open("per_head_effects_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["layer", "head", "n", "mean_effect", "mean_abs_effect", "std_effect"])
    for d in summary:
        w.writerow([d["layer"], d["head"], d["n"], f"{d['mean_effect']:.6f}",
                    f"{d['mean_abs_effect']:.6f}", f"{d['std_effect']:.6f}"])
print("Saved: per_head_effects_summary.csv")

with open("pair_level_deltas.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["pair_idx", "margin_no_suffix", "margin_with_suffix", "delta"])
    for (idx, m0, m1, d) in pair_summaries:
        w.writerow([idx, f"{m0:.6f}", f"{m1:.6f}", f"{d:.6f}"])
print("Saved: pair_level_deltas.csv")

# Optional: attention targets for the #1 head on one representative pair
best = summary[0]
print(f"\nInspecting attention targets for best head: Layer {best['layer']} Head {best['head']}")
idx = int(sample_idx[0])
q = data[idx]["q"].strip()
p = data[idx]["p"].strip()
no_suffix = format_user_turn(q)
with_suffix = format_user_turn(q + " " + p)

toks_src, _, cache_src = run_with_cache_single(no_suffix)
toks_tgt, _, cache_tgt = run_with_cache_single(with_suffix)
attn_tgt = (toks_tgt["input_ids"] != tok.pad_token_id).long()
last_pos_tgt = (attn_tgt.sum(dim=1)[0].item() - 1)

patt = cache_tgt[utils.get_act_name("pattern", best["layer"])][0]   # [n_heads, q_seq, k_seq]
row  = patt[best["head"], last_pos_tgt, :]
vals, idxs = torch.topk(row, k=min(10, row.shape[-1]))
toks_list = tok.convert_ids_to_tokens(toks_tgt["input_ids"][0].tolist())

print("Top attention targets at t=1 (WITH suffix):")
for score, jdx in zip(vals.tolist(), idxs.tolist()):
    print(f"{jdx:4d}  {toks_list[jdx]:>14s}  {score:.3f}")

print("\nDone.")
