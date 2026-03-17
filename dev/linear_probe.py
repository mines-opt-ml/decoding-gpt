import sys
import json
import torch
import numpy as np
from transformer_lens import HookedTransformer
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression, Ridge

MODEL = "pythia-14m"
N_DOCS = 80
MAX_SEQ_LEN = 256
N_DISPLAY = 10
OUTPUT = "linear_probe.json"

# ── Feature functions ──────────────────────────────────────────────
# Signature: list[str] -> list[bool]  or  list[str] -> list[int]

def in_quotes(tokens: list[str]) -> list[bool]:
    """True for tokens inside double-quoted strings.
    Uses count-of-quotes-so-far mod 2."""
    depth = 0
    out = []
    for t in tokens:
        out.append(depth % 2 == 1)
        depth += t.count('"')
    return out

def tokens_since_newline(tokens: list[str]) -> list[int]:
    """Number of tokens since the most recent newline."""
    count = 0
    out = []
    for t in tokens:
        if "\n" in t:
            count = 0
        out.append(count)
        count += 1
    return out

FEATURES = {
    "in_quotes": (in_quotes, "bool"),
    "tokens_since_newline": (tokens_since_newline, "int"),
}

# ── Config ─────────────────────────────────────────────────────────

feature_name = sys.argv[1] if len(sys.argv) > 1 else "in_quotes"
feature_fn, feature_type = FEATURES[feature_name]
print(f"Feature: {feature_name} (type: {feature_type})")

# ── Load model & data ─────────────────────────────────────────────

model = HookedTransformer.from_pretrained(MODEL)
ds = load_dataset("NeelNanda/pile-10k", split="train")

all_tok_strs: list[list[str]] = []
all_labels: list[list] = []
all_tok_ids: list[torch.Tensor] = []

for text in ds["text"]:
    toks = model.to_tokens(text, prepend_bos=True)
    if toks.shape[1] < 16:
        continue
    toks = toks[0, :MAX_SEQ_LEN]
    strs = [model.tokenizer.decode(t.item()) for t in toks]
    labels = feature_fn(strs)
    all_tok_strs.append(strs)
    all_labels.append(labels)
    all_tok_ids.append(toks)
    if len(all_tok_strs) >= N_DOCS:
        break

print(f"Collected {len(all_tok_strs)} documents")

# ── Collect hidden states ─────────────────────────────────────────

n_layers = model.cfg.n_layers
layer_names = ["embed"] + [f"L{i}" for i in range(n_layers)]
hook_names = ["blocks.0.hook_resid_pre"] + [
    f"blocks.{i}.hook_resid_post" for i in range(n_layers)
]

# all_hiddens[layer_name][doc_idx] = ndarray (seq_len, d_model)
all_hiddens: dict[str, list[np.ndarray]] = {ln: [] for ln in layer_names}

print("Running forward passes...")
with torch.no_grad():
    for idx, tok_ids in enumerate(all_tok_ids):
        _, cache = model.run_with_cache(tok_ids.unsqueeze(0))
        for lname, hname in zip(layer_names, hook_names):
            all_hiddens[lname].append(cache[hname][0].cpu().numpy())
        if (idx + 1) % 20 == 0:
            print(f"  {idx + 1}/{len(all_tok_ids)}")

# ── Train / test split by document ────────────────────────────────

n_train = int(len(all_tok_strs) * 0.75)
train_idx = list(range(n_train))
test_idx = list(range(n_train, len(all_tok_strs)))

train_y = np.concatenate([np.array(all_labels[i], dtype=float) for i in train_idx])
test_y = np.concatenate([np.array(all_labels[i], dtype=float) for i in test_idx])

if feature_type == "bool":
    baseline = float(max(test_y.mean(), 1 - test_y.mean()))
else:
    baseline = 0.0  # R² of always-predict-mean

# ── Train probes per layer ─────────────────────────────────────────

print("Training probes...")
layer_results = []
trained_models: dict = {}

for lname in layer_names:
    train_X = np.concatenate([all_hiddens[lname][i] for i in train_idx])
    test_X = np.concatenate([all_hiddens[lname][i] for i in test_idx])

    if feature_type == "bool":
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(train_X, train_y.astype(int))
        preds = clf.predict(test_X)
        acc = float((preds == test_y.astype(int)).mean())
        layer_results.append({"label": lname, "accuracy": round(acc, 4)})
    else:
        reg = Ridge(alpha=1.0)
        reg.fit(train_X, train_y)
        preds = reg.predict(test_X)
        ss_res = float(((test_y - preds) ** 2).sum())
        ss_tot = float(((test_y - test_y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        layer_results.append({"label": lname, "r2": round(r2, 4)})

    trained_models[lname] = clf if feature_type == "bool" else reg
    print(f"  {lname}: {layer_results[-1]}")

# ── Build display examples ─────────────────────────────────────────

examples = []
for di in test_idx[:N_DISPLAY]:
    preds_by_layer: dict[str, list] = {}
    for lname in layer_names:
        X_doc = all_hiddens[lname][di]
        m = trained_models[lname]
        if feature_type == "bool":
            preds_by_layer[lname] = m.predict(X_doc).astype(int).tolist()
        else:
            p = m.predict(X_doc)
            preds_by_layer[lname] = np.clip(np.round(p), 0, None).astype(int).tolist()

    examples.append({
        "tokens": all_tok_strs[di],
        "ground_truth": [int(x) for x in all_labels[di]],
        "predictions": preds_by_layer,
    })

# ── Write output ───────────────────────────────────────────────────

data = {
    "feature_name": feature_name,
    "feature_type": feature_type,
    "n_train": n_train,
    "n_test": len(test_idx),
    "baseline": round(baseline, 4),
    "layers": layer_results,
    "layer_names": layer_names,
    "examples": examples,
}

with open(OUTPUT, "w") as f:
    json.dump(data, f)

print(f"Wrote {OUTPUT}")
