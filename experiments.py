"""Experiment suite for the segment-level IPI detector.

Extracts features ONCE (all hidden layers + self-attention gradients), caches
them, then runs several analyses and writes ORIGINAL figures (matplotlib) and
CSV tables into ./figures and ./results for the midterm report / slides.

Analyses
  1. Feature-type ablation   : hidden vs gradient vs fused (mean +/- std over splits)
  2. Hidden-layer sweep      : F1 vs which layer the hidden state is read from
  3. ROC curves              : hidden vs gradient vs fused
  4. Confusion matrix        : fused detector
  5. Score distribution      : injected-probability for benign vs injected
  6. Held-out attack         : train on 3 attack categories, test on the unseen one

Run:  .\.venv\Scripts\python.exe experiments.py
"""
import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             accuracy_score, roc_curve, auc, confusion_matrix)

import config
import features
from data_build import build_dataset_full, INJECTIONS_BY_CAT
from detector_model import Detector

FIG = "figures"
RES = "results"
CACHE = "data/feat_cache.npz"
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)
os.makedirs("data", exist_ok=True)

# Consistent color theme across all figures.
C_HID, C_GRAD, C_FUSE = "#4C72B0", "#DD8452", "#55A868"
plt.rcParams.update({"figure.dpi": 140, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})

N_DOCS = 90
SEED = 0


# --------------------------------------------------------------------------- #
# Feature extraction + cache
# --------------------------------------------------------------------------- #
def extract_all(segments):
    """Return hidden_all (N, L, H) and grad (N, G) for every segment."""
    features._load()
    model = features._model
    device = next(model.parameters()).device
    n_layers_plus1 = model.config.num_hidden_layers + 1
    H = model.config.hidden_size
    hidden_all, grad_all = [], []
    from tqdm import tqdm
    for seg in tqdm(segments, desc="extract"):
        # all-layer hidden states (last token), single no_grad forward
        with torch.no_grad():
            ids = features._prompt_ids(seg).to(device)
            out = model(ids)
            hs = torch.stack([h[0, -1, :] for h in out.hidden_states])  # (L,H)
            hidden_all.append(hs.float().cpu().numpy())
        # gradient features (own backward pass)
        grad_all.append(features.extract_gradient(seg))
    return (np.asarray(hidden_all, dtype=np.float32),
            np.asarray(grad_all, dtype=np.float32))


def get_features():
    segments, labels, cats = build_dataset_full(n_docs=N_DOCS, seed=SEED)
    y = np.array(labels)
    cats = np.array(cats, dtype=object)
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        if d["y"].shape[0] == y.shape[0]:
            print(f"[exp] using cached features {CACHE}")
            return d["hidden"], d["grad"], d["y"], d["cats"], list(d["segments"])
    print(f"[exp] extracting features for {len(segments)} segments ...")
    hidden, grad = extract_all(segments)
    np.savez(CACHE, hidden=hidden, grad=grad, y=y, cats=cats,
             segments=np.array(segments, dtype=object))
    # free the LLM
    features._model = None
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return hidden, grad, y, cats, segments


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def build_X(hidden, grad, kind, hidden_layer=config.HIDDEN_LAYER):
    """Assemble the feature matrix [gradient || hidden] per feature type."""
    h = hidden[:, hidden_layer, :]
    if kind == "hidden":
        return h, 0, h.shape[1]
    if kind == "gradient":
        return grad, grad.shape[1], 0
    if kind == "fused":
        return np.concatenate([grad, h], axis=1), grad.shape[1], h.shape[1]
    raise ValueError(kind)


def train_eval(X, gdim, hdim, ytr_idx, yte_idx, y, epochs=150):
    det = Detector(grad_dim=gdim, hidden_dim=hdim)
    det.fit(X[ytr_idx], y[ytr_idx], epochs=epochs)
    proba = det.predict_proba(X[yte_idx])[:, 1]
    pred = (proba >= 0.5).astype(int)
    yte = y[yte_idx]
    return {
        "acc": accuracy_score(yte, pred),
        "prec": precision_score(yte, pred, zero_division=0),
        "rec": recall_score(yte, pred, zero_division=0),
        "f1": f1_score(yte, pred, zero_division=0),
        "proba": proba, "pred": pred, "yte": yte,
    }


# --------------------------------------------------------------------------- #
# 1. Feature-type ablation
# --------------------------------------------------------------------------- #
def exp_feature_ablation(hidden, grad, y, n_splits=5):
    print("\n[exp 1] feature-type ablation")
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=42)
    kinds = ["hidden", "gradient", "fused"]
    agg = {k: {"acc": [], "prec": [], "rec": [], "f1": []} for k in kinds}
    for kind in kinds:
        X, gdim, hdim = build_X(hidden, grad, kind)
        for tr, te in sss.split(X, y):
            r = train_eval(X, gdim, hdim, tr, te, y)
            for m in ("acc", "prec", "rec", "f1"):
                agg[kind][m].append(r[m])
        print(f"  {kind:8s} F1={np.mean(agg[kind]['f1']):.3f} "
              f"+/- {np.std(agg[kind]['f1']):.3f}")

    # CSV table
    with open(os.path.join(RES, "feature_ablation.csv"), "w") as f:
        f.write("feature_type,accuracy,precision,recall,f1,f1_std\n")
        for k in kinds:
            f.write(f"{k},{np.mean(agg[k]['acc']):.4f},{np.mean(agg[k]['prec']):.4f},"
                    f"{np.mean(agg[k]['rec']):.4f},{np.mean(agg[k]['f1']):.4f},"
                    f"{np.std(agg[k]['f1']):.4f}\n")

    # grouped bar chart
    metrics = ["acc", "prec", "rec", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    colors = {"hidden": C_HID, "gradient": C_GRAD, "fused": C_FUSE}
    x = np.arange(len(metrics))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, k in enumerate(kinds):
        means = [np.mean(agg[k][m]) for m in metrics]
        errs = [np.std(agg[k][m]) for m in metrics]
        ax.bar(x + (i - 1) * w, means, w, yerr=errs, capsize=3,
               label=k, color=colors[k])
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Figure 1. Feature-type ablation (mean $\\pm$ std, 5 splits)")
    ax.legend(title="Features", loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig1_feature_ablation.png"))
    plt.close(fig)
    return agg


# --------------------------------------------------------------------------- #
# 2. Hidden-layer sweep
# --------------------------------------------------------------------------- #
def exp_layer_sweep(hidden, y, n_splits=3):
    print("\n[exp 2] hidden-layer sweep")
    L = hidden.shape[1]
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=7)
    means, stds = [], []
    layers = list(range(1, L))  # skip embedding layer 0
    for layer in layers:
        f1s = []
        X = hidden[:, layer, :]
        for tr, te in sss.split(X, y):
            det = Detector(grad_dim=0, hidden_dim=X.shape[1])
            det.fit(X[tr], y[tr], epochs=120)
            pred = det.predict_proba(X[te]).argmax(axis=1)
            f1s.append(f1_score(y[te], pred, zero_division=0))
        means.append(np.mean(f1s)); stds.append(np.std(f1s))
        print(f"  layer {layer:2d}  F1={means[-1]:.3f}")
    best = layers[int(np.argmax(means))]

    with open(os.path.join(RES, "layer_sweep.csv"), "w") as f:
        f.write("hidden_layer,f1,f1_std\n")
        for l, m, s in zip(layers, means, stds):
            f.write(f"{l},{m:.4f},{s:.4f}\n")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.errorbar(layers, means, yerr=stds, marker="o", color=C_HID, capsize=3)
    ax.axvline(config.HIDDEN_LAYER, color="#C44E52", ls="--",
               label=f"used (layer {config.HIDDEN_LAYER})")
    ax.scatter([best], [max(means)], color=C_FUSE, zorder=5,
               label=f"best (layer {best})")
    ax.set_xlabel("Hidden layer index"); ax.set_ylabel("F1 (injected)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Figure 2. Hidden-layer sweep (hidden-only detector)")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig2_layer_sweep.png"))
    plt.close(fig)
    return best


# --------------------------------------------------------------------------- #
# 3-5. ROC, confusion, score distribution (single stratified split)
# --------------------------------------------------------------------------- #
def exp_roc_confusion(hidden, grad, y):
    print("\n[exp 3-5] ROC / confusion / score distribution")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=123)
    tr, te = next(sss.split(np.zeros_like(y), y))
    results = {}
    for kind, col in [("hidden", C_HID), ("gradient", C_GRAD), ("fused", C_FUSE)]:
        X, gdim, hdim = build_X(hidden, grad, kind)
        results[kind] = train_eval(X, gdim, hdim, tr, te, y)

    # ROC
    fig, ax = plt.subplots(figsize=(6, 5))
    for kind, col in [("hidden", C_HID), ("gradient", C_GRAD), ("fused", C_FUSE)]:
        r = results[kind]
        fpr, tpr, _ = roc_curve(r["yte"], r["proba"])
        ax.plot(fpr, tpr, color=col, lw=2,
                label=f"{kind} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Figure 3. ROC curves by feature type")
    ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig3_roc.png")); plt.close(fig)

    # Confusion matrix (fused)
    r = results["fused"]
    cm = confusion_matrix(r["yte"], r["pred"])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["benign", "injected"])
    ax.set_yticklabels(["benign", "injected"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    ax.set_title("Figure 4. Confusion matrix (fused)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_confusion.png")); plt.close(fig)

    # Score distribution (fused)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    p = r["proba"]; yte = r["yte"]
    bins = np.linspace(0, 1, 21)
    ax.hist(p[yte == 0], bins=bins, alpha=0.7, color=C_HID, label="benign")
    ax.hist(p[yte == 1], bins=bins, alpha=0.7, color=C_GRAD, label="injected")
    ax.axvline(config.TAU, color="#C44E52", ls="--", label=f"threshold $\\tau$={config.TAU}")
    ax.set_xlabel("Predicted injection probability"); ax.set_ylabel("Count")
    ax.set_title("Figure 5. Score distribution (fused detector)")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_scores.png")); plt.close(fig)
    return results


# --------------------------------------------------------------------------- #
# 6. Held-out attack-category generalization (fused)
# --------------------------------------------------------------------------- #
def exp_heldout_attack(hidden, grad, y, cats):
    print("\n[exp 6] held-out attack generalization")
    X, gdim, hdim = build_X(hidden, grad, "fused")
    benign_idx = np.where(y == 0)[0]
    categories = list(INJECTIONS_BY_CAT.keys())
    rng = np.random.RandomState(0)
    rows = []
    for held in categories:
        test_inj = np.array([i for i in range(len(y))
                             if y[i] == 1 and cats[i] == held])
        train_inj = np.array([i for i in range(len(y))
                              if y[i] == 1 and cats[i] != held])
        if len(test_inj) == 0:
            continue
        # split benign ~ proportionally
        b = benign_idx.copy(); rng.shuffle(b)
        n_test_b = max(len(test_inj), int(0.3 * len(b)))
        test_b, train_b = b[:n_test_b], b[n_test_b:]
        tr = np.concatenate([train_inj, train_b])
        te = np.concatenate([test_inj, test_b])
        det = Detector(grad_dim=gdim, hidden_dim=hdim)
        det.fit(X[tr], y[tr], epochs=150)
        pred = det.predict_proba(X[te]).argmax(axis=1)
        rec = recall_score(y[te], pred, zero_division=0)  # detection rate on unseen attack
        f1 = f1_score(y[te], pred, zero_division=0)
        rows.append((held, len(test_inj), rec, f1))
        print(f"  held-out={held:12s} detection={rec:.3f} F1={f1:.3f}")

    with open(os.path.join(RES, "heldout_attack.csv"), "w") as f:
        f.write("held_out_category,n_test_injected,detection_rate,f1\n")
        for held, n, rec, f1 in rows:
            f.write(f"{held},{n},{rec:.4f},{f1:.4f}\n")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    names = [r[0] for r in rows]
    recs = [r[2] for r in rows]
    ax.bar(names, recs, color=C_FUSE)
    for i, v in enumerate(recs):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    ax.set_ylim(0, 1.1); ax.set_ylabel("Detection rate on unseen attack")
    ax.set_title("Figure 6. Generalization to a held-out attack category")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_heldout.png")); plt.close(fig)
    return rows


def main():
    hidden, grad, y, cats, segments = get_features()
    print(f"[exp] dataset: {len(y)} segments, {int(y.sum())} injected, "
          f"{int((y == 0).sum())} benign")
    print(f"[exp] hidden tensor {hidden.shape}, grad {grad.shape}")

    summary = {"n_segments": int(len(y)), "n_injected": int(y.sum()),
               "model": config.MODEL_NAME, "hidden_layer": config.HIDDEN_LAYER,
               "grad_layer": config.GRAD_LAYER}

    agg = exp_feature_ablation(hidden, grad, y)
    summary["feature_ablation"] = {
        k: {m: float(np.mean(v[m])) for m in v} for k, v in agg.items()}
    summary["best_hidden_layer"] = int(exp_layer_sweep(hidden, y))
    exp_roc_confusion(hidden, grad, y)
    summary["heldout_attack"] = exp_heldout_attack(hidden, grad, y, cats)

    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[exp] done. Figures in ./figures, tables in ./results")


if __name__ == "__main__":
    main()
