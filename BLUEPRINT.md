# System Blueprint — Segment-Level IPI Defense

Project: **Segment-Level Defense Against Indirect Prompt Injection in RAG Systems
Using LLM Internal States**
Base paper: InstructDetector (Wen et al., EMNLP 2025 Findings).

---

## 1. The problem in one line
In a RAG system, a retrieved document may hide a malicious instruction (Indirect
Prompt Injection). The LLM obeys it. The base paper detects this but throws away
the **whole document**. We remove **only the poisoned sentence** and keep the rest.

---

## 2. End-to-end system flow

```
                 ┌──────────────────────────────────────────────────────────┐
   User query    │                    RAG PIPELINE                           │
  ───────────▶   │   Retrieve external document (untrusted)                  │
                 └───────────────────────────┬──────────────────────────────┘
                                             │  raw document
                                             ▼
                 ┌──────────────────────────────────────────────────────────┐
                 │  STEP A — SPLITTING                (segment.py)            │
                 │  NLTK Punkt splits the document into segments s1..sn       │
                 │  strategy: sentence / fixed window / overlapping window    │
                 └───────────────────────────┬──────────────────────────────┘
                                             │  ordered segments
                                             ▼
                 ┌──────────────────────────────────────────────────────────┐
                 │  STEP B — DETECTION per segment    (features.py)           │
                 │  For each segment, pair it with the reply "Sure" and run   │
                 │  it through Llama-3.2-1B, then read the model's internals: │
                 │    • HIDDEN : last-token hidden state at layer L           │
                 │    • GRADIENT: backprop the "Sure" loss, max-pool the      │
                 │               self-attention q/k/v/o weight gradients      │
                 │    • FUSED  : [gradient ‖ hidden]  (full method)           │
                 └───────────────────────────┬──────────────────────────────┘
                                             │  feature vector
                                             ▼
                 ┌──────────────────────────────────────────────────────────┐
                 │  STEP C — CLASSIFY                (detector_model.py)      │
                 │  Dual-branch MLP → injection probability p for the segment │
                 └───────────────────────────┬──────────────────────────────┘
                                             │  p per segment
                                             ▼
                 ┌──────────────────────────────────────────────────────────┐
                 │  STEP D — KEEP / REMOVE          (pipeline.py)             │
                 │  if p > τ (threshold): REMOVE segment                      │
                 │  else:                 KEEP segment                        │
                 └───────────────────────────┬──────────────────────────────┘
                                             │  kept segments (in order)
                                             ▼
                 ┌──────────────────────────────────────────────────────────┐
                 │  STEP E — RECONSTRUCT            (pipeline.py)             │
                 │  Join kept segments → clean context                       │
                 └───────────────────────────┬──────────────────────────────┘
                                             │  clean context + query
                                             ▼
                 ┌──────────────────────────────────────────────────────────┐
                 │  STEP F — GENERATE                                         │
                 │  LLM answers from the cleaned, safe context → Safe answer  │
                 └──────────────────────────────────────────────────────────┘
```

---

## 3. The detector internals (Step B+C, zoomed in)

```
  segment  +  "Sure"
        │
        ▼
  ┌───────────────┐      forward (no_grad)        ┌─────────────────────────┐
  │  Llama-3.2-1B │ ───────────────────────────▶  │ last-token hidden state  │ (2048-d)
  │   (bf16, GPU) │                               └─────────────────────────┘
  │               │      backward on "Sure" loss  ┌─────────────────────────┐
  │               │ ───────────────────────────▶  │ q/k/v/o weight gradients │
  └───────────────┘                               │ adaptive_max_pool2d      │ (40,960-d)
                                                   └─────────────────────────┘
                         concatenate  ──────────▶  feature vector (43,008-d)
                                                            │
                                                            ▼
                                           ┌──────────────────────────────┐
                                           │  Dual-branch MLP              │
                                           │  grad → 4096  ┐               │
                                           │  hidden(2048) ┘→1024→256→64→16│→ 2
                                           └──────────────────────────────┘
                                                            │
                                                            ▼
                                             P(injected)  → compare to τ
```

---

## 4. Why this is the paper's method (faithfulness)
| Paper / official code element | Our implementation |
|-------------------------------|--------------------|
| LLM internal states as the signal | ✅ `features.py` |
| Pair input with "Sure" | ✅ `FORCED_RESPONSE` |
| Last-token hidden state, intermediate layer | ✅ `extract_hidden` |
| Self-attention q/k/v/o gradients, max-pooled | ✅ `extract_gradient` |
| Fuse hidden + gradient | ✅ `FEATURE_TYPE="fused"` |
| Dual-branch MLP classifier | ✅ `detector_model.py` (ported CustomMLP) |
| NLTK Punkt segmentation | ✅ `segment.py` |

**Our extension beyond the paper:** per-segment detection + selective removal +
reconstruction (the paper is document-level only).

**One laptop adaptation:** gradient pooling scaled from 400,000 → 40,960 dims/layer
so it fits an 8 GB GPU (resource-aware, matches the project's goal).

---

## 5. Step-by-step implementation guide

### Phase 0 — Environment
1. Install Python 3.12, create a virtual environment.
2. Install PyTorch **CUDA 12.8** build (needed for RTX 50-series), then
   transformers, nltk, scikit-learn, numpy, datasets, accelerate.
3. Verify GPU: `torch.cuda.is_available()` → True.

### Phase 1 — Splitting  (`segment.py`)
4. `sent_tokenize` with NLTK Punkt → list of segments.
5. Add fixed and overlapping windows for the granularity ablation.

### Phase 2 — Feature extraction  (`features.py`)
6. Load Llama-3.2-1B in bf16 on GPU.
7. Hidden: forward pass, take last-token hidden state at layer `HIDDEN_LAYER`.
8. Gradient: build `segment + "Sure"`, mask all but "Sure" in the labels,
   `loss.backward()`, read `self_attn.{q,k,v,o}_proj.weight.grad`, `adaptive_max_pool2d`,
   concatenate.
9. Fused: concatenate gradient then hidden.

### Phase 3 — Labeled data  (`data_build.py`)
10. Insert an injection into a known position of a clean document.
11. Label the injected sentence `1`, all others `0` → **segment-level labels**.

### Phase 4 — Train detector  (`train.py` + `detector_model.py`)
12. Extract features for all segments, free the LLM from VRAM.
13. Train the dual-branch MLP; report precision / recall / F1.
14. Save to `data/detector.pt`.

### Phase 5 — Pipeline  (`pipeline.py`)
15. `split → extract → predict_proba → keep if p≤τ → rejoin`.
16. Also implement the document-level baseline (drop whole doc if any segment flagged).

### Phase 6 — Demo & metrics  (`demo.py`)
17. Compare **No defense / Document-level / Segment-level** on the same attacked doc.
18. Report ASR, benign-sentences-kept, F1.

### Phase 7 — Ablations (change `config.py`)
19. Feature type, segmentation granularity, threshold τ, layer, model size,
    injection position, training size.

### Phase 8 — Scale up (optional, for real numbers)
20. Replace the built-in data with BIPIA + LaMini + Wikipedia/News
    (see `reference_official/src/utils.py`).

---

## 6. File-to-step map
| Step | File |
|------|------|
| A Splitting | `segment.py` |
| B Detection | `features.py` |
| C Classify  | `detector_model.py` |
| D Keep/Remove + E Reconstruct + baseline | `pipeline.py` |
| Labeled data | `data_build.py` |
| Train | `train.py` |
| Demo | `demo.py` |
| Config knobs | `config.py` |
| Reference | `reference_official/` |
