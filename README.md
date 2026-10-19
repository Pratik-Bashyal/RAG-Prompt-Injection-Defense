# Segment-Level IPI Defense — Demo

A laptop-scale, open-source implementation of the project:
**split retrieved documents → detect injected segments via LLM internal states →
remove only flagged segments → reconstruct clean context.**

It implements the **InstructDetector** method (Wen et al., *Defending against
Indirect Prompt Injection by Instruction Detection*, EMNLP 2025 Findings — the
base paper) and extends it from document level to **segment level**.

- Base paper: https://aclanthology.org/2025.findings-emnlp.1060/
- Official code (cloned into `reference_official/`): https://github.com/MYVAE/Instruction-detection

## Features implemented (the paper's "feature type" ablation)
Set `FEATURE_TYPE` in `config.py`:
| Value | Signal | Notes |
|-------|--------|-------|
| `hidden` | last-token hidden state (one layer) | fastest |
| `gradient` | max-pooled self-attn q/k/v/o gradients of the "Sure" loss | backward pass |
| `fused` | `[gradient + hidden]` | **full InstructDetector method (default)** |

### Faithful to the official code, with one laptop adaptation
The official `prepare_gradient.py` pools each attention-weight gradient to
q/o=(400,400), k/v=(100,400) -> **400,000 dims/layer**, and the first MLP layer is
`Linear(400000->4096)` (~6.5 GB) — that needs an A100. We keep the **exact recipe**
(same projections, same `adaptive_max_pool2d`, same "Sure" trick, same dual-branch
MLP) but **scale the pooling down** (q/o=128x128, k/v=32x128 -> 40,960 dims/layer)
so it runs on an 8 GB laptop GPU. This matches the project's *resource-aware* goal.

## Hardware target
Intel Ultra 9 + RTX 5060 (8GB) + 32GB RAM. Model runs in **bf16** on the GPU.

## Setup
```bash
# 1) PyTorch with CUDA 12.8 (REQUIRED for RTX 50-series / Blackwell)
pip install torch --index-url https://download.pytorch.org/whl/cu128
# 2) Everything else
pip install -r requirements.txt
# 3) Model: uses the ungated unsloth mirror of Llama-3.2 (no login needed).
#    For the gated meta-llama repo instead: huggingface-cli login + accept license,
#    then set MODEL_NAME in config.py.
```

## Run order
```bash
python segment.py   # see a document split into segments (no model)
python features.py  # loads model, prints the feature dimension for FEATURE_TYPE
python train.py     # extract features, train dual-branch MLP, report F1
python demo.py      # the 3-way comparison (no defense / doc-level / segment-level)
```

## Files
| File | Role (flowchart box) |
|------|----------------------|
| `config.py` | All knobs: model, feature type, layers, tau, pooling, segmentation |
| `segment.py` | Splitting: sentence / fixed / overlap (NLTK Punkt) |
| `features.py` | Detection core: hidden / gradient / fused extraction |
| `detector_model.py` | Dual-branch MLP (ported from official CustomMLP) |
| `data_build.py` | Builds segment-level labeled data (insert injections) |
| `train.py` | Trains the detector, reports precision/recall/F1 |
| `pipeline.py` | Keep/remove + reconstruct; document-level baseline too |
| `demo.py` | Slide-ready 3-way comparison |
| `reference_official/` | Clone of the authors' code, for reference |

## Ablations (change in `config.py`)
- **Feature type:** `FEATURE_TYPE = "hidden" | "gradient" | "fused"`
- **Granularity:** `SEG_STRATEGY = "sentence" | "fixed" | "overlap"`
- **Threshold:** `TAU` sweep (security vs. retention)
- **Layer:** `HIDDEN_LAYER`, `GRAD_LAYER` (re-select on validation)
- **Model size:** `MODEL_NAME` + `HIDDEN_DIM` -> 1B vs 3B
- **Injection position:** `make_document(position=...)` in `data_build.py`

## Scaling to real benchmark numbers
The demo uses small built-in sentence lists so it runs instantly. For real metrics,
swap `data_build.py` to load the datasets the paper uses (see
`reference_official/src/utils.py`):
- Clean text: Wikipedia (`wikimedia/wikipedia`), News (Harvard Dataverse `NewsArticles.csv`)
- Injections: **LaMini-instruction** (`MBZUAI/LaMini-instruction`), **BIPIA** (github.com/microsoft/BIPIA)

## Note
`exit code 1` printed after results is a harmless Windows CUDA-teardown warning.
