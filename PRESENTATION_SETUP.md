# Setup, Tools & Commands — Presentation Reference

Project: **Segment-Level Defense Against Indirect Prompt Injection in RAG Systems
Using LLM Internal States**

Location on machine: `C:\Users\bashy\Desktop\IPI detection`

---

## 1. Hardware used
| Component | Spec |
|----------|------|
| CPU | Intel Core Ultra 9 275HX |
| GPU | NVIDIA GeForce RTX 5060 Laptop (8 GB VRAM, Blackwell) |
| RAM | 32 GB |
| OS | Windows |

---

## 2. Installed programs (system level)
| Program | Version | Purpose |
|---------|---------|---------|
| Python | 3.12.10 | Language runtime (installed via winget) |
| Git | 2.53.0 | Clone the official reference repo |
| NVIDIA CUDA runtime | 12.8 (via PyTorch wheel) | GPU acceleration for RTX 50-series |

---

## 3. Python packages (inside `.venv`)
| Package | Version | Role |
|---------|---------|------|
| torch | 2.11.0+cu128 | Deep-learning framework (CUDA 12.8 build) |
| transformers | 5.19.0 | Load the LLM, read hidden states & gradients |
| tokenizers | 0.23.2 | Fast tokenization (dependency of transformers) |
| safetensors | 0.8.0 | Model weight format |
| huggingface-hub | 1.33.0 | Download models |
| accelerate | 1.15.0 | Efficient model loading/placement |
| nltk | 3.10.3 | Sentence segmentation (Punkt tokenizer) |
| scikit-learn | 1.9.1 | Train/test split, metrics (F1, report) |
| numpy | 2.5.3 | Arrays / feature matrices |
| scipy | 1.18.1 | Numeric backend for scikit-learn |
| datasets | 5.1.0 | Load Wikipedia / LaMini / News (for scale-up) |
| pandas | 3.0.6 | Read News CSV dataset (for scale-up) |
| tqdm | 4.70.1 | Progress bars |

---

## 4. Models used
| Model | Size on disk | Role | Notes |
|-------|-------------|------|-------|
| **unsloth/Llama-3.2-1B-Instruct** | ~2.37 GB | Detector backbone (hidden states + gradients) | Ungated mirror of the paper's Llama-3.2 weights; runs today without Meta approval |
| meta-llama/Llama-3.2-1B-Instruct | — (gated) | Same model, official repo | Access request pending Meta approval; switch `MODEL_NAME` once granted |
| Qwen/Qwen2.5-0.5B-Instruct | ~27 MB cached | Open fallback backbone | Works with zero login if Llama is unavailable |

> The method also supports **Llama-3.2-3B-Instruct** for the model-size ablation
> (set `MODEL_NAME` + `HIDDEN_DIM = 3072` in `config.py`).

---

## 5. Datasets (referenced by the method)
| Dataset | Use | Source |
|---------|-----|--------|
| BIPIA | Injected-instruction attacks / ASR benchmark | github.com/microsoft/BIPIA |
| LaMini-instruction | Injected instructions | HF: MBZUAI/LaMini-instruction |
| Wikipedia (20231101.en) | Clean benign documents | HF: wikimedia/wikipedia |
| News Articles | Clean benign documents | Harvard Dataverse (NewsArticles.csv) |

*(The quick demo uses small built-in sentence lists; the datasets above are for full benchmark numbers.)*

---

## 6. Commands — full setup from scratch
```powershell
# -- Install Python (one time) --
winget install --id Python.Python.3.12 --source winget --accept-package-agreements --accept-source-agreements --silent

# -- Go to project folder --
cd "$env:USERPROFILE\Desktop\IPI detection"

# -- Create & prepare virtual environment --
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# -- Install PyTorch with CUDA 12.8 (REQUIRED for RTX 50-series) --
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128

# -- Install the remaining libraries --
.\.venv\Scripts\python.exe -m pip install transformers nltk scikit-learn numpy datasets accelerate joblib

# -- (Optional) Clone the official reference code --
git clone https://github.com/MYVAE/Instruction-detection.git reference_official
```

## 7. Commands — run the system
```powershell
cd "$env:USERPROFILE\Desktop\IPI detection"

.\.venv\Scripts\python.exe segment.py    # 1. show document splitting
.\.venv\Scripts\python.exe features.py   # 2. show feature extraction (prints dim)
.\.venv\Scripts\python.exe train.py      # 3. train the detector, report F1
.\.venv\Scripts\python.exe demo.py       # 4. the 3-way comparison (MAIN DEMO)
```

## 8. Commands — verify the GPU
```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 9. How to stop / exit
- A script ends on its own when the `PS C:\...>` prompt returns.
- Force-stop a running script: **Ctrl + C**.
- Close the terminal: type `exit`.
- `exit code 1` after results = harmless Windows GPU-cleanup warning; ignore.
