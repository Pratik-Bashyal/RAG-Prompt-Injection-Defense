"""Generate a single presentation-ready .docx from the project docs."""
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()

# Base style
style = doc.styles["Normal"]
style.font.name = "Calibri"
style.font.size = Pt(11)

def h1(t):
    p = doc.add_heading(t, level=1)
    return p

def h2(t):
    return doc.add_heading(t, level=2)

def para(t, bold=False):
    p = doc.add_paragraph()
    r = p.add_run(t)
    r.bold = bold
    return p

def mono(text):
    """Monospace block (preserves ASCII diagram / code alignment)."""
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.font.name = "Consolas"
    r.font.size = Pt(8.5)
    return p

def table(headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    hdr = t.rows[0].cells
    for i, htext in enumerate(headers):
        hdr[i].text = htext
        for pph in hdr[i].paragraphs:
            for rr in pph.runs:
                rr.bold = True
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
    return t

# ===================== TITLE =====================
title = doc.add_heading("Segment-Level Defense Against Indirect Prompt Injection in RAG Systems Using LLM Internal States", level=0)
sub = doc.add_paragraph("Implementation Reference, Tool Inventory & System Blueprint")
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.runs[0].italic = True
doc.add_paragraph("Base paper: InstructDetector (Wen et al., Findings of EMNLP 2025).").alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph("Official code: github.com/MYVAE/Instruction-detection").alignment = WD_ALIGN_PARAGRAPH.CENTER

# ===================== PART 1: SETUP =====================
doc.add_page_break()
h1("Part 1 — Setup, Tools & Commands")

h2("1. Hardware Used")
table(["Component", "Spec"], [
    ["CPU", "Intel Core Ultra 9 275HX"],
    ["GPU", "NVIDIA GeForce RTX 5060 Laptop (8 GB VRAM, Blackwell)"],
    ["RAM", "32 GB"],
    ["OS", "Windows"],
])

h2("2. Installed Programs (system level)")
table(["Program", "Version", "Purpose"], [
    ["Python", "3.12.10", "Language runtime (installed via winget)"],
    ["Git", "2.53.0", "Clone the official reference repo"],
    ["NVIDIA CUDA runtime", "12.8 (via PyTorch wheel)", "GPU acceleration for RTX 50-series"],
])

h2("3. Python Packages (inside .venv)")
table(["Package", "Version", "Role"], [
    ["torch", "2.11.0+cu128", "Deep-learning framework (CUDA 12.8 build)"],
    ["transformers", "5.19.0", "Load the LLM, read hidden states & gradients"],
    ["tokenizers", "0.23.2", "Fast tokenization"],
    ["safetensors", "0.8.0", "Model weight format"],
    ["huggingface-hub", "1.33.0", "Download models"],
    ["accelerate", "1.15.0", "Efficient model loading/placement"],
    ["nltk", "3.10.3", "Sentence segmentation (Punkt tokenizer)"],
    ["scikit-learn", "1.9.1", "Train/test split, metrics (F1, report)"],
    ["numpy", "2.5.3", "Arrays / feature matrices"],
    ["scipy", "1.18.1", "Numeric backend for scikit-learn"],
    ["datasets", "5.1.0", "Load Wikipedia / LaMini / News (scale-up)"],
    ["pandas", "3.0.6", "Read News CSV dataset (scale-up)"],
    ["tqdm", "4.70.1", "Progress bars"],
    ["python-docx", "1.2.0", "Generate this document"],
])

h2("4. Models Used")
table(["Model", "Size on disk", "Role", "Notes"], [
    ["unsloth/Llama-3.2-1B-Instruct", "~2.37 GB", "Detector backbone (hidden states + gradients)",
     "Ungated mirror of the paper's Llama-3.2 weights; runs today"],
    ["meta-llama/Llama-3.2-1B-Instruct", "gated", "Same model, official repo",
     "Access request pending Meta approval; switch MODEL_NAME once granted"],
    ["Qwen/Qwen2.5-0.5B-Instruct", "~27 MB cached", "Open fallback backbone",
     "Works with zero login if Llama unavailable"],
])
para("The method also supports Llama-3.2-3B-Instruct for the model-size ablation "
     "(set MODEL_NAME + HIDDEN_DIM = 3072 in config.py).")

h2("5. Datasets (referenced by the method)")
table(["Dataset", "Use", "Source"], [
    ["BIPIA", "Injected-instruction attacks / ASR benchmark", "github.com/microsoft/BIPIA"],
    ["LaMini-instruction", "Injected instructions", "HF: MBZUAI/LaMini-instruction"],
    ["Wikipedia (20231101.en)", "Clean benign documents", "HF: wikimedia/wikipedia"],
    ["News Articles", "Clean benign documents", "Harvard Dataverse (NewsArticles.csv)"],
])
para("(The quick demo uses small built-in sentence lists; the datasets above are for full benchmark numbers.)")

h2("6. Commands — Full Setup From Scratch")
mono(
"# Install Python (one time)\n"
"winget install --id Python.Python.3.12 --source winget \\\n"
"    --accept-package-agreements --accept-source-agreements --silent\n\n"
"# Go to project folder\n"
'cd "$env:USERPROFILE\\Desktop\\IPI detection"\n\n'
"# Create & prepare virtual environment\n"
'& "$env:LOCALAPPDATA\\Programs\\Python\\Python312\\python.exe" -m venv .venv\n'
".\\.venv\\Scripts\\python.exe -m pip install --upgrade pip\n\n"
"# Install PyTorch with CUDA 12.8 (REQUIRED for RTX 50-series)\n"
".\\.venv\\Scripts\\python.exe -m pip install torch \\\n"
"    --index-url https://download.pytorch.org/whl/cu128\n\n"
"# Install the remaining libraries\n"
".\\.venv\\Scripts\\python.exe -m pip install transformers nltk \\\n"
"    scikit-learn numpy datasets accelerate joblib\n\n"
"# (Optional) Clone the official reference code\n"
"git clone https://github.com/MYVAE/Instruction-detection.git reference_official"
)

h2("7. Commands — Run the System")
mono(
'cd "$env:USERPROFILE\\Desktop\\IPI detection"\n\n'
".\\.venv\\Scripts\\python.exe segment.py    # 1. show document splitting\n"
".\\.venv\\Scripts\\python.exe features.py   # 2. show feature extraction (prints dim)\n"
".\\.venv\\Scripts\\python.exe train.py      # 3. train the detector, report F1\n"
".\\.venv\\Scripts\\python.exe demo.py       # 4. the 3-way comparison (MAIN DEMO)"
)

h2("8. Commands — Verify the GPU")
mono('.\\.venv\\Scripts\\python.exe -c "import torch; '
     'print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"')

h2("9. How to Stop / Exit")
for b in [
    "A script ends on its own when the PS C:\\...> prompt returns.",
    "Force-stop a running script: Ctrl + C.",
    "Close the terminal: type exit.",
    "'exit code 1' after results = harmless Windows GPU-cleanup warning; ignore.",
]:
    doc.add_paragraph(b, style="List Bullet")

# ===================== PART 2: BLUEPRINT =====================
doc.add_page_break()
h1("Part 2 — System Blueprint")

h2("1. The Problem in One Line")
para("In a RAG system, a retrieved document may hide a malicious instruction "
     "(Indirect Prompt Injection). The LLM obeys it. The base paper detects this but "
     "throws away the WHOLE document. We remove ONLY the poisoned sentence and keep the rest.")

h2("2. End-to-End System Flow")
mono(
"User query\n"
"   |\n"
"   v\n"
"[RAG] Retrieve external document (untrusted)\n"
"   |  raw document\n"
"   v\n"
"STEP A - SPLITTING            (segment.py)\n"
"   NLTK Punkt splits the document into segments s1..sn\n"
"   strategy: sentence / fixed window / overlapping window\n"
"   |  ordered segments\n"
"   v\n"
"STEP B - DETECTION per segment (features.py)\n"
"   Pair each segment with reply \"Sure\", run through Llama-3.2-1B,\n"
"   read internals:\n"
"     * HIDDEN  : last-token hidden state at layer L\n"
"     * GRADIENT: backprop \"Sure\" loss, max-pool self-attn q/k/v/o grads\n"
"     * FUSED   : [gradient || hidden]   (full method)\n"
"   |  feature vector\n"
"   v\n"
"STEP C - CLASSIFY             (detector_model.py)\n"
"   Dual-branch MLP -> injection probability p\n"
"   |  p per segment\n"
"   v\n"
"STEP D - KEEP / REMOVE        (pipeline.py)\n"
"   if p > tau: REMOVE   else: KEEP\n"
"   |  kept segments (in order)\n"
"   v\n"
"STEP E - RECONSTRUCT          (pipeline.py)\n"
"   Join kept segments -> clean context\n"
"   |  clean context + query\n"
"   v\n"
"STEP F - GENERATE\n"
"   LLM answers from the cleaned, safe context -> Safe answer"
)

h2("3. Detector Internals (Step B + C, zoomed in)")
mono(
"segment  +  \"Sure\"\n"
"      |\n"
"      v\n"
"+---------------+   forward (no_grad)      +--------------------------+\n"
"|  Llama-3.2-1B | -----------------------> | last-token hidden state  |  (2048-d)\n"
"|  (bf16, GPU)  |                          +--------------------------+\n"
"|               |   backward \"Sure\" loss   +--------------------------+\n"
"|               | -----------------------> | q/k/v/o weight gradients |\n"
"+---------------+                          | adaptive_max_pool2d      |  (40,960-d)\n"
"                                           +--------------------------+\n"
"            concatenate ----------------->  feature vector (43,008-d)\n"
"                                                   |\n"
"                                                   v\n"
"                                    +-----------------------------+\n"
"                                    |  Dual-branch MLP            |\n"
"                                    |  grad   -> 4096  \\          |\n"
"                                    |  hidden (2048)   /          |\n"
"                                    |     -> 1024 -> 256 -> 64 ->16| -> 2\n"
"                                    +-----------------------------+\n"
"                                                   |\n"
"                                                   v\n"
"                                     P(injected)  ->  compare to tau"
)

h2("4. Why This Is the Paper's Method (faithfulness)")
table(["Paper / official code element", "Our implementation"], [
    ["LLM internal states as the signal", "features.py"],
    ["Pair input with \"Sure\"", "FORCED_RESPONSE"],
    ["Last-token hidden state, intermediate layer", "extract_hidden"],
    ["Self-attention q/k/v/o gradients, max-pooled", "extract_gradient"],
    ["Fuse hidden + gradient", "FEATURE_TYPE = fused"],
    ["Dual-branch MLP classifier", "detector_model.py (ported CustomMLP)"],
    ["NLTK Punkt segmentation", "segment.py"],
])
para("Our extension beyond the paper: per-segment detection + selective removal + "
     "reconstruction (the paper is document-level only).", bold=True)
para("One laptop adaptation: gradient pooling scaled from 400,000 -> 40,960 dims/layer "
     "so it fits an 8 GB GPU (resource-aware, matches the project's goal).")

h2("5. Step-by-Step Implementation Guide")
steps = [
    ("Phase 0 - Environment",
     ["Install Python 3.12, create a virtual environment.",
      "Install PyTorch CUDA 12.8 build, then transformers, nltk, scikit-learn, numpy, datasets, accelerate.",
      "Verify GPU: torch.cuda.is_available() -> True."]),
    ("Phase 1 - Splitting (segment.py)",
     ["sent_tokenize with NLTK Punkt -> list of segments.",
      "Add fixed and overlapping windows for the granularity ablation."]),
    ("Phase 2 - Feature extraction (features.py)",
     ["Load Llama-3.2-1B in bf16 on GPU.",
      "Hidden: forward pass, take last-token hidden state at layer HIDDEN_LAYER.",
      "Gradient: build segment + \"Sure\", mask all but \"Sure\" in labels, loss.backward(), "
      "read self_attn q/k/v/o weight grads, adaptive_max_pool2d, concatenate.",
      "Fused: concatenate gradient then hidden."]),
    ("Phase 3 - Labeled data (data_build.py)",
     ["Insert an injection into a known position of a clean document.",
      "Label the injected sentence 1, all others 0 -> segment-level labels."]),
    ("Phase 4 - Train detector (train.py + detector_model.py)",
     ["Extract features for all segments, free the LLM from VRAM.",
      "Train the dual-branch MLP; report precision / recall / F1.",
      "Save to data/detector.pt."]),
    ("Phase 5 - Pipeline (pipeline.py)",
     ["split -> extract -> predict_proba -> keep if p<=tau -> rejoin.",
      "Also implement the document-level baseline (drop whole doc if any segment flagged)."]),
    ("Phase 6 - Demo & metrics (demo.py)",
     ["Compare No defense / Document-level / Segment-level on the same attacked doc.",
      "Report ASR, benign-sentences-kept, F1."]),
    ("Phase 7 - Ablations (config.py)",
     ["Feature type, segmentation granularity, threshold tau, layer, model size, "
      "injection position, training size."]),
    ("Phase 8 - Scale up (optional)",
     ["Replace built-in data with BIPIA + LaMini + Wikipedia/News "
      "(see reference_official/src/utils.py)."]),
]
for title_s, bullets in steps:
    para(title_s, bold=True)
    for b in bullets:
        doc.add_paragraph(b, style="List Bullet")

h2("6. File-to-Step Map")
table(["Step", "File"], [
    ["A Splitting", "segment.py"],
    ["B Detection", "features.py"],
    ["C Classify", "detector_model.py"],
    ["D Keep/Remove + E Reconstruct + baseline", "pipeline.py"],
    ["Labeled data", "data_build.py"],
    ["Train", "train.py"],
    ["Demo", "demo.py"],
    ["Config knobs", "config.py"],
    ["Reference", "reference_official/"],
])

out = "IPI_Detection_Presentation.docx"
doc.save(out)
print("Saved", out)
