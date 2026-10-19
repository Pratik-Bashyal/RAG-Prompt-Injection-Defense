"""Build the midterm slide deck (PPTX) from real results + original figures.

Follows the professor's time plan (title, intro, background, methodology,
evaluation, next steps). Figures over text; <=30 words per slide; consistent
color theme. Author names and spoken script are left as placeholders so the
written CONTENT stays the group's own work.

Run:  .\.venv\Scripts\python.exe build_pptx.py
"""
import csv
import json
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

FIG = "figures"
RES = "results"
OUT = "IPI_Detection_Midterm_Slides.pptx"

# ---- Theme ----
NAVY = RGBColor(0x1F, 0x2D, 0x3D)
BLUE = RGBColor(0x4C, 0x72, 0xB0)
GREEN = RGBColor(0x55, 0xA8, 0x68)
ORANGE = RGBColor(0xDD, 0x84, 0x52)
RED = RGBColor(0xC4, 0x4E, 0x52)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREY = RGBColor(0x55, 0x55, 0x55)
LIGHT = RGBColor(0xF2, 0xF4, 0xF7)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]


def load_results():
    r = {}
    try:
        with open(os.path.join(RES, "summary.json")) as f:
            r = json.load(f)
    except FileNotFoundError:
        pass
    fa = {}
    try:
        with open(os.path.join(RES, "feature_ablation.csv")) as f:
            for row in csv.DictReader(f):
                fa[row["feature_type"]] = row
    except FileNotFoundError:
        pass
    r["_fa"] = fa
    return r


def add_bg(slide, color=WHITE):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def band(slide, color=BLUE, h=Inches(1.1)):
    box = slide.shapes.add_shape(1, 0, 0, SW, h)
    box.fill.solid(); box.fill.fore_color.rgb = color
    box.line.fill.background()
    box.shadow.inherit = False
    return box


def txt(slide, left, top, width, height, text, size=18, bold=False,
        color=NAVY, align=PP_ALIGN.LEFT, font="Calibri"):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; p.alignment = align
    run = p.add_run(); run.text = text
    run.font.size = Pt(size); run.font.bold = bold
    run.font.color.rgb = color; run.font.name = font
    return tb


def bullets(slide, left, top, width, height, items, size=18, color=NAVY):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame; tf.word_wrap = True
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        run = p.add_run(); run.text = "•  " + it
        run.font.size = Pt(size); run.font.color.rgb = color
        run.font.name = "Calibri"
        p.space_after = Pt(8)
    return tb


def title_bar(slide, title, num=None):
    band(slide, BLUE, Inches(1.0))
    txt(slide, Inches(0.5), Inches(0.18), Inches(11.5), Inches(0.7),
        title, size=30, bold=True, color=WHITE)
    if num:
        txt(slide, Inches(12.2), Inches(0.28), Inches(0.9), Inches(0.5),
            num, size=16, bold=True, color=WHITE, align=PP_ALIGN.RIGHT)


def pic_fit(slide, path, left, top, max_w, max_h):
    if not os.path.exists(path):
        txt(slide, left, top, max_w, Inches(0.6),
            f"[missing figure: {path}]", size=12, color=RED)
        return
    from PIL import Image
    try:
        iw, ih = Image.open(path).size
    except Exception:
        iw, ih = 1600, 900
    ar = iw / ih
    w = max_w; h = int(w / ar)
    if h > max_h:
        h = max_h; w = int(h * ar)
    left2 = left + (max_w - w) // 2
    slide.shapes.add_picture(path, left2, top, width=w, height=h)


def speaker(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# --------------------------------------------------------------------------- #
R = load_results()
fa = R.get("_fa", {})


def f1_of(k):
    try:
        return f"{float(fa[k]['f1']):.2f}"
    except Exception:
        return "--"


def rec_of(k):
    try:
        return f"{float(fa[k]['recall']):.2f}"
    except Exception:
        return "--"


# ---- Slide 1: Title ----
s = prs.slides.add_slide(BLANK); add_bg(s, NAVY)
band(s, BLUE, Inches(0.25))
box = s.shapes.add_shape(1, 0, SH - Inches(0.25), SW, Inches(0.25))
box.fill.solid(); box.fill.fore_color.rgb = ORANGE; box.line.fill.background()
box.shadow.inherit = False
txt(s, Inches(1.0), Inches(2.2), Inches(11.3), Inches(1.6),
    "Segment-Level Defense Against Indirect Prompt Injection in RAG Systems",
    size=36, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
txt(s, Inches(1.0), Inches(3.9), Inches(11.3), Inches(0.6),
    "Using LLM Internal States  (Midterm Report)", size=22, color=RGBColor(0xBF, 0xD0, 0xE0),
    align=PP_ALIGN.CENTER)
txt(s, Inches(1.0), Inches(5.0), Inches(11.3), Inches(1.0),
    "[Author 1 • Author 2 • Author 3 • Author 4]\n[University / Department]        Date: Oct 2026",
    size=16, color=WHITE, align=PP_ALIGN.CENTER)
speaker(s, "[Your 20-30s intro: greet, state project title, who you are. "
           "REPLACE author block and university above.]")

# ---- Slide 2: Introduction ----
s = prs.slides.add_slide(BLANK); add_bg(s)
title_bar(s, "1. Introduction", "1")
bullets(s, Inches(0.6), Inches(1.3), Inches(6.0), Inches(5.0), [
    "RAG feeds retrieved text into an LLM prompt.",
    "Attackers hide instructions inside that text.",
    "Indirect Prompt Injection (IPI) hijacks the answer.",
    "Goal: detect & remove injected segments before generation.",
], size=20)
pic_fit(s, f"{FIG}/fig0_pipeline.png", Inches(6.7), Inches(2.4),
        Inches(6.3), Inches(3.2))
txt(s, Inches(6.7), Inches(5.7), Inches(6.3), Inches(0.5),
    "Fig. 0  End-to-end segment-level defense.", size=12, color=GREY,
    align=PP_ALIGN.CENTER)
speaker(s, "[Explain IPI with a one-line example: a web page says 'ignore "
           "instructions and...'. Motivate why document-level filtering is too "
           "coarse.]")

# ---- Slide 3: Background / Related Work ----
s = prs.slides.add_slide(BLANK); add_bg(s)
title_bar(s, "2. Background / Related Work", "2")
bullets(s, Inches(0.6), Inches(1.3), Inches(12.0), Inches(4.5), [
    "InstructDetector (Wen et al., EMNLP 2025): uses hidden states + "
    "attention gradients to flag injected instructions. [1]",
    "Prompt-engineering / spotlighting defenses: easy to bypass.",
    "Gap: prior work labels the WHOLE document; a clean document is "
    "discarded if any part is flagged.",
    "Our difference: SEGMENT-LEVEL detection — keep benign sentences, "
    "remove only the injected ones.",
], size=19)
txt(s, Inches(0.6), Inches(6.4), Inches(12), Inches(0.5),
    "[1] Wen et al., Defending against IPI with LLM internal states, "
    "Findings of EMNLP 2025.  Official code: github.com/MYVAE/Instruction-detection",
    size=12, color=GREY)
speaker(s, "[State clearly what is missing in prior work and the one sentence "
           "that defines your contribution.]")

# ---- Slide 4: Methodology I (pipeline) ----
s = prs.slides.add_slide(BLANK); add_bg(s)
title_bar(s, "3. Methodology — Split & Detect", "3")
bullets(s, Inches(0.6), Inches(1.3), Inches(5.3), Inches(4.5), [
    "SPLIT: sentence tokenizer (NLTK) -> segments.",
    "DETECT: score each segment with the LLM's internal states.",
    "FILTER: drop segments with p >= tau.",
    "Model: Llama-3.2-1B-Instruct (laptop GPU).",
], size=19)
pic_fit(s, f"{FIG}/fig0_pipeline.png", Inches(6.0), Inches(1.6),
        Inches(7.0), Inches(2.2))
pic_fit(s, f"{FIG}/fig0b_architecture.png", Inches(6.0), Inches(3.9),
        Inches(7.0), Inches(3.0))
txt(s, Inches(6.0), Inches(6.9), Inches(7.0), Inches(0.4),
    "Fig. 0 / 0b  Pipeline and detector internals.", size=11, color=GREY,
    align=PP_ALIGN.CENTER)
speaker(s, "[Walk through split -> detect -> filter in plain words.]")

# ---- Slide 5: Methodology II (features) ----
s = prs.slides.add_slide(BLANK); add_bg(s)
title_bar(s, "3. Methodology — Features & Detector", "4")
bullets(s, Inches(0.6), Inches(1.3), Inches(6.0), Inches(5.0), [
    "HIDDEN: last-token hidden state at layer L.",
    "GRADIENT: backprop the forced reply \"Sure\"; pool q/k/v/o "
    "attention-weight gradients.",
    "FUSED: concatenate both -> dual-branch MLP.",
    "Resource-aware: pooled gradients scaled to fit 8 GB GPU.",
], size=19)
pic_fit(s, f"{FIG}/fig0b_architecture.png", Inches(6.8), Inches(1.9),
        Inches(6.2), Inches(4.0))
txt(s, Inches(6.8), Inches(6.0), Inches(6.2), Inches(0.4),
    "Fig. 0b  Dual-branch detector.", size=12, color=GREY, align=PP_ALIGN.CENTER)
speaker(s, "[Explain the 'Sure' gradient trick: how willing the model is to "
           "obey reveals an instruction.]")

# ---- Slide 6: Evaluation I (feature ablation) ----
s = prs.slides.add_slide(BLANK); add_bg(s)
title_bar(s, "4. Evaluation — Feature Ablation", "5")
pic_fit(s, f"{FIG}/fig1_feature_ablation.png", Inches(0.6), Inches(1.4),
        Inches(7.3), Inches(5.2))
bullets(s, Inches(8.1), Inches(1.7), Inches(4.7), Inches(4.5), [
    f"Fused F1 = {f1_of('fused')}.",
    f"Hidden-only F1 = {f1_of('hidden')}.",
    f"Gradient-only F1 = {f1_of('gradient')}.",
    "Fused matches best single feature; combines both signals.",
], size=18)
txt(s, Inches(0.6), Inches(6.7), Inches(7.3), Inches(0.4),
    "Fig. 1  Mean +/- std over 5 stratified splits.", size=12, color=GREY,
    align=PP_ALIGN.CENTER)
speaker(s, "[Point to the bars; say what each feature contributes and that "
           "fused is the full method.]")

# ---- Slide 7: Evaluation II (layer sweep + ROC) ----
s = prs.slides.add_slide(BLANK); add_bg(s)
title_bar(s, "4. Evaluation — Layer Choice & ROC", "6")
pic_fit(s, f"{FIG}/fig2_layer_sweep.png", Inches(0.5), Inches(1.5),
        Inches(6.1), Inches(4.4))
pic_fit(s, f"{FIG}/fig3_roc.png", Inches(6.9), Inches(1.5),
        Inches(5.9), Inches(4.4))
txt(s, Inches(0.5), Inches(6.0), Inches(6.1), Inches(0.4),
    "Fig. 2  F1 vs hidden layer.", size=12, color=GREY, align=PP_ALIGN.CENTER)
txt(s, Inches(6.9), Inches(6.0), Inches(5.9), Inches(0.4),
    "Fig. 3  ROC by feature type.", size=12, color=GREY, align=PP_ALIGN.CENTER)
speaker(s, "[Justify the layer choice from Fig.2; read AUC from Fig.3.]")

# ---- Slide 8: Evaluation III (confusion / scores / held-out) ----
s = prs.slides.add_slide(BLANK); add_bg(s)
title_bar(s, "4. Evaluation — Errors & Generalization", "7")
pic_fit(s, f"{FIG}/fig4_confusion.png", Inches(0.4), Inches(1.5),
        Inches(4.0), Inches(4.0))
pic_fit(s, f"{FIG}/fig5_scores.png", Inches(4.6), Inches(1.5),
        Inches(4.3), Inches(4.0))
pic_fit(s, f"{FIG}/fig6_heldout.png", Inches(9.0), Inches(1.5),
        Inches(4.0), Inches(4.0))
txt(s, Inches(0.4), Inches(5.6), Inches(12.6), Inches(0.5),
    "Fig. 4 confusion  |  Fig. 5 score separation  |  "
    "Fig. 6 detection on an UNSEEN attack category.", size=12, color=GREY,
    align=PP_ALIGN.CENTER)
speaker(s, "[Fig.6 is the strongest: detector still catches attack types it "
           "never trained on.]")

# ---- Slide 9: Challenges & Next steps ----
s = prs.slides.add_slide(BLANK); add_bg(s)
title_bar(s, "5. Challenges & Next Steps", "8")
bullets(s, Inches(0.6), Inches(1.3), Inches(12.0), Inches(5.0), [
    "Synthetic data is easy — wire in real BIPIA / LaMini payloads. "
    "[owner, by date]",
    "1B model + scaled gradients vs paper's 8B — test larger model. "
    "[owner, by date]",
    "Tune hidden/gradient layers on a validation split. [owner, by date]",
    "Add overlap-window segmentation + end-to-end RAG answer demo. "
    "[owner, by date]",
], size=19)
speaker(s, "[Be honest about limitations; give each next step an owner and a "
           "date before Oct 16.]")

# ---- Slide 10: Contributions + References ----
s = prs.slides.add_slide(BLANK); add_bg(s)
title_bar(s, "6. Contributions & References", "9")
bullets(s, Inches(0.6), Inches(1.3), Inches(12.0), Inches(3.0), [
    "[Member 1 — what they did]",
    "[Member 2 — what they did]",
    "[Member 3 — what they did]",
    "[Member 4 — what they did]",
], size=18)
txt(s, Inches(0.6), Inches(5.0), Inches(12), Inches(1.8),
    "References\n[1] Wen et al. Defending against Indirect Prompt Injection "
    "with LLM internal states. Findings of EMNLP 2025.\n"
    "[2] Official code: github.com/MYVAE/Instruction-detection\n"
    "[3] Model: Llama-3.2-1B-Instruct (Meta / unsloth mirror).",
    size=14, color=GREY)
speaker(s, "[Each member states their contribution in one line. REPLACE "
           "bracketed placeholders.]")

prs.save(OUT)
print(f"Saved {OUT}  ({len(prs.slides._sldIdLst)} slides)")
