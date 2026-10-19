"""Generate ORIGINAL system diagrams (matplotlib) for the report / slides.

Outputs:
  figures/fig0_pipeline.png      -- end-to-end segment-level defense flow
  figures/fig0b_architecture.png -- detector internals (features -> dual MLP)

These are drawn programmatically (vector-like shapes), NOT screenshots, so they
satisfy the "every figure must be original" requirement.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

os.makedirs("figures", exist_ok=True)
C_BG, C_SEG, C_DET, C_OUT = "#4C72B0", "#55A868", "#DD8452", "#C44E52"
plt.rcParams.update({"font.size": 10})


def box(ax, x, y, w, h, text, color, tc="white"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       linewidth=1.2, edgecolor="#333333", facecolor=color)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=tc, fontsize=9.5, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=14, linewidth=1.4, color="#333333"))


# --------------------------------------------------------------------------- #
# Figure 0: end-to-end pipeline
# --------------------------------------------------------------------------- #
def pipeline():
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 3.2); ax.axis("off")
    y, h, w = 1.1, 1.0, 1.7
    xs = [0.2, 2.3, 4.4, 6.5, 8.6]
    box(ax, xs[0], y, w, h, "Retrieved\ndocument\n(RAG context)", C_BG)
    box(ax, xs[1], y, w, h, "SPLIT\ninto segments\n(sentence / window)", C_SEG)
    box(ax, xs[2], y, w, h, "DETECT\nper segment\n(LLM internal states)", C_DET)
    box(ax, xs[3], y, w, h, "FILTER\nremove if\n p >= tau", C_OUT)
    box(ax, xs[4], y, w, h, "Clean context\n-> safe LLM\nanswer", C_BG)
    for i in range(4):
        arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)
    ax.text(5.5, 2.9, "Segment-Level Defense Against Indirect Prompt Injection",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(xs[1] + w / 2, 0.7, "sentence tokenizer", ha="center",
            fontsize=8, style="italic", color="#555")
    ax.text(xs[2] + w / 2, 0.7, "hidden + gradient features", ha="center",
            fontsize=8, style="italic", color="#555")
    fig.tight_layout(); fig.savefig("figures/fig0_pipeline.png", dpi=160,
                                    bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 0b: detector architecture
# --------------------------------------------------------------------------- #
def architecture():
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.set_xlim(0, 9.5); ax.set_ylim(0, 5.4); ax.axis("off")
    box(ax, 0.2, 3.3, 2.0, 0.9, "Segment text\n+ forced \"Sure\"", C_BG)
    # two branches
    box(ax, 3.0, 3.7, 2.4, 0.9, "Forward pass\n-> hidden state\n(layer L)", C_SEG)
    box(ax, 3.0, 2.0, 2.4, 0.9, "Backward pass\n-> attn gradients\n(q,k,v,o pooled)", C_DET)
    arrow(ax, 2.2, 3.9, 3.0, 4.15)
    arrow(ax, 2.2, 3.6, 3.0, 2.6)
    box(ax, 6.0, 3.7, 1.9, 0.9, "hidden\nbranch (BN)", C_SEG)
    box(ax, 6.0, 2.0, 1.9, 0.9, "gradient branch\nLinear+BN", C_DET)
    arrow(ax, 5.4, 4.15, 6.0, 4.15)
    arrow(ax, 5.4, 2.45, 6.0, 2.45)
    # fuse + MLP
    box(ax, 6.6, 0.5, 2.4, 0.9, "Concat -> MLP\n1024-256-64-16-2", "#8172B3")
    arrow(ax, 6.95, 3.7, 7.5, 1.4)
    arrow(ax, 6.95, 2.0, 7.7, 1.4)
    box(ax, 3.3, 0.5, 2.4, 0.9, "P(injected)\n-> remove if >= tau", C_OUT)
    arrow(ax, 6.6, 0.95, 5.7, 0.95)
    ax.text(4.75, 5.15, "Detector internals (InstructDetector, fused)",
            ha="center", fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig("figures/fig0b_architecture.png", dpi=160,
                                    bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    pipeline()
    architecture()
    print("saved figures/fig0_pipeline.png and figures/fig0b_architecture.png")
