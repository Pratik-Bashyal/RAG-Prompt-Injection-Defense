"""Central configuration for the segment-level IPI defense demo.

Implements the InstructDetector method (hidden states + self-attention gradients,
fused) from Wen et al., EMNLP 2025 -- adapted to a laptop-scale GPU and extended
to SEGMENT level.
"""

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# PAPER model family (gated meta-llama repos need approval; unsloth mirror is the
# SAME weights, ungated, so the demo runs today):
#   "unsloth/Llama-3.2-1B-Instruct"  -> 16 layers, hidden 2048  (HIDDEN_DIM = 2048)
#   "unsloth/Llama-3.2-3B-Instruct"  -> 28 layers, hidden 3072  (HIDDEN_DIM = 3072)
# Open fallback (no login): "Qwen/Qwen2.5-0.5B-Instruct" (hidden 896)
MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"
HIDDEN_DIM = 2048   # last-dim of the model's hidden states (must match MODEL_NAME)

# ---------------------------------------------------------------------------
# Feature type (this IS the paper's "feature type" ablation)
#   "hidden"   -> last-token hidden state only
#   "gradient" -> max-pooled self-attention gradients only
#   "fused"    -> both concatenated (the full InstructDetector method)
# ---------------------------------------------------------------------------
FEATURE_TYPE = "fused"

# Layers to read features from (re-selected per model on validation data).
# Paper's 8B best layers were hidden=14, gradient=13 out of 33. For Llama-3.2-1B
# (16 layers) we use middle layers; sweep these in the layer ablation.
HIDDEN_LAYER = 8    # index into hidden_states (0 = embeddings, so 1..n_layers)
GRAD_LAYER = 7      # index into the decoder layers (0..n_layers-1)

# Forced response paired with each segment to compute gradients ("Sure" trick).
FORCED_RESPONSE = "Sure"

# ---------------------------------------------------------------------------
# Gradient pooling target shapes (rows, cols) per attention projection.
# Paper uses q/o=(400,400), k/v=(100,400) -> 400,000 dims/layer, needing an A100.
# We scale DOWN for an 8 GB laptop GPU (resource-aware). Same recipe, smaller map.
#   per-layer grad dim = (128+32+32+128) * 128 = 40,960
# ---------------------------------------------------------------------------
GRAD_POOL = {
    "q_proj.weight": (128, 128),
    "k_proj.weight": (32, 128),
    "v_proj.weight": (32, 128),
    "o_proj.weight": (128, 128),
}
_cols = list(GRAD_POOL.values())[0][1]
GRAD_DIM_PER_LAYER = sum(r for r, _ in GRAD_POOL.values()) * _cols

# Effective feature dimensions given FEATURE_TYPE (used by the classifier).
GRAD_DIM = GRAD_DIM_PER_LAYER if FEATURE_TYPE in ("gradient", "fused") else 0
HID_DIM = HIDDEN_DIM if FEATURE_TYPE in ("hidden", "fused") else 0

# ---------------------------------------------------------------------------
# Detection / segmentation knobs
# ---------------------------------------------------------------------------
# Removal threshold tau. Lower = more security (removes more), higher = keeps more.
TAU = 0.5

# Segmentation: "sentence" | "fixed" | "overlap"
SEG_STRATEGY = "sentence"
WINDOW_SIZE = 2
WINDOW_STRIDE = 1

DETECTOR_PATH = "data/detector.pt"
DEVICE = "cuda"  # auto-falls back to cpu in code
