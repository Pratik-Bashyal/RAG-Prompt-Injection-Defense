"""Train the segment-level IPI detector (hidden / gradient / fused features)."""
import gc
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

from config import DETECTOR_PATH, FEATURE_TYPE, GRAD_DIM, HID_DIM
from data_build import build_dataset
from detector_model import Detector
import features


def main():
    print(f"[train] FEATURE_TYPE = {FEATURE_TYPE}  (grad_dim={GRAD_DIM}, hidden_dim={HID_DIM})")
    segments, labels = build_dataset(n_docs=60, seed=0)
    y = np.array(labels)

    print(f"[train] extracting features for {len(segments)} segments ...")
    X = features.extract_batch(segments)
    print(f"[train] feature matrix: {X.shape}")

    # Free the LLM before training the classifier (reclaim VRAM).
    features._model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)

    det = Detector(grad_dim=GRAD_DIM, hidden_dim=HID_DIM)
    print("[train] fitting dual-branch MLP ...")
    det.fit(Xtr, ytr, epochs=300)

    pred = det.predict_proba(Xte).argmax(axis=1)
    print("\n=== Segment-level detection report ===")
    print(classification_report(yte, pred, target_names=["benign", "injected"]))
    print("F1 (injected):", round(f1_score(yte, pred, zero_division=0), 4))

    det.save(DETECTOR_PATH)
    print(f"\n[train] saved detector -> {DETECTOR_PATH}")


if __name__ == "__main__":
    main()
