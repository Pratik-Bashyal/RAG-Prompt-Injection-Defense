"""Export the generated segment-level dataset to a human-readable CSV."""
import csv
import data_build as d

segs, labs, cats = d.build_dataset_full(90, 0)
with open("dataset.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "segment_text", "label", "label_name", "attack_category"])
    for i, (s, l, c) in enumerate(zip(segs, labs, cats), 1):
        w.writerow([i, s, l, "injected" if l == 1 else "benign", c if c else "-"])

print(f"Wrote dataset.csv: {len(segs)} rows | "
      f"{sum(labs)} injected, {len(segs)-sum(labs)} benign")
