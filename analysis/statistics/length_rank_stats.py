import ijson
import numpy as np
import pandas as pd
import torch
import clip
import warnings
from scipy.stats import pearsonr

FILENAME = "out_final_4.json"

caption_types = [
    "generated_caption_2",
    "generated_caption_3",
    "generated_caption_4",
    "chosen_coco_caption",
]

label_map = {
    "generated_caption_2": "Composite",
    "generated_caption_3": "Verbose",
    "generated_caption_4": "Image-to-Text",
    "chosen_coco_caption": "Original",
}

length_bins = list(range(25, 351, 25)) + [float("inf")]

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    warnings.warn("Running on CPU - results may differ from float16 GPU results.")

model, _ = clip.load("ViT-B/32", device = device)
model.eval()

records = []

with open(FILENAME, "r") as f:
    parser = ijson.kvitems(f, "")
    for image_id, details in parser:
        for cap_type in caption_types:
            caption = details.get(cap_type)
            if caption is None or caption.strip() == "":
                continue
            try:
                tokenized_text = clip.tokenize([caption])
                token_count = tokenized_text.shape[1]
            except Exception:
                token_count = 78
            if token_count > 77:
                continue

            clip_key = f"{cap_type}_clipscore"
            expand_key = f"{cap_type}_expanded_clipscores"
            length_key = f"{cap_type}_length"

            if clip_key not in details or expand_key not in details:
                continue
            try:
                real_val = float(details[clip_key])
            except (TypeError, ValueError):
                continue

            expanded = details[expand_key] or {}
            rank = 1 + sum(1 for v in expanded.values() if float(v["clipscore"]) > real_val)

            try:
                length_val = float(details.get(length_key, np.nan))
            except ValueError:
                length_val = np.nan

            records.append({
                "description_label": label_map[cap_type],
                "rank": rank,
                "length": length_val,
            })

df = pd.DataFrame(records)
df = df[df["length"] >= 25]

print("Pearson correlation between description length and rank: ")

results = []
for label in sorted(df["description_label"].unique()):
    subset = df[df["description_label"] == label]
    r, p = pearsonr(subset["length"], subset["rank"])
    results.append((label, r, p))
    print(f"{label}: r = {r:.3f}, p = {p:.4g}")
