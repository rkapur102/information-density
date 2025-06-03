import ijson
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import torch
import clip
import warnings

rcParams['pdf.fonttype'] = 42
plt.figure(figsize = (12, 10))
sns.set_style("white")

FILENAME = "out_final_4.json"

caption_types = [
    "generated_caption_2",
    "generated_caption_3",
    "generated_caption_4",
    "chosen_coco_caption"
]

label_map = {
    "generated_caption_2": "Composite",
    "generated_caption_3": "Verbose",
    "generated_caption_4": "Image-to-Text",
    "chosen_coco_caption": "Original"
}

label_color_map = {
    "Original": "#FE6100",
    "Composite": "#DC267F",
    "Verbose": "#FFB000",
    "Image-to-Text": "#785EF0"
}

length_bins = list(range(25, 351, 25)) + [float("inf")]
length_labels = [f"{start}-{end}" for start, end in zip(length_bins[:-2], length_bins[1:-1])] + ["350+"]

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
                "length": length_val
            })

df = pd.DataFrame(records)
df = df[df["length"] >= 25]
df["length_bin"] = pd.cut(df["length"], bins = length_bins, labels = length_labels, include_lowest = True).astype(str)
bin_midpoints = {label: (low + high) / 2 for label, low, high in zip(length_labels, length_bins[:-1], length_bins[1:])}
df["x_val"] = df["length_bin"].apply(lambda b: bin_midpoints.get(str(b), np.nan))

bin_sizes = df.groupby(["description_label", "x_val"], observed = True).size().reset_index(name = "count")
max_bin_size = bin_sizes.groupby("description_label")["count"].transform("max")
bin_sizes["normalized"] = bin_sizes["count"] / max_bin_size
df = df.merge(bin_sizes[["description_label", "x_val", "normalized"]], on = ["description_label", "x_val"], how = "left")

summary_points = (
    df[df["x_val"].notnull()]
    .groupby("description_label")
    .agg(global_mean_length = ("length", "mean"), global_mean_rank = ("rank", "mean"))
    .reset_index()
)

sns.lineplot(
    data = df,
    x = "x_val",
    y = "rank",
    hue = "description_label",
    palette = label_color_map,
    errorbar = "ci",
    err_style = "band",
    lw = 2,
    legend = False
)

for label in df["description_label"].unique():
    subset = df[df["description_label"] == label]
    bin_means = subset.groupby("x_val").agg(rank_mean = ("rank", "mean"), normalized = ("normalized", "first")).reset_index()
    sizes = 40 + 160 * bin_means["normalized"]
    plt.scatter(bin_means["x_val"], bin_means["rank_mean"], s = sizes, color = label_color_map[label], label = label, zorder = 3)

for _, row in summary_points.iterrows():
    label = row["description_label"]
    plt.scatter(
        row["global_mean_length"],
        row["global_mean_rank"],
        marker = "D",
        s = 150,
        color = label_color_map[label],
        edgecolor = "black",
        linewidth = 1.2,
        zorder = 4
    )

xtick_positions = list(bin_midpoints.values())
xtick_labels = list(bin_midpoints.keys())
plt.xticks(ticks = xtick_positions, labels = xtick_labels, rotation = 45, fontsize = 14)
plt.yticks(fontsize = 14)
plt.xlim(25, 350)
plt.ylim(0, 70)
plt.xlabel("Binned description length (characters)", fontsize = 16, labelpad = 10)
plt.ylabel("Mean rank of target image", fontsize = 16, labelpad = 5)

legend_order = ["Image-to-Text", "Composite", "Original", "Verbose"]
handles, labels = plt.gca().get_legend_handles_labels()
label_to_handle = dict(zip(labels, handles))
ordered_handles = [label_to_handle[label] for label in legend_order if label in label_to_handle]

plt.legend(
    ordered_handles,
    legend_order,
    title = "Description type",
    title_fontsize = 12,
    fontsize = 11,
    loc = "upper right"
)

plt.tight_layout()
plt.savefig("rank_vs_length.pdf", format = "pdf")
plt.show()
