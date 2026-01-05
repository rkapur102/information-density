import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

rcParams['pdf.fonttype'] = 42
DATAFRAME_CACHE = "dataset/processed_data_cache.pkl"
df_raw = pd.read_pickle(DATAFRAME_CACHE)

label_color_map = {
    "Image-to-Text": "#785EF0",
    "Concise": "#44AA99",
    "200-Character-Limited": "#B6C45E",
    "k-Limited": "#648FFF"
}

conditions_to_plot = ["Image-to-Text", "Concise", "200-Character-Limited", "k-Limited"]
df = df_raw[df_raw["description_label"].isin(conditions_to_plot)].copy()
df = df[df["length"] >= 25]

length_bins = list(range(25, 426, 25)) + [float("inf")]
length_labels = [f"{start}-{end}" for start, end in zip(length_bins[:-2], length_bins[1:-1])] + ["425+"]

df["length_bin"] = pd.cut(
    df["length"],
    bins=length_bins,
    labels=length_labels,
    include_lowest=True,
).astype(str)

bin_midpoints = {lab: (low + high) / 2 for lab, low, high in zip(length_labels, length_bins[:-1], length_bins[1:])}
df["x_val"] = df["length_bin"].apply(lambda b: bin_midpoints.get(str(b), np.nan))

bin_sizes = df.groupby(["description_label", "x_val"], observed=True).size().reset_index(name="count")
max_bin_size = bin_sizes.groupby("description_label")["count"].transform("max")
bin_sizes["normalized"] = bin_sizes["count"] / max_bin_size
df = df.merge(bin_sizes[["description_label", "x_val", "normalized"]], on=["description_label", "x_val"], how="left")

summary_points = (
    df[df["x_val"].notnull()]
    .groupby("description_label")
    .agg(global_mean_length=("length", "mean"), global_mean_rank=("rank", "mean"))
    .reset_index()
)

plt.figure(figsize=(12, 10))
sns.set_style("white")
sns.lineplot(
    data=df,
    x="x_val",
    y="rank",
    hue="description_label",
    palette=label_color_map,
    errorbar="ci",
    err_style="band",
    lw=3.0,
    legend=False
)

for label in df["description_label"].unique():
    subset = df[df["description_label"] == label]
    bin_means = subset.groupby("x_val").agg(rank_mean=("rank", "mean"), normalized=("normalized", "first")).reset_index()
    sizes = 40 + 160 * bin_means["normalized"]
    plt.scatter(
        bin_means["x_val"],
        bin_means["rank_mean"],
        s=sizes,
        color=label_color_map[label],
        label=label,
        zorder=3
    )

print("Summary statistics:")
print(summary_points)

for _, row in summary_points.iterrows():
    label = row["description_label"]
    plt.scatter(
        row["global_mean_length"],
        row["global_mean_rank"],
        marker="D",
        s=150,
        color=label_color_map[label],
        edgecolor="black",
        linewidth=1.2,
        zorder=4
    )

xtick_positions = list(bin_midpoints.values())
xtick_labels = list(bin_midpoints.keys())
plt.xticks(ticks=xtick_positions, labels=xtick_labels, rotation=45, fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(25, 425)
plt.ylim(0, 50)
plt.xlabel("Binned description length (characters)", fontsize=16, labelpad=10)
plt.ylabel("Mean rank of target image", fontsize=16, labelpad=10)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(),
           title="Description type", title_fontsize=14, fontsize=13, loc="upper left")

plt.grid(False)
plt.tight_layout()
plt.savefig("vlm_generated_fig4.pdf", format="pdf")
plt.show()
