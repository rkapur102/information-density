import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

rcParams['pdf.fonttype'] = 42
DATAFRAME_CACHE = "dataset/processed_data_cache.pkl"
df_raw = pd.read_pickle(DATAFRAME_CACHE)
df_raw["description_label"] = df_raw["description_label"].replace({"COCO": "Original"})

conditions_to_plot = ["Image-to-Text", "Composite", "Verbose", "Original"]
df = df_raw[df_raw["description_label"].isin(conditions_to_plot)].copy()

label_color_map = {
    "Image-to-Text": "#785EF0",
    "Original": "#FE6100",
    "Composite": "#DC267F",
    "Verbose": "#FFB000",
}

length_bins = list(range(25, 351, 25))
length_labels = [f"{start}-{end}" for start, end in zip(length_bins[:-1], length_bins[1:])]
sns.set_style("white")
plt.figure(figsize=(14, 10))

df_length = df[(df["length"] >= 25) & (df["length"] <= 350)].copy()
df_length["length_bin"] = pd.cut(df_length["length"], bins=length_bins, labels=length_labels, include_lowest=True).astype(str)
bin_midpoints = {label: (low + high) / 2 for label, low, high in zip(length_labels, length_bins[:-1], length_bins[1:])}
df_length["x_val"] = df_length["length_bin"].apply(lambda b: bin_midpoints.get(str(b), np.nan))

bin_sizes = df_length.groupby(["description_label", "x_val"], observed=True).size().reset_index(name="count")
max_bin_size = bin_sizes.groupby("description_label")["count"].transform("max")
bin_sizes["normalized"] = bin_sizes["count"] / max_bin_size
df_length = df_length.merge(bin_sizes[["description_label", "x_val", "normalized"]], on=["description_label", "x_val"], how="left")

summary_points = (
    df.groupby("description_label")
    .agg(global_mean_length=("length", "mean"), global_mean_rank=("rank", "mean"))
    .reset_index()
)

sns.lineplot(
    data=df_length,
    x="x_val",
    y="rank",
    hue="description_label",
    palette=label_color_map,
    errorbar="ci",
    err_style="band",
    lw=2,
    legend=False
)

for label in df_length["description_label"].unique():
    subset = df_length[df_length["description_label"] == label]
    bin_means = subset.groupby("x_val").agg(rank_mean=("rank", "mean"), normalized=("normalized", "first")).reset_index()
    sizes = 40 + 160 * bin_means["normalized"]
    plt.scatter(bin_means["x_val"], bin_means["rank_mean"], s=sizes, color=label_color_map[label], label=label, zorder=3)

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
plt.xlim(25, 350)
plt.ylim(0, 70)
plt.xlabel("Binned description length (characters)", fontsize=16, labelpad=10)
plt.ylabel("Mean rank of target image", fontsize=16, labelpad=5)

legend_order = ["Image-to-Text", "Composite", "Original", "Verbose"]
handles, labels = plt.gca().get_legend_handles_labels()
label_to_handle = dict(zip(labels, handles))
ordered_handles = [label_to_handle[label] for label in legend_order if label in label_to_handle]

plt.legend(
    ordered_handles,
    [label for label in legend_order if label in label_to_handle],
    title="Description type",
    title_fontsize=14,
    fontsize=13,
    loc="upper right"
)

plt.tight_layout()
plt.savefig("binned_rank_length_fig3.pdf", format="pdf")
plt.show()

print("Summary statistics:")
for label in legend_order:
    if label in summary_points["description_label"].values:
        row = summary_points[summary_points["description_label"] == label].iloc[0]
        print(f"{label}: Mean rank = {row['global_mean_rank']:.2f}, Mean length = {row['global_mean_length']:.2f}")
