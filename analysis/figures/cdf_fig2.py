import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter

rcParams['pdf.fonttype'] = 42
DATAFRAME_CACHE = "dataset/processed_data_cache.pkl"
df_raw = pd.read_pickle(DATAFRAME_CACHE)

label_mapping = {
    "Composite": "Composite",
    "Verbose": "Verbose",
    "Image-to-Text": "Image-to-Text",
    "COCO": "Original"
}

label_color_map = {
    "Original": "#FE6100",
    "Composite": "#DC267F",
    "Verbose": "#FFB000",
    "Image-to-Text": "#785EF0"
}

conditions_to_plot = list(label_mapping.keys())
df_filtered = df_raw[df_raw["description_label"].isin(conditions_to_plot)].copy()

plot_data = []
for label in conditions_to_plot:
    subset = df_filtered[df_filtered["description_label"] == label]
    rank_list = subset["rank"].values
    total_pairs = len(rank_list)

    if total_pairs == 0:
        continue

    sorted_ranks = sorted(rank_list)

    for i, rank_val in enumerate(sorted_ranks):
        percentage = ((i + 1) / total_pairs)
        plot_data.append({
            'Description Type': label_mapping[label],
            'Rank': rank_val,
            'Percentage': percentage
        })

df = pd.DataFrame(plot_data)

plt.figure(figsize=(12, 8))
sns.set_style("white")

ax = sns.lineplot(
    data=df,
    x='Rank',
    y='Percentage',
    hue='Description Type',
    hue_order=["Image-to-Text", "Composite", "Original", "Verbose"],  # Custom order
    palette=label_color_map,
    marker=None,
    linestyle='-',
    alpha=1.0,
    linewidth=3.0
)

plt.xscale('log')
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.ticklabel_format(style='plain', axis='x')

plt.xlabel('(Log) rank', labelpad=5, fontsize=16)
plt.ylabel('Cumulative distribution\nfunction (CDF)', labelpad=12, fontsize=16, multialignment='center')
plt.tick_params(labelsize=14)
plt.legend(title='Description Type', title_fontsize=14, fontsize=13, loc='lower right')
plt.xlim(0, 500)
plt.ylim(0.1, 1)

plt.grid(False)
plt.tight_layout()
plt.savefig("cdf_fig2.pdf", format='pdf')
plt.show()

print("\nSummary Statistics:")
summary_stats = (
    df.groupby('Description Type')
      .agg({
          'Rank': ['count', 'mean', 'std', 'min', 'max'],
          'Percentage': 'max'
      })
      .round(2)
)
print(summary_stats)
