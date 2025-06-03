import ijson
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter

rcParams['pdf.fonttype'] = 42
filename = "out_final_4.json"

caption_types = [
    "generated_caption_2",
    "generated_caption_3",
    "generated_caption_4",
    "chosen_coco_caption"
]

custom_labels = ["Composite", "Verbose", "Image-to-Text", "Original"]

label_color_map = {
    "Original": "#FE6100",
    "Composite": "#DC267F",
    "Verbose": "#FFB000",
    "Image-to-Text": "#785EF0"
}

ranks = {caption: [] for caption in caption_types}

with open(filename, 'r') as f:
    parser = ijson.kvitems(f, '')
    for image_id, details in parser:
        for caption_type in caption_types:
            clipscore_key = f"{caption_type}_clipscore"
            expanded_key = f"{caption_type}_expanded_clipscores"
            if clipscore_key not in details or expanded_key not in details:
                continue
            real_clipscore = details[clipscore_key]
            expanded_clipscores = details[expanded_key]
            if (real_clipscore is None) or (real_clipscore == "") or (not expanded_clipscores):
                continue
            real_val = float(real_clipscore)
            count_higher = sum(
                1 for v in expanded_clipscores.values()
                if float(v["clipscore"]) > real_val
            )
            rank = 1 + count_higher
            ranks[caption_type].append(rank)

plot_data = []
for caption_type, rank_list in ranks.items():
    label = custom_labels[caption_types.index(caption_type)]
    total_pairs = len(rank_list)
    if total_pairs == 0:
        continue
    sorted_ranks = sorted(rank_list)
    for i, rank_val in enumerate(sorted_ranks):
        percentage = ((i + 1) / total_pairs)
        plot_data.append({
            'Description Type': label,
            'Rank': rank_val,
            'Percentage': percentage
        })

df = pd.DataFrame(plot_data)

plt.figure(figsize = (12, 8))
sns.set_style("white")

ax = sns.lineplot(
    data = df,
    x = 'Rank',
    y = 'Percentage',
    hue = 'Description Type',
    hue_order = ["Image-to-Text", "Composite", "Original", "Verbose"],
    palette = label_color_map,
    marker = None,
    linestyle = '-',
    alpha = 1.0,
    linewidth = 3.0
)

plt.xscale('log')
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.ticklabel_format(style = 'plain', axis = 'x')

plt.xlabel('(Log) rank', labelpad = 12, fontsize = 16)
plt.ylabel('Cumulative distribution\nfunction (CDF)', labelpad = 12, fontsize = 16, multialignment = 'center')

plt.tick_params(labelsize = 14)
plt.legend(title = 'Description Type', title_fontsize = 14, fontsize = 13, loc = 'lower right')

plt.xlim(0, 500)
plt.ylim(0.1, 1)

plt.grid(False)
plt.tight_layout()
plt.savefig("rank_distribution.pdf", format = 'pdf')
plt.show()

print("Summary Statistics:")
summary_stats = (
    df.groupby('Description Type')
      .agg({
          'Rank': ['count', 'mean', 'std', 'min', 'max'],
          'Percentage': 'max'
      })
      .round(2)
)
print(summary_stats)