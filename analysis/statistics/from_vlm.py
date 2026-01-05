import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from itertools import combinations

DATAFRAME_CACHE = "dataset/processed_data_cache.pkl"

df = pd.read_pickle(DATAFRAME_CACHE)

conditions = ["Image-to-Text", "k-Limited", "Concise", "200-Character-Limited"]
df_filtered = df[df["description_label"].isin(conditions)].copy()

# Within-condition: length → rank
print("\nWithin-condition: rank ~ length:\n")
for cond in conditions:
    subset = df_filtered[df_filtered['description_label'] == cond].copy()

    X_len = subset[["length"]].values
    y = subset["rank"].values
    n = len(y)

    model = LinearRegression()
    model.fit(X_len, y)

    y_pred = model.predict(X_len)
    ss_res = np.sum((y - y_pred) ** 2)
    dof = n - 2
    mse = ss_res / dof

    X_intercept = np.column_stack([np.ones(n), X_len])
    XtX_inv = np.linalg.pinv(X_intercept.T @ X_intercept)
    var_covar = mse * XtX_inv
    se = np.sqrt(np.diag(var_covar))

    coefs = np.concatenate([[model.intercept_], model.coef_])
    t_stats = coefs / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))

    beta = model.coef_[0]
    z = t_stats[1]
    p = p_values[1]

    print(f"{cond}:")
    if p < 0.05:
        sig = "p < 0.05" if p >= 0.01 else "p < 0.01" if p >= 0.001 else "p < 0.001"
        print(f"  β = {beta:.3f}, z({dof})={z:.2f}, {sig}")
    else:
        print(f"  β = {beta:.3f}, z({dof})={z:.2f}, p = {p:.4f}")

print("\nBetween-condition rank comparisons (no control):\n")

for ref, comp in combinations(conditions, 2):
    ref_ranks = df_filtered[df_filtered['description_label'] == ref]['rank'].values
    comp_ranks = df_filtered[df_filtered['description_label'] == comp]['rank'].values

    beta = comp_ranks.mean() - ref_ranks.mean()
    t_stat, p_value = stats.ttest_ind(comp_ranks, ref_ranks)
    df_resid = len(ref_ranks) + len(comp_ranks) - 2
    z = t_stat

    print(f"{comp} vs {ref}:")
    if p_value < 0.001:
        print(f"  β = {beta:.2f}, z({df_resid})={z:.2f}, p < 0.001")
    else:
        print(f"  β = {beta:.2f}, z({df_resid})={z:.2f}, p = {p_value:.4f}")

print("\nBetween-condition rank comparisons (controlling for length):\n")

for ref, comp in combinations(conditions, 2):
    df_pair = df_filtered[df_filtered['description_label'].isin([ref, comp])].copy()

    df_pair["condition_binary"] = (df_pair["description_label"] == comp).astype(int)

    y = df_pair["rank"].values
    X_full = df_pair[["condition_binary", "length"]].values
    X_length = df_pair[["length"]].values
    n = len(y)

    # Model 1: rank ~ length only
    model_length = LinearRegression()
    model_length.fit(X_length, y)
    r2_length = model_length.score(X_length, y)

    # Model 2: rank ~ condition + length
    model_full = LinearRegression()
    model_full.fit(X_full, y)
    r2_full = model_full.score(X_full, y)

    # ΔR² = incremental variance explained by condition beyond length
    delta_r2 = r2_full - r2_length

    y_pred = model_full.predict(X_full)
    ss_res = np.sum((y - y_pred) ** 2)
    dof = n - X_full.shape[1] - 1
    mse = ss_res / dof

    X_intercept = np.column_stack([np.ones(n), X_full])
    XtX_inv = np.linalg.pinv(X_intercept.T @ X_intercept)
    var_covar = mse * XtX_inv
    se = np.sqrt(np.diag(var_covar))

    coefs = np.concatenate([[model_full.intercept_], model_full.coef_])
    t_stats = coefs / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))

    beta = model_full.coef_[0]
    z = t_stats[1]
    p_value = p_values[1]

    print(f"{comp} vs {ref}:")
    if p_value < 0.001:
        print(f"  β = {beta:.2f}, z({dof})={z:.2f}, p < 0.001, ΔR² = {delta_r2:.4f}")
    else:
        print(f"  β = {beta:.2f}, z({dof})={z:.2f}, p = {p_value:.4f}, ΔR² = {delta_r2:.4f}")

print("\nBetween-condition length comparisons:\n")

for ref, comp in combinations(conditions, 2):
    ref_lengths = df_filtered[df_filtered['description_label'] == ref]['length'].values
    comp_lengths = df_filtered[df_filtered['description_label'] == comp]['length'].values

    beta = comp_lengths.mean() - ref_lengths.mean()
    t_stat, p_value = stats.ttest_ind(comp_lengths, ref_lengths)
    df_resid = len(ref_lengths) + len(comp_lengths) - 2
    z = t_stat

    print(f"{comp} vs {ref}:")
    if p_value < 0.001:
        print(f"  β = {beta:.2f}, z({df_resid})={z:.2f}, p < 0.001")
    else:
        print(f"  β = {beta:.2f}, z({df_resid})={z:.2f}, p = {p_value:.4f}")

# Within-condition: rank ~ length:

# Image-to-Text:
#   β = 0.010, z(941)=0.58, p = 0.5647
# k-Limited:
#   β = 0.092, z(4998)=2.56, p < 0.05
# Concise:
#   β = 0.015, z(2265)=1.95, p = 0.0508
# 200-Character-Limited:
#   β = 0.093, z(4829)=8.05, p < 0.001

# Between-condition rank comparisons (no control):

# k-Limited vs Image-to-Text:
#   β = 3.21, z(5941)=3.31, p < 0.001
# Concise vs Image-to-Text:
#   β = -2.34, z(3208)=-3.90, p < 0.001
# 200-Character-Limited vs Image-to-Text:
#   β = 1.03, z(5772)=1.01, p = 0.3106
# Concise vs k-Limited:
#   β = -5.55, z(7265)=-8.67, p < 0.001
# 200-Character-Limited vs k-Limited:
#   β = -2.18, z(9829)=-3.66, p < 0.001
# 200-Character-Limited vs Concise:
#   β = 3.37, z(7096)=5.03, p < 0.001

# Between-condition rank comparisons (controlling for length):

# k-Limited vs Image-to-Text:
#   β = 15.32, z(5940)=2.49, p = 0.0128, ΔR² = 0.0010
# Concise vs Image-to-Text:
#   β = -1.97, z(3207)=-3.12, p = 0.0018, ΔR² = 0.0030
# 200-Character-Limited vs Image-to-Text:
#   β = 11.79, z(5771)=7.08, p < 0.001, ΔR² = 0.0086
# Concise vs k-Limited:
#   β = -12.30, z(7264)=-3.89, p < 0.001, ΔR² = 0.0021
# 200-Character-Limited vs k-Limited:
#   β = -16.04, z(9828)=-9.34, p < 0.001, ΔR² = 0.0088
# 200-Character-Limited vs Concise:
#   β = 10.19, z(7095)=9.49, p < 0.001, ΔR² = 0.0125

# Between-condition length comparisons:

# k-Limited vs Image-to-Text:
#   β = -279.57, z(5941)=-484.10, p < 0.001
# Concise vs Image-to-Text:
#   β = -26.34, z(3208)=-18.02, p < 0.001
# 200-Character-Limited vs Image-to-Text:
#   β = -130.00, z(5772)=-99.63, p < 0.001
# Concise vs k-Limited:
#   β = 253.23, z(7265)=412.80, p < 0.001
# 200-Character-Limited vs k-Limited:
#   β = 149.57, z(9829)=269.29, p < 0.001
# 200-Character-Limited vs Concise:
#   β = -103.65, z(7096)=-106.12, p < 0.001