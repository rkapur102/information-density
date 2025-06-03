import warnings
import ijson
import numpy as np
from scipy.stats import ttest_ind, levene
import torch
import clip

JSON_PATH = "OF4_gpt4o_updated.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    warnings.warn("Running on CPU - results may differ from float16 GPU results.")

model, _ = clip.load("ViT-B/32", device = device)
model.eval()

def caption_ok(caption: str) -> bool:
    if caption is None or caption.strip() == "":
        return False
    try:
        tok_cnt = clip.tokenize([caption]).shape[1]
    except Exception:
        tok_cnt = 78
    return tok_cnt <= 77

def rank(true_score, expanded_scores) -> int:
    true = float(true_score)
    others = [float(d["clipscore"]) for d in expanded_scores.values()]
    higher = sum(s > true for s in others)
    return higher + 1

ranks_200, ranks_unrestricted = [], []

with open(JSON_PATH, "rb") as f:
    for _, rec in ijson.kvitems(f, ""):
        cap_200 = rec.get("gpt4o_200char", "")
        if caption_ok(cap_200):
            try:
                ranks_200.append(
                    rank(
                        rec["gpt4o_200char_clipscore"],
                        rec["gpt4o_200char_expanded_clipscores"],
                    )
                )
            except KeyError:
                pass

        cap_unrestricted = rec.get("generated_caption_4", "")
        if caption_ok(cap_unrestricted):
            try:
                ranks_unrestricted.append(
                    rank(
                        rec["generated_caption_4_clipscore"],
                        rec["generated_caption_4_expanded_clipscores"],
                    )
                )
            except KeyError:
                pass

ranks_200 = np.asarray(ranks_200, dtype = float)
ranks_unrestricted = np.asarray(ranks_unrestricted, dtype = float)

mean_200, mean_unrestricted = ranks_200.mean(), ranks_unrestricted.mean()
var_200, var_unrestricted = ranks_200.var(ddof = 1), ranks_unrestricted.var(ddof = 1)
ratio = max(var_200, var_unrestricted) / min(var_200, var_unrestricted)

print(f"Samples retained - 200-char: {len(ranks_200):,}")
print(f"Samples retained - unrestricted: {len(ranks_unrestricted):,}")
print(f"Mean rank - 200-char: {mean_200:.3f}")
print(f"Mean rank - unrestricted: {mean_unrestricted:.3f}")
print(f"Variance (ddof=1) - 200-char: {var_200:.3f}")
print(f"Variance (ddof=1) - unrestricted: {var_unrestricted:.3f}")
print(f"Variance ratio (larger/smaller): {ratio:.2f}")

W_lev, p_lev = levene(ranks_200, ranks_unrestricted)
print("\nLevene's test for equal variances")
print(f"W = {W_lev:.3f}, p = {p_lev:.3e}")

t_student, p_student = ttest_ind(ranks_200, ranks_unrestricted, equal_var = True)
print("\nStudent's t-test (equal-variance)")
print(f"t = {t_student:.3f}, p = {p_student:.3e}")

t_welch, p_welch = ttest_ind(ranks_200, ranks_unrestricted, equal_var = False)
print("\nWelch's t-test (unequal-variance)")
print(f"t = {t_welch:.3f}, p = {p_welch:.3e}")
