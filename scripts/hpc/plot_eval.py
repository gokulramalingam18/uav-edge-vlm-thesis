import csv, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "/home/s3335739/uav_finetune/eval_results.csv"
OUT = "/home/s3335739/uav_finetune/eval_distribution.png"

gt, pred, err = [], [], []
with open(CSV) as f:
    r = csv.DictReader(f)
    for row in r:
        if row["abs_error"] not in ("NA", "") and row["abs_error"].isdigit():
            gt.append(int(row["gt_total"]))
            pred.append(int(row["pred_total"]))
            err.append(int(row["abs_error"]))

n = len(err)
err_sorted = sorted(err)
mean = sum(err)/n
median = err_sorted[n//2]
p90 = err_sorted[int(n*0.9)]
mx = err_sorted[-1]

# Trimmed (robust) MAE: drop worst 5%
keep = err_sorted[:int(n*0.95)]
trimmed_mae = sum(keep)/len(keep)

print("n=%d mean=%.2f median=%d p90=%d max=%d trimmed95_MAE=%.2f"
      % (n, mean, median, p90, mx, trimmed_mae))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left: error histogram, capped at 60 for readability with an overflow note
capped = [min(e, 60) for e in err]
ax1.hist(capped, bins=30, color="#028090", edgecolor="white")
ax1.axvline(median, color="#C1121F", linestyle="--", linewidth=2, label="median = %d" % median)
ax1.axvline(mean, color="#1E293B", linestyle="-", linewidth=2, label="mean = %.1f" % mean)
ax1.set_xlabel("Absolute error in total object count (capped at 60)")
ax1.set_ylabel("Number of images")
ax1.set_title("Distribution of counting error (n=%d)" % n)
ax1.legend()
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Right: predicted vs ground truth scatter with y=x line
ax2.scatter(gt, pred, alpha=0.35, s=18, color="#028090", edgecolors="none")
lim = max(max(gt), max(pred)) + 5
ax2.plot([0, lim], [0, lim], color="#C1121F", linestyle="--", linewidth=1.5, label="perfect (y = x)")
ax2.set_xlim(0, lim); ax2.set_ylim(0, lim)
ax2.set_xlabel("Ground-truth total objects")
ax2.set_ylabel("Predicted total objects")
ax2.set_title("Predicted vs. ground-truth count")
ax2.legend()
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print("SAVED", OUT)
