
import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================

LOG_DIR = "logs_stage1/logs"
OUTPUT_DIR = "outputs/learning_rate_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Map logger directories to actual fold numbers
FOLD_MAP = {
    "v1_0": 1,
    "v1_1": 2,
    "v1_2": 3,
    "v1_3": 4,
    "v1_4": 5,
}

# ==========================================
# LOAD AND MERGE LEARNING RATE RECORDS
# ==========================================

records = []

csv_files = glob.glob(
    os.path.join(LOG_DIR, "**", "metrics.csv"),
    recursive=True
)

for path in csv_files:

    match = re.search(
        r"/(v1_[0-4])/version_(\d+)/metrics\.csv$",
        path.replace("\\", "/")
    )

    if not match:
        continue

    fold_dir = match.group(1)
    version = int(match.group(2))
    fold = FOLD_MAP[fold_dir]

    df = pd.read_csv(path)

    required = {"epoch", "step", "lr"}

    if not required.issubset(df.columns):
        continue

    df = df.dropna(subset=["epoch", "step", "lr"])

    if df.empty:
        continue

    df = df[["epoch", "step", "lr"]].copy()

    df["fold"] = fold
    df["version"] = version

    records.append(df)

if not records:
    raise RuntimeError("No learning-rate records found.")

data = pd.concat(records, ignore_index=True)

# ==========================================
# REMOVE DUPLICATE STEPS
# ==========================================

# Prefer the latest logger version when resumed sessions overlap.
data = (
    data.sort_values(["fold", "step", "version"])
        .drop_duplicates(["fold", "step"], keep="last")
        .sort_values(["fold", "step"])
        .reset_index(drop=True)
)

data["epoch"] = data["epoch"].astype(int)
data["step"] = data["step"].astype(int)

# ==========================================
# PLOT 1: LEARNING RATE VS EPOCH
# ==========================================

plt.figure(figsize=(11, 6))

for fold in sorted(data["fold"].unique()):

    subset = data[data["fold"] == fold]

    plt.plot(
        subset["epoch"],
        subset["lr"],
        marker=".",
        markersize=2.5,
        linewidth=1.5,
        label=f"Fold {fold}"
    )

plt.title("Learning Rate vs Epoch — Five-Fold Training")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")

plt.grid(True, alpha=0.3)
plt.legend(title="Cross-validation Fold")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "learning_rate_vs_epoch.png"),
    dpi=300
)

plt.close()

# ==========================================
# PLOT 2: LEARNING RATE VS CUMULATIVE STEP
# ==========================================

plt.figure(figsize=(12, 6))

cumulative_offset = 0

for fold in sorted(data["fold"].unique()):

    subset = data[data["fold"] == fold].copy()

    subset["cumulative_step"] = (
        subset["step"] + cumulative_offset
    )

    plt.plot(
        subset["cumulative_step"],
        subset["lr"],
        marker=".",
        markersize=2.5,
        linewidth=1.5,
        label=f"Fold {fold}"
    )

    cumulative_offset += int(subset["step"].max()) + 1

plt.title("Learning Rate vs Cumulative Training Step")
plt.xlabel("Cumulative Training Step")
plt.ylabel("Learning Rate")

plt.grid(True, alpha=0.3)
plt.legend(title="Cross-validation Fold")

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "learning_rate_vs_cumulative_step.png"),
    dpi=300
)

plt.close()

# ==========================================
# EXPORT MERGED DATA
# ==========================================

data.to_csv(
    os.path.join(OUTPUT_DIR, "merged_learning_rate_history.csv"),
    index=False
)

print("\nLearning-rate plots generated successfully!")
print(f"Total unique records: {len(data)}")
print(f"Output directory: {OUTPUT_DIR}")

for fold in sorted(data["fold"].unique()):

    subset = data[data["fold"] == fold]

    print(
        f"Fold {fold}: "
        f"{subset['epoch'].max() + 1} logged epochs, "
        f"{len(subset)} LR records, "
        f"final LR = {subset.iloc[-1]['lr']:.10f}"
    )
