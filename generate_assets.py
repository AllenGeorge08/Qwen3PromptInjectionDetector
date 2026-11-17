"""Generate training-results.png and confusion-matrix.png for the README.

Data comes from V2 notebook output (FineTunedQwenThreatClassifierV2.ipynb).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
os.makedirs(ASSET_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# V2 training data (from notebook cell 34 output)
# ---------------------------------------------------------------------------
steps = np.array([100, 200, 300, 400, 500])
train_loss = np.array([0.5729, 0.4979, 0.4625, 0.4033, 0.4319])
val_loss = np.array([0.4784, 0.3917, 0.3704, 0.3604, 0.3561])
mean_token_acc = np.array([92.26, 93.07, 93.31, 93.40, 93.45])

# ---------------------------------------------------------------------------
# Image 1: Training results (loss curves + token accuracy)
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: loss curves
ax1.plot(steps, train_loss, "o-", label="Training Loss", color="#2E86AB", linewidth=2, markersize=6)
ax1.plot(steps, val_loss, "s-", label="Validation Loss", color="#A23B72", linewidth=2, markersize=6)
ax1.set_xlabel("Training Step", fontsize=11)
ax1.set_ylabel("Loss", fontsize=11)
ax1.set_title("QLoRA Training vs Validation Loss (V2)", fontsize=12, fontweight="bold")
ax1.grid(True, alpha=0.3, linestyle="--")
ax1.legend(fontsize=10)
ax1.set_ylim(0.3, 0.65)

# Annotate final values
ax1.annotate(f"Final train: {train_loss[-1]:.4f}",
             xy=(steps[-1], train_loss[-1]), xytext=(steps[-1] - 80, train_loss[-1] + 0.03),
             fontsize=9, color="#2E86AB")
ax1.annotate(f"Final val: {val_loss[-1]:.4f}",
             xy=(steps[-1], val_loss[-1]), xytext=(steps[-1] - 80, val_loss[-1] - 0.07),
             fontsize=9, color="#A23B72")

# Right: mean token accuracy
ax2.plot(steps, mean_token_acc, "o-", color="#F18F01", linewidth=2, markersize=6)
ax2.set_xlabel("Training Step", fontsize=11)
ax2.set_ylabel("Mean Token Accuracy (%)", fontsize=11)
ax2.set_title("Mean Token Accuracy (V2)", fontsize=12, fontweight="bold")
ax2.grid(True, alpha=0.3, linestyle="--")
ax2.set_ylim(91.5, 94.0)

for x, y in zip(steps, mean_token_acc):
    ax2.annotate(f"{y:.2f}%", xy=(x, y), xytext=(0, 5),
                 textcoords="offset points", ha="center", fontsize=9, color="#F18F01")

plt.tight_layout()
plt.savefig(os.path.join(ASSET_DIR, "training-results.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Image 2: Confusion matrix
# ---------------------------------------------------------------------------
cm = np.array([[367, 23], [17, 533]])
labels = ["Benign", "Malicious"]

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels, fontsize=11)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
ax.set_ylabel("Actual Label", fontsize=12, fontweight="bold")
ax.set_title("Confusion Matrix — Held-out Test Set (V2)\n940 samples, Accuracy = 0.96",
             fontsize=12, fontweight="bold")

# Annotate cells with counts and percentages
total = cm.sum()
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        pct = count / total * 100
        ax.text(j, i, f"{count}\n({pct:.1f}%)",
                ha="center", va="center", fontsize=12, fontweight="bold",
                color="white" if count > cm.max() / 2 else "black")

plt.tight_layout()
plt.savefig(os.path.join(ASSET_DIR, "confusion-matrix.png"), dpi=150, bbox_inches="tight")
plt.close()

print("Generated:")
print(f"  {os.path.join(ASSET_DIR, 'training-results.png')}")
print(f"  {os.path.join(ASSET_DIR, 'confusion-matrix.png')}")
