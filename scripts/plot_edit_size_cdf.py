#!/usr/bin/env python3
"""
plot_edit_size_cdf.py
=====================
Read exp_out/counterfactual_pairs.jsonl and plot the cumulative distribution
function (CDF) of token-edit sizes for **valid** counterfactual pairs.

Outputs:
  - exp_out/tables/edit_size_cdf.png
  - exp_out/tables/edit_size_cdf.pdf
"""

from __future__ import annotations

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
JSONL = ROOT / "exp_out" / "counterfactual_pairs.jsonl"
OUT_DIR = ROOT / "exp_out" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load data ────────────────────────────────────────────────────────────────
pairs = []
with open(JSONL, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            pairs.append(json.loads(line))

valid_pairs = [p for p in pairs if p.get("valid_cf")]

# Extract token_edits per bucket and overall
bucket_edits: dict[str, list[int]] = {}
all_edits: list[int] = []

for p in valid_pairs:
    tok = p.get("edit_stats", {}).get("token_edits")
    if tok is None:
        continue
    all_edits.append(tok)
    b = p.get("bucket", "?")
    bucket_edits.setdefault(b, []).append(tok)

print(f"Total valid pairs with token_edits: {len(all_edits)}")
for b in sorted(bucket_edits):
    print(f"  Bucket {b}: {len(bucket_edits[b])} pairs")

# ── plot helpers ─────────────────────────────────────────────────────────────
BUCKET_LABELS = {"A": "A (intrinsic→no)", "B": "B (extrinsic→no)",
                 "C": "C (no→extrinsic)", "D": "D (no→intrinsic)"}
BUCKET_COLORS = {"A": "#e74c3c", "B": "#3498db", "C": "#2ecc71", "D": "#9b59b6"}


def ecdf(data: list[int]):
    """Return sorted x values and their CDF y values (0-1)."""
    xs = np.sort(data)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


# ── Figure 1: Overall CDF + per-bucket CDF ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

# --- Left panel: overall CDF ---
ax = axes[0]
xs, ys = ecdf(all_edits)
ax.step(xs, ys * 100, where="post", color="#2c3e50", linewidth=2, label="All valid")
ax.set_xlabel("Token edit distance", fontsize=12)
ax.set_ylabel("Cumulative %", fontsize=12)
ax.set_title("Overall Edit-Size CDF", fontsize=13, fontweight="bold")
ax.set_xlim(0, max(all_edits) + 1)
ax.set_ylim(0, 102)
ax.axhline(50, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
ax.axhline(90, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)

# annotate key percentiles
median_val = int(np.median(all_edits))
p90_val = int(np.percentile(all_edits, 90))
ax.annotate(f"Median = {median_val}", xy=(median_val, 50), fontsize=9,
            xytext=(median_val + 3, 45), arrowprops=dict(arrowstyle="->", color="grey"),
            color="grey")
ax.annotate(f"P90 = {p90_val}", xy=(p90_val, 90), fontsize=9,
            xytext=(p90_val + 3, 85), arrowprops=dict(arrowstyle="->", color="grey"),
            color="grey")

ax.legend(fontsize=10, loc="lower right")
ax.grid(axis="y", alpha=0.3)

# --- Right panel: per-bucket CDF ---
ax = axes[1]
for b in sorted(bucket_edits):
    if len(bucket_edits[b]) < 2:
        continue
    xs, ys = ecdf(bucket_edits[b])
    label = BUCKET_LABELS.get(b, f"Bucket {b}")
    color = BUCKET_COLORS.get(b, None)
    ax.step(xs, ys * 100, where="post", linewidth=1.8, label=label, color=color)

ax.set_xlabel("Token edit distance", fontsize=12)
ax.set_ylabel("Cumulative %", fontsize=12)
ax.set_title("Edit-Size CDF by Bucket", fontsize=13, fontweight="bold")
ax.set_xlim(0, max(all_edits) + 1)
ax.set_ylim(0, 102)
ax.axhline(50, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
ax.axhline(90, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
ax.legend(fontsize=9, loc="lower right")
ax.grid(axis="y", alpha=0.3)

# ── save ─────────────────────────────────────────────────────────────────────
for ext in ("png", "pdf"):
    out_path = OUT_DIR / f"edit_size_cdf.{ext}"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved → {out_path}")

plt.show()
print("Done.")
