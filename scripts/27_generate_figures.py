#!/usr/bin/env python3
"""
ROADMAP v2 Day 7 — Generate all thesis figures.

Outputs (in paper/figures/):
  roc_curves.pdf        — ROC curves for all 12 systems (one panel, colour-coded by family)
  roc_curves.png        — Same as PNG for quick review
  confusion_matrices.pdf — 3x4 grid of confusion matrices (one per selected system)
  per_platform_bar.pdf  — Per-platform ROC-AUC bar chart (podcast_rss vs youtube)

Prerequisites:
  data/annotations.csv        — 300-item GT
  data/benchmark_predictions/ — completed .json files
  data/benchmark_per_platform.csv — from script 26 (optional; regenerated inline if absent)
  matplotlib, scikit-learn, numpy, pandas
"""

from __future__ import annotations

import csv
import json
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive; no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix

sys.path.append(str(Path(__file__).resolve().parents[1]))

PREDS_DIR = Path("data/benchmark_predictions")
ANNOTATIONS_CSV = Path("data/annotations.csv")
QUEUE_DB = Path("data/queue.db")
PER_PLATFORM_CSV = Path("data/benchmark_per_platform.csv")
FIGURES_DIR = Path("paper/figures")

GT_LABEL_MAP = {"lebanese": 1, "mostly_lebanese": 1, "not_lebanese": 0}

# Display labels (shortened for figures)
SYSTEM_LABELS: dict[str, str] = {
    "v1_text_only":                    "V1 (lex+embed)",
    "v1_lex_only":                     "V1 lex-only",
    "v1_embedding_only":               "V1 embed-only",
    "v2_frozen_mlp":                   "V2 MLP",
    "v2_balanced":                     "V2-balanced",
    "hybrid_v1v2_mlp":                 "V1+V2 hybrid",
    "v25_finetuned":                   "V2.5 XLS-R",
    "whisper_lid_arabic_prob":         "Whisper LID",
    "badr_mms_300m_levantine":         "Badr MMS-300m",
    "voxlect_mms_lid256_levantine":    "Voxlect MMS-256",
    "elyadata_whisper_adi20_leb":      "Elyadata ADI-20",
    "marbertv2_lev":                   "MARBERTv2",
    "groq_llama31_8b_zeroshot":        "Llama-3.1-8B 0-shot",
    "groq_llama31_8b_3shot":           "Llama-3.1-8B 3-shot",
    "v2_same_source":                  "V2 same-source",
}

# Colour family grouping
FAMILY_COLORS: dict[str, str] = {
    "v1_text_only":                 "#1f77b4",
    "v1_lex_only":                  "#aec7e8",
    "v1_embedding_only":            "#6baed6",
    "v2_frozen_mlp":                "#d62728",
    "v2_balanced":                  "#ff9896",
    "hybrid_v1v2_mlp":              "#9467bd",
    "v25_finetuned":                "#8c564b",
    "whisper_lid_arabic_prob":      "#bcbd22",
    "badr_mms_300m_levantine":      "#2ca02c",
    "voxlect_mms_lid256_levantine": "#98df8a",
    "elyadata_whisper_adi20_leb":   "#ff7f0e",
    "marbertv2_lev":                "#17becf",
    "groq_llama31_8b_zeroshot":     "#e377c2",
    "groq_llama31_8b_3shot":        "#f7b6d2",
    "v2_same_source":               "#ffbb78",
}

LINESTYLES: dict[str, str] = {
    "v1_text_only":                 "-",
    "v1_lex_only":                  "--",
    "v1_embedding_only":            "-.",
    "v2_frozen_mlp":                "-",
    "v2_balanced":                  "--",
    "hybrid_v1v2_mlp":              "-.",
    "v25_finetuned":                "-",
    "whisper_lid_arabic_prob":      ":",
    "badr_mms_300m_levantine":      "-",
    "voxlect_mms_lid256_levantine": "--",
    "elyadata_whisper_adi20_leb":   "-.",
    "marbertv2_lev":                "-",
    "groq_llama31_8b_zeroshot":     "-",
    "groq_llama31_8b_3shot":        "--",
    "v2_same_source":               ":",
}

# Systems to show in the confusion matrix grid (most interesting)
CM_SYSTEMS = [
    "v1_text_only",
    "elyadata_whisper_adi20_leb",
    "marbertv2_lev",
    "v2_balanced",
    "badr_mms_300m_levantine",
    "v25_finetuned",
]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_gt() -> list[dict]:
    annotations: dict[int, dict] = {}
    with ANNOTATIONS_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iid = int(row["item_id"])
            label = row.get("ground_truth", "").strip().lower()
            if label not in GT_LABEL_MAP:
                continue
            annotations[iid] = {"item_id": iid, "y": GT_LABEL_MAP[label]}

    if QUEUE_DB.exists():
        conn = sqlite3.connect(QUEUE_DB)
        cur = conn.cursor()
        ids_str = ",".join(str(i) for i in annotations.keys())
        for iid, platform in cur.execute(
            f"SELECT id, platform FROM queue WHERE id IN ({ids_str})"
        ).fetchall():
            if iid in annotations:
                annotations[iid]["platform"] = platform
        conn.close()
    return list(annotations.values())


def load_predictions(system_name: str) -> dict[int, float]:
    path = PREDS_DIR / f"{system_name}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "predictions" in data:
        return {int(p["item_id"]): float(p["prob"]) for p in data["predictions"]}
    return {int(k): float(v) for k, v in data.items()}


def get_system_names() -> list[str]:
    return sorted(
        p.stem for p in PREDS_DIR.glob("*.json")
        if not p.stem.endswith(".partial")
    )


# ---------------------------------------------------------------------------
# Figure 1 — ROC curves
# ---------------------------------------------------------------------------

def fig_roc_curves(gt_items: list[dict]) -> None:
    system_names = get_system_names()

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance (AUC=0.500)")

    for sname in system_names:
        preds = load_predictions(sname)
        if not preds:
            continue
        matched = [(it["y"], preds[it["item_id"]]) for it in gt_items if it["item_id"] in preds]
        if len(matched) < 10:
            continue
        y_true = [m[0] for m in matched]
        y_score = [m[1] for m in matched]
        if len(set(y_true)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        label_name = SYSTEM_LABELS.get(sname, sname)
        color = FAMILY_COLORS.get(sname, "#333333")
        ls = LINESTYLES.get(sname, "-")
        ax.plot(fpr, tpr, color=color, lw=1.6, ls=ls,
                label=f"{label_name} ({roc_auc:.3f})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Lebanese DID Benchmark (all 14 systems)", fontsize=13)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.grid(True, alpha=0.3)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curves.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)
    print(f"[saved] {FIGURES_DIR}/roc_curves.pdf/.png", flush=True)


# ---------------------------------------------------------------------------
# Figure 2 — Confusion matrix grid
# ---------------------------------------------------------------------------

def fig_confusion_matrices(gt_items: list[dict]) -> None:
    systems_to_plot = [s for s in CM_SYSTEMS if (PREDS_DIR / f"{s}.json").exists()]
    n = len(systems_to_plot)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.2))
    axes = np.array(axes).flatten()

    for i, sname in enumerate(systems_to_plot):
        preds = load_predictions(sname)
        matched = [(it["y"], preds[it["item_id"]]) for it in gt_items if it["item_id"] in preds]
        y_true = [m[0] for m in matched]
        y_pred = [1 if m[1] >= 0.5 else 0 for m in matched]

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        ax = axes[i]
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, shrink=0.8)

        classes = ["Not-LB", "Lebanese"]
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, fontsize=9)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontsize=9)

        thresh = cm.max() / 2.0
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(cm[row, col]),
                        ha="center", va="center", fontsize=11,
                        color="white" if cm[row, col] > thresh else "black")

        ax.set_title(SYSTEM_LABELS.get(sname, sname), fontsize=10)
        ax.set_ylabel("True label", fontsize=8)
        ax.set_xlabel("Predicted label", fontsize=8)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Confusion Matrices @ τ=0.50 (selected systems)", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrices.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {FIGURES_DIR}/confusion_matrices.pdf/.png", flush=True)


# ---------------------------------------------------------------------------
# Figure 3 — Per-platform ROC-AUC bar chart
# ---------------------------------------------------------------------------

def fig_per_platform_bar() -> None:
    if not PER_PLATFORM_CSV.exists():
        print("[skip] per_platform_bar: data/benchmark_per_platform.csv not found — run script 26 first",
              flush=True)
        return

    # Read CSV
    data: dict[tuple, float] = {}
    ns: dict[tuple, int] = {}
    with PER_PLATFORM_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["system"], row["platform"])
            try:
                data[key] = float(row["roc_auc"])
                ns[key] = int(row["n"])
            except (ValueError, KeyError):
                pass

    systems = get_system_names()
    # Only show systems that have both podcast_rss and youtube
    plat_a, plat_b = "podcast_rss", "youtube"
    systems_shown = [
        s for s in systems
        if (s, plat_a) in data and (s, plat_b) in data
    ]

    if not systems_shown:
        print("[skip] per_platform_bar: no systems with both podcast_rss and youtube data",
              flush=True)
        return

    x = np.arange(len(systems_shown))
    width = 0.35
    auc_podcast = [data[(s, plat_a)] for s in systems_shown]
    auc_youtube = [data[(s, plat_b)] for s in systems_shown]
    labels_x = [SYSTEM_LABELS.get(s, s) for s in systems_shown]

    fig, ax = plt.subplots(figsize=(max(10, len(systems_shown) * 1.1), 5))
    bars_a = ax.bar(x - width / 2, auc_podcast, width, label="Podcast RSS", color="#1f77b4", alpha=0.85)
    bars_b = ax.bar(x + width / 2, auc_youtube, width, label="YouTube",     color="#ff7f0e", alpha=0.85)

    ax.axhline(0.5, color="gray", linestyle="--", lw=0.8, label="Chance")
    ax.set_ylabel("ROC-AUC", fontsize=12)
    ax.set_title("Per-platform ROC-AUC: Podcast RSS vs. YouTube", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x, rotation=30, ha="right", fontsize=8)
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "per_platform_bar.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "per_platform_bar.png", dpi=150)
    plt.close(fig)
    print(f"[saved] {FIGURES_DIR}/per_platform_bar.pdf/.png", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not ANNOTATIONS_CSV.exists():
        print(f"[error] GT not found: {ANNOTATIONS_CSV}", flush=True)
        sys.exit(1)

    print("[load] GT ...", flush=True)
    gt_items = load_gt()
    print(f"  {len(gt_items)} items", flush=True)

    print("[fig 1] ROC curves ...", flush=True)
    fig_roc_curves(gt_items)

    print("[fig 2] Confusion matrices ...", flush=True)
    fig_confusion_matrices(gt_items)

    print("[fig 3] Per-platform bar chart ...", flush=True)
    fig_per_platform_bar()

    print("\n[done] All figures saved to paper/figures/", flush=True)


if __name__ == "__main__":
    main()
