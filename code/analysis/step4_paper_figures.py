#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paper-Ready Figure Generation (Step 4)

Generates publication-quality figures from the analysis pipeline outputs:

  Step 2 figures (Activity):
    - Activity class pie chart (Active / Swimming / Resting).
    - Activity timeline (color-coded state per time chunk).
    - Markov transition matrix heatmap.
    - Movement distance vs. swimming count dual-axis plot.

  Step 3 figures (Curvature):
    - Frame-level curvature scatter (colored by bending class).
    - Chunk-level curvature timeline with P5-P95 band.
    - Curvature distribution histogram + KDE.
    - Bending class pie chart (excluding Invalid frames).

All panel layouts, font sizes, DPI, labels, thresholds, and color schemes
are controlled via the GLOBAL SETTINGS dictionaries below.

Input
-----
A CSV file (or directory containing one) with columns matching the
STEP2_COLUMNS and/or curvature column templates defined below.

Output
------
PNG (or TIFF/PDF) figure files saved to OUTPUT_DIR.

No argparse. No external project dependencies.
Pure pandas / numpy / matplotlib / seaborn.
"""

import os
import re
import glob
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# =============================== GLOBAL SETTINGS =============================
# =============================================================================

# ---- IO ----
# Path to a CSV file or a directory containing CSVs.
# Typically this is the per-file tracking analysis CSV from Step 1 or
# the windowed statistics CSV from Step 2.
INPUT_SOURCE = "<INPUT_CSV_OR_DIR>"
# Example: "/path/to/results/tracking_analysis.csv"

# Directory where output figures will be saved.
OUTPUT_DIR = "<OUTPUT_FIGURE_DIR>"
# Example: "/path/to/figures"
VIDEO_BASENAME = None                          # if None -> derived from CSV name

# If INPUT_SOURCE is a directory, use this glob pattern to pick the first CSV
INPUT_GLOB_PATTERN = "*.csv"

# Run toggles
RUN_STEP2 = True     # Activity figures
RUN_STEP3 = True     # Curvature figures

# Save options
SAVE_FORMAT = "png"      # "png", "tiff", "pdf", etc.
SAVE_TRANSPARENT = False
SAVE_BBOX = "tight"

# ---- Manual title prettifier for raw metric/file names ----
MANUAL_TITLE_MAP = {
    "file_sum_Tail1_swim_count": "Tail1 swim count",
    "file_sum_MFC5_frame_dist_mm": "MFC5 moving distance (mm)",
    "circle_curvature_mm^-1": "Curvature (mm⁻¹)",
}

# ---- Step2 (Activity) column mapping (map your CSV columns here) ----
STEP2_COLUMNS = {
    # Required logical names -> your CSV columns
    "activity_class": "activity_class",           # e.g., "Active" | "Swimming" | "Resting"
    "movement_distance": "MFC5_movement_distance",# distance per chunk (mm)
    "swimming_count": "Tail1_swim_count",         # counts per chunk
    # Optional; if missing, chunk_id will be auto 0..N-1
    "chunk_id": "chunk_id",
}

# Activity states and colors (in legend/order)
STEP2_ACTIVITY_STATES = ["Active", "Swimming", "Resting"]
STEP2_ACTIVITY_COLORS = {"Active": "#E74C3C", "Swimming": "#F39C12", "Resting": "#2980B9"}

# Step2 thresholds (per hour); lines will be scaled to per-chunk if chunk=1 min
STEP2_SWIM_THRESH_PER_H = 7200       # counts / hour
STEP2_DIST_THRESH_PER_H = 1_200_000  # mm / hour
STEP2_MIN_PER_CHUNK = 1              # your pipeline’s chunk duration in minutes

# ---- Step3 (Curvature) mappings/sets ----
# Your curvature column names. Typical are like "curvature_1_mm" or "circle_curvature_mm^-1".
# You can choose ONE of the templates & comment the other.
CURVATURE_COL_TEMPLATE = "{set_name}_mm"         # ex: set_name="curvature_1" -> "curvature_1_mm"
# CURVATURE_COL_TEMPLATE = "circle_curvature_mm^-1"  # if you only have a single unified column

# If you use per-chunk stats, map their suffixes:
CURVATURE_PER_CHUNK_SUFFIX = {
    "mean": "_mean",
    "p5": "_p5",
    "p95": "_p95",
    "median": "_median",
    "max": "_max",
    "min": "_min",
    "raw_max": "_raw_max",
    "invalid_count": "_invalid_count",
}

# Curvature classes column (frame-level) if available
CURVATURE_CLASS_TEMPLATE = "{set_name}_class"   # ex: "curvature_1_class"
# If using a single class column for all sets, set e.g.: CURVATURE_CLASS_TEMPLATE = "curvature_class"

# Sets to draw in Step3 (edit to match your pipeline)
STEP3_VALID_CURVATURE_SETS = [
    {"name": "curvature_1", "parts": ["MFC1", "MFC2", "MFC3", "MFC4", "MFC5"]},
    # add more sets if you analyze multiple arcs
]

# Curvature thresholds
CURV_INVALID_TH = 0.02
CURV_EXTREME_TH = 0.01
CURV_NORMAL_TH  = 0.0033

# ---- Figure configs (journal/paper-ready) ----

STEP2_FIGCFG = {
    "enabled": True,
    "figsize": (18, 12),    # inches
    "save_dpi": 600,        # export dpi
    "screen_dpi": 150,      # canvas dpi
    "wspace": 0.22,
    "hspace": 0.32,
    "rect": [0.02, 0.04, 0.98, 0.95],  # for tight_layout(rect=...)
    "grid_alpha": 0.35,
    "legend": {"loc": "upper right", "framealpha": 0.95, "ncol": 1},
    "panels": {  # 2x2 layout (missing ones are skipped)
        "pie": True,
        "timeline": True,
        "transition": True,
        "movement": True,
    },
    "cbar": {
        "pad": 0.02,  # gap between heatmap and colorbar (addresses "colorbar gap" request)
        "shrink": 0.9,
    },
    "fs": {  # font sizes
        "suptitle": 22,
        "title": 16,
        "label": 13,
        "tick": 11,
        "legend": 11,
    },
    "titles": {
        "suptitle": "Activity Analysis — {video}",
        "pie": "Activity Distribution",
        "timeline": "Activity Timeline",
        "transition": "Activity Transition Matrix (%)",
        "movement": "Movement vs Swimming",
    },
    "labels": {
        "timeline_xlabel": "Time (min)",
        "timeline_ylabel": "State",
        "movement_xlabel": "Time (min)",
        "movement_ylabel_left": "Movement Distance (mm) — MFC5",
        "movement_ylabel_right": "Swimming Count — Tail1",
        "transition_xlabel": "Next State",
        "transition_ylabel": "Current State",
    },
}

STEP3_FIGCFG = {
    "enabled": True,
    "figsize": (20, 14),
    "save_dpi": 600,
    "screen_dpi": 150,
    "wspace": 0.22,
    "hspace": 0.32,
    "rect": [0.02, 0.04, 0.98, 0.95],
    "grid_alpha": 0.35,
    "legend": {"loc": "upper right", "framealpha": 0.95, "ncol": 1},
    "panels": {
        "scatter": True,
        "timeline": True,
        "distribution": True,
        "classpie": True,
    },
    "cbar": {
        "pad": 0.02,  # generic colorbar pad (if needed later)
        "shrink": 0.9,
    },
    "fs": {
        "suptitle": 22,
        "title": 16,
        "label": 13,
        "tick": 11,
        "legend": 11,
    },
    "titles": {
        "suptitle": "Curvature Analysis — {set_name}",
        "scatter": "{set_name}: Frame-level Curvature (colored by class)",
        "timeline": "{set_name}: Chunk-level Curvature (mean ± P5–P95)",
        "distribution_raw": "{set_name}: Curvature Distribution (frame-level)",
        "distribution_mean": "{set_name}: Curvature Distribution (chunk means)",
        "classpie": "{set_name}: Curvature Class Distribution (valid frames only)",
    },
    "labels": {
        "scatter_xlabel": "Time (min)",
        "scatter_ylabel": "Curvature (mm⁻¹)",
        "timeline_xlabel": "Time (min)",
        "timeline_ylabel": "Curvature (mm⁻¹)",
        "distribution_xlabel": "Curvature (mm⁻¹)",
        "distribution_ylabel": "Density",
    },
}

# =============================================================================
# ============================== UTILITY HELPERS ==============================
# =============================================================================

def _resolve_input_path(src: str, glob_pat: str) -> Path:
    p = Path(src)
    if p.is_file() and p.suffix.lower() == ".csv":
        return p
    if p.is_dir():
        csvs = sorted(p.glob(glob_pat))
        if not csvs:
            raise FileNotFoundError(f"No CSV files found in folder: {p} (pattern={glob_pat})")
        return csvs[0]
    raise FileNotFoundError(f"Input not found or not a CSV/folder: {src}")

def _ensure_outdir(p: str) -> Path:
    out = Path(p)
    out.mkdir(parents=True, exist_ok=True)
    return out

def _derive_video_basename(csv_path: Path) -> str:
    return csv_path.stem

def _pretty(raw: str) -> str:
    if raw in MANUAL_TITLE_MAP:
        return MANUAL_TITLE_MAP[raw]
    s = raw.replace("_mm^-1", " (mm⁻¹)").replace("_mm", " (mm)")
    s = re.sub(r"[_]+", " ", s).strip()
    return s[:1].upper() + s[1:]

def _safe_get(df: pd.DataFrame, col: str, default=None):
    return df[col] if col in df.columns else default

# =============================================================================
# ========================= STEP 2 — ACTIVITY FIGURES =========================
# =============================================================================

def step2_prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure required logical columns exist using STEP2_COLUMNS mapping.
    If 'chunk_id' missing, create 0..N-1.
    """
    out = pd.DataFrame()
    for logic, col in STEP2_COLUMNS.items():
        if col in df.columns:
            out[logic] = df[col].copy()
    # minimal checks
    for must in ["activity_class", "movement_distance", "swimming_count"]:
        if must not in out.columns:
            raise KeyError(f"[Step2] Missing mapped column for '{must}'. "
                           f"Map your CSV column in STEP2_COLUMNS.")
    if "chunk_id" not in out.columns:
        out["chunk_id"] = np.arange(len(out), dtype=int)
    # enforce categorical order for plotting
    out["activity_class"] = pd.Categorical(out["activity_class"],
                                           categories=STEP2_ACTIVITY_STATES,
                                           ordered=True)
    return out

def step2_plot(activity_df: pd.DataFrame, video_basename: str, out_dir: Path, cfg: dict):
    panels = cfg["panels"]
    fs = cfg["fs"]
    titles = cfg["titles"]
    labels = cfg["labels"]

    # figure and gridspec
    fig = plt.figure(figsize=cfg["figsize"], dpi=cfg.get("screen_dpi", 150))
    gs = fig.add_gridspec(2, 2, hspace=cfg["hspace"], wspace=cfg["wspace"])

    # 1) Pie
    if panels.get("pie", True):
        ax = fig.add_subplot(gs[0, 0])
        _step2_panel_pie(ax, activity_df, fs, titles["pie"])

    # 2) Timeline
    if panels.get("timeline", True):
        ax = fig.add_subplot(gs[0, 1])
        _step2_panel_timeline(ax, activity_df, fs, labels, titles["timeline"])

    # 3) Transition matrix
    if panels.get("transition", True):
        ax = fig.add_subplot(gs[1, 0])
        _step2_panel_transition(ax, activity_df, fs, labels, titles["transition"], cfg["cbar"])

    # 4) Movement (distance + swimming on twin axis)
    if panels.get("movement", True):
        ax = fig.add_subplot(gs[1, 1])
        _step2_panel_movement(ax, activity_df, fs, labels, titles["movement"])

    suptitle = titles["suptitle"].format(video=video_basename)
    fig.suptitle(suptitle, fontsize=fs["suptitle"], fontweight='bold')
    plt.tight_layout(rect=cfg["rect"])
    out_path = out_dir / f"{video_basename}_Step2_activity.{SAVE_FORMAT}"
    plt.savefig(out_path, dpi=cfg["save_dpi"], bbox_inches=SAVE_BBOX, transparent=SAVE_TRANSPARENT)
    plt.close(fig)
    return out_path

def _step2_panel_pie(ax, activity_df, fs, title):
    counts = activity_df["activity_class"].value_counts().reindex(STEP2_ACTIVITY_STATES, fill_value=0)
    colors = [STEP2_ACTIVITY_COLORS.get(s, "#95A5A6") for s in counts.index]
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=[f"{k} ({v})" for k, v in zip(counts.index, counts.values)],
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.78,
        textprops={'fontsize': fs["label"], 'fontweight': 'bold'}
    )
    for a in autotexts:
        a.set_fontsize(fs["legend"]); a.set_fontweight('bold'); a.set_color('white')
    ax.set_title(title, fontsize=fs["title"], fontweight='bold', pad=14)

def _step2_panel_timeline(ax, activity_df, fs, labels, title):
    tmin = activity_df["chunk_id"] * STEP2_MIN_PER_CHUNK
    colors = [STEP2_ACTIVITY_COLORS.get(s, "#95A5A6") for s in activity_df["activity_class"]]
    ax.bar(tmin, np.ones(len(activity_df)), width=0.9*STEP2_MIN_PER_CHUNK, color=colors, edgecolor='black', linewidth=0.3)
    ax.set_xlabel(labels["timeline_xlabel"], fontsize=fs["label"], fontweight='bold')
    ax.set_ylabel(labels["timeline_ylabel"], fontsize=fs["label"], fontweight='bold')
    ax.set_title(title, fontsize=fs["title"], fontweight='bold', pad=10)
    ax.set_ylim(0, 1.2); ax.set_yticks([]); ax.grid(True, alpha=0.25, axis='x')
    ax.tick_params(labelsize=fs["tick"])

def _step2_panel_transition(ax, activity_df, fs, labels, title, cbar_cfg):
    # Build simple Markov transition counts (state -> next_state)
    states = STEP2_ACTIVITY_STATES
    idx = {s: i for i, s in enumerate(states)}
    mat = np.zeros((len(states), len(states)), dtype=float)
    classes = activity_df["activity_class"].astype(str).tolist()
    for i in range(len(classes) - 1):
        a, b = classes[i], classes[i+1]
        if a in idx and b in idx:
            mat[idx[a], idx[b]] += 1.0
    # Row-normalize to percentage
    with np.errstate(divide='ignore', invalid='ignore'):
        row_sums = mat.sum(axis=1, keepdims=True)
        pct = np.where(row_sums > 0, mat / row_sums * 100.0, 0.0)

    hm = sns.heatmap(
        pct, annot=True, fmt=".1f", cmap="YlOrRd",
        xticklabels=states, yticklabels=states,
        cbar_kws={"pad": cbar_cfg.get("pad", 0.02), "shrink": cbar_cfg.get("shrink", 0.9)},
        ax=ax, annot_kws={"fontsize": fs["label"], "fontweight": "bold"}
    )
    ax.set_xlabel(labels["transition_xlabel"], fontsize=fs["label"], fontweight='bold')
    ax.set_ylabel(labels["transition_ylabel"], fontsize=fs["label"], fontweight='bold')
    ax.set_title(title, fontsize=fs["title"], fontweight='bold', pad=10)
    ax.tick_params(labelsize=fs["tick"])

def _step2_panel_movement(ax, activity_df, fs, labels, title):
    tmin = activity_df["chunk_id"] * STEP2_MIN_PER_CHUNK
    # Convert per-hour thresholds to per-chunk (assuming chunk minutes)
    swim_th = STEP2_SWIM_THRESH_PER_H * (STEP2_MIN_PER_CHUNK / 60.0)
    dist_th = STEP2_DIST_THRESH_PER_H * (STEP2_MIN_PER_CHUNK / 60.0)

    # Left: distance
    ax.plot(tmin, activity_df["movement_distance"], "-",
            color="#2E86C1", linewidth=2.2, label="Movement Distance")
    ax.set_ylabel(labels["movement_ylabel_left"], fontsize=fs["label"], fontweight='bold', color="#2E86C1")
    ax.tick_params(axis='y', labelcolor="#2E86C1", labelsize=fs["tick"])
    ax.axhline(dist_th, color="#2E86C1", linestyle="--", linewidth=1.6, alpha=0.6, label=f"Active dist th ≈ {dist_th:.0f}")
    ax.grid(True, alpha=0.25)

    # Right: swimming counts
    ax2 = ax.twinx()
    ax2.plot(tmin, activity_df["swimming_count"], "-", color="#C0392B", linewidth=2.2, label="Swimming Count")
    ax2.set_ylabel(labels["movement_ylabel_right"], fontsize=fs["label"], fontweight='bold', color="#C0392B")
    ax2.tick_params(axis='y', labelcolor="#C0392B", labelsize=fs["tick"])
    ax2.axhline(swim_th, color="#C0392B", linestyle="--", linewidth=1.6, alpha=0.6, label=f"Active swim th ≈ {swim_th:.0f}")

    # Legend merge
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=fs["legend"], framealpha=0.95)

    ax.set_xlabel(labels["movement_xlabel"], fontsize=fs["label"], fontweight='bold')
    ax.set_title(title, fontsize=fs["title"], fontweight='bold', pad=10)
    ax.tick_params(axis='x', labelsize=fs["tick"])

# =============================================================================
# ========================= STEP 3 — CURVATURE FIGURES ========================
# =============================================================================

def step3_prepare_objects(df: pd.DataFrame):
    """
    Returns:
      original_data: frame-level df (for scatter/distribution if available)
      chunked_df: chunk-level df (for timeline)
      valid_sets: the STEP3_VALID_CURVATURE_SETS (unchanged)
    Assumes df already includes the needed columns.
    """
    original_data = df.copy()
    chunked_df = df.copy()
    return original_data, chunked_df, STEP3_VALID_CURVATURE_SETS

def step3_plot_for_set(original_data: pd.DataFrame,
                       chunked_df: pd.DataFrame,
                       set_name: str,
                       out_dir: Path,
                       video_basename: str,
                       cfg: dict):
    fs = cfg["fs"]; titles = cfg["titles"]; labels = cfg["labels"]; panels = cfg["panels"]

    fig = plt.figure(figsize=cfg["figsize"], dpi=cfg.get("screen_dpi", 150))
    gs = fig.add_gridspec(3, 2, hspace=cfg["hspace"], wspace=cfg["wspace"])

    # Identify columns
    curv_col = CURVATURE_COL_TEMPLATE.format(set_name=set_name) if "{set_name}" in CURVATURE_COL_TEMPLATE else CURVATURE_COL_TEMPLATE
    class_col = CURVATURE_CLASS_TEMPLATE.format(set_name=set_name) if "{set_name}" in CURVATURE_CLASS_TEMPLATE else CURVATURE_CLASS_TEMPLATE

    # Row 1 (scatter spanning two columns)
    if panels.get("scatter", True):
        ax = fig.add_subplot(gs[0, :])
        _step3_panel_scatter(ax, original_data, curv_col, class_col, set_name, fs, titles, labels, cfg)

    # Row 2 (timeline span)
    if panels.get("timeline", True):
        ax = fig.add_subplot(gs[1, :])
        _step3_panel_timeline(ax, chunked_df, curv_col, set_name, fs, titles, labels, cfg)

    # Row 3: distribution (left) + classpie (right)
    if panels.get("distribution", True):
        ax = fig.add_subplot(gs[2, 0])
        _step3_panel_distribution(ax, original_data, chunked_df, curv_col, set_name, fs, titles, labels, cfg)
    if panels.get("classpie", True):
        ax = fig.add_subplot(gs[2, 1])
        _step3_panel_classpie(ax, original_data, class_col, set_name, fs, titles)

    suptitle = titles["suptitle"].format(set_name=set_name)
    fig.suptitle(suptitle, fontsize=fs["suptitle"], fontweight='bold')
    plt.tight_layout(rect=cfg["rect"])
    out_path = out_dir / f"{video_basename}_Step3_{set_name}.{SAVE_FORMAT}"
    plt.savefig(out_path, dpi=cfg["save_dpi"], bbox_inches=SAVE_BBOX, transparent=SAVE_TRANSPARENT)
    plt.close(fig)
    return out_path

def _step3_panel_scatter(ax, original_data, curv_col, class_col, set_name, fs, titles, labels, cfg):
    # Frame-level scatter colored by class (excluding 'Invalid')
    if curv_col in original_data.columns:
        n = len(original_data)
        tmin = np.arange(n) / (60.0 * (1.0/STEP2_MIN_PER_CHUNK))  # if STEP2_MIN_PER_CHUNK=1 → frames/60; adjust if needed
        valid = original_data[curv_col].notna()
        x = tmin[valid]
        y = original_data.loc[valid, curv_col]

        if class_col in original_data.columns:
            c = original_data.loc[valid, class_col].astype(str)
            mask = (c != "Invalid") & c.notna()
            x, y, c = x[mask], y[mask], c[mask]
            classes = [k for k in sorted(c.unique()) if k != "Invalid"]
            cmap = plt.cm.get_cmap('tab10', len(classes))
            for i, k in enumerate(classes):
                m = (c == k)
                ax.scatter(x[m], y[m], s=9, alpha=0.5, color=cmap(i), label=str(k))
        else:
            ax.scatter(x, y, s=9, alpha=0.4, color="#34495E", label="Curvature")

    ax.axhline(CURV_EXTREME_TH, color="#C0392B", linestyle="--", linewidth=1.6, alpha=0.8,
               label=f"Extreme ({CURV_EXTREME_TH:.4f})")
    ax.axhline(CURV_NORMAL_TH, color="#F39C12", linestyle="--", linewidth=1.6, alpha=0.8,
               label=f"Normal ({CURV_NORMAL_TH:.4f})")

    ax.set_xlabel(labels["scatter_xlabel"], fontsize=fs["label"], fontweight='bold')
    ax.set_ylabel(labels["scatter_ylabel"], fontsize=fs["label"], fontweight='bold')
    ax.set_title(titles["scatter"].format(set_name=set_name), fontsize=fs["title"], fontweight='bold', pad=8)
    ax.legend(fontsize=fs["legend"], loc="upper right", framealpha=0.95, ncol=2)
    ax.grid(True, alpha=cfg["grid_alpha"])
    ax.tick_params(labelsize=fs["tick"])

def _step3_panel_timeline(ax, chunked_df, curv_col, set_name, fs, titles, labels, cfg):
    mean = curv_col + CURVATURE_PER_CHUNK_SUFFIX["mean"]
    p5   = curv_col + CURVATURE_PER_CHUNK_SUFFIX["p5"]
    p95  = curv_col + CURVATURE_PER_CHUNK_SUFFIX["p95"]

    if all(col in chunked_df.columns for col in [mean, p5, p95]):
        # assume chunk_id exists or create sequential minutes
        tmin = _safe_get(chunked_df, "chunk_id", pd.Series(np.arange(len(chunked_df)))).values * STEP2_MIN_PER_CHUNK
        ax.plot(tmin, chunked_df[mean], "-", color="#2E86C1", linewidth=2.2, label="Mean curvature")
        ax.fill_between(tmin, chunked_df[p5], chunked_df[p95], color="#AED6F1", alpha=0.5, label="P5–P95")

    ax.axhline(CURV_INVALID_TH, color="#7F8C8D", linestyle="-", linewidth=1.6, alpha=0.8, label=f"Invalid ({CURV_INVALID_TH:.4f})")
    ax.axhline(CURV_EXTREME_TH, color="#C0392B", linestyle="--", linewidth=1.6, alpha=0.8, label=f"Extreme ({CURV_EXTREME_TH:.4f})")
    ax.axhline(CURV_NORMAL_TH, color="#F39C12", linestyle="--", linewidth=1.6, alpha=0.8, label=f"Normal ({CURV_NORMAL_TH:.4f})")

    ax.set_xlabel(labels["timeline_xlabel"], fontsize=fs["label"], fontweight='bold')
    ax.set_ylabel(labels["timeline_ylabel"], fontsize=fs["label"], fontweight='bold')
    ax.set_title(titles["timeline"].format(set_name=set_name), fontsize=fs["title"], fontweight='bold', pad=8)
    ax.legend(fontsize=fs["legend"], loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=cfg["grid_alpha"])
    ax.tick_params(labelsize=fs["tick"])

def _step3_panel_distribution(ax, original_data, chunked_df, curv_col, set_name, fs, titles, labels, cfg):
    # Prefer frame-level raw if available; otherwise chunk means
    plot_raw = curv_col in original_data.columns
    if plot_raw:
        data = original_data[curv_col].dropna()
        title = titles["distribution_raw"].format(set_name=set_name)
    else:
        mean = curv_col + CURVATURE_PER_CHUNK_SUFFIX["mean"]
        if mean not in chunked_df.columns:
            ax.text(0.5, 0.5, "No curvature data found", ha="center", va="center", fontsize=fs["title"])
            ax.axis("off")
            return
        data = chunked_df[mean].dropna()
        title = titles["distribution_mean"].format(set_name=set_name)

    if len(data) == 0:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", fontsize=fs["title"])
        ax.axis("off")
        return

    # histogram + KDE
    n, bins, patches = ax.hist(data, bins=50, density=True, alpha=0.7, color="#85C1E9", edgecolor="black", label="Histogram")
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        xs = np.linspace(data.min(), data.max(), 300)
        ax.plot(xs, kde(xs), "-", color="#2C3E50", linewidth=2.0, label="KDE")
    except Exception:
        pass

    ax.axvline(CURV_EXTREME_TH, color="#C0392B", linestyle="--", linewidth=1.6, alpha=0.8)
    ax.axvline(CURV_NORMAL_TH, color="#F39C12", linestyle="--", linewidth=1.6, alpha=0.8)
    ax.axvline(data.mean(), color="#27AE60", linestyle="-", linewidth=2.0, alpha=0.9, label=f"Mean {data.mean():.4f}")

    ax.set_xlabel(labels["distribution_xlabel"], fontsize=fs["label"], fontweight='bold')
    ax.set_ylabel(labels["distribution_ylabel"], fontsize=fs["label"], fontweight='bold')
    ax.set_title(title, fontsize=fs["title"], fontweight='bold', pad=8)
    ax.legend(fontsize=fs["legend"], loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=cfg["grid_alpha"])
    ax.tick_params(labelsize=fs["tick"])

def _step3_panel_classpie(ax, original_data, class_col, set_name, fs, titles):
    if class_col not in original_data.columns:
        ax.text(0.5, 0.5, "No class column", ha="center", va="center", fontsize=fs["title"]); ax.axis("off"); return
    classes = original_data[class_col].dropna().astype(str)
    if len(classes) == 0:
        ax.text(0.5, 0.5, "No valid classes", ha="center", va="center", fontsize=fs["title"]); ax.axis("off"); return

    valid = classes[classes != "Invalid"]
    if len(valid) == 0:
        ax.text(0.5, 0.5, "All frames Invalid", ha="center", va="center", fontsize=fs["title"]); ax.axis("off"); return

    vc = valid.value_counts()
    labels = [f"{k} ({v/len(valid)*100:.1f}%)" for k, v in vc.items()]
    colors = plt.cm.Set3(np.linspace(0, 1, len(vc)))
    wedges, texts, autotexts = ax.pie(vc.values, labels=labels, colors=colors,
                                      autopct="%1.1f%%", startangle=90,
                                      pctdistance=0.78,
                                      textprops={'fontsize': fs["label"], 'fontweight': 'bold'})
    for a in autotexts:
        a.set_fontsize(fs["legend"]); a.set_fontweight('bold')

    inv_count = (classes == "Invalid").sum()
    total = len(classes)
    inv_pct = (inv_count / total * 100.0) if total else 0.0
    title = titles["classpie"].format(set_name=set_name) + f"\n(Excluding {inv_count} Invalid — {inv_pct:.1f}%)"
    ax.set_title(title, fontsize=fs["title"], fontweight='bold', pad=8)

# =============================================================================
# =================================== MAIN ====================================
# =============================================================================

def main():
    csv_path = _resolve_input_path(INPUT_SOURCE, INPUT_GLOB_PATTERN)
    out_dir = _ensure_outdir(OUTPUT_DIR)
    video = VIDEO_BASENAME or _derive_video_basename(csv_path)

    df = pd.read_csv(csv_path)

    if RUN_STEP2 and STEP2_FIGCFG.get("enabled", True):
        act_df = step2_prepare_dataframe(df)
        outpath = step2_plot(act_df, video, out_dir, STEP2_FIGCFG)
        print(f"[Step2] Saved: {outpath}")

    if RUN_STEP3 and STEP3_FIGCFG.get("enabled", True):
        original_data, chunked_df, sets = step3_prepare_objects(df)
        for s in sets:
            set_name = s["name"]
            outpath = step3_plot_for_set(original_data, chunked_df, set_name, out_dir, video, STEP3_FIGCFG)
            print(f"[Step3] Saved: {outpath}")

if __name__ == "__main__":
    main()
