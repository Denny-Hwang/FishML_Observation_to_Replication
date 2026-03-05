#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seasonal and Diel Pattern Analysis (Step 3)

Visualizes temporal activity patterns from the integrated summary CSV
produced by Step 2 (windowed_statistics.py).

Generates four categories of plots:
  1. Diel curves    — 24-hour activity profiles with Day/Night shading.
  2. Heatmaps       — Date x Hour activity intensity grids.
  3. Bubble charts   — Date x Hour with bubble size encoding magnitude.
  4. Day/Night bars  — Aggregated Day vs. Night comparison bar charts.

Additionally produces long-term stability diagnostics:
  - Rolling mean with 95% CI for key metrics.
  - Cumulative-maxima and weekly extreme-bending percentage.
  - ICC and Spearman-Brown reliability curves (split-half consistency).
  - Per-day Diel Index (Night / Day ratio).

Input
-----
An integrated summary CSV with columns including:
  - ``Datetime`` (format: ``YYYYMMDD_time_HH``, where HH is 1-24)
  - Tail-beat frequency columns (e.g., ``tail1_hz_mean``)
  - Curvature columns (e.g., ``curvature_1_mm_mean_all``)
  - Speed columns (e.g., ``Average speed of Head1(BL/s)``)

Output
------
PNG plot files saved in sub-folders alongside the input CSV.
"""

import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tqdm import tqdm

# =======================
# === USER SETTINGS =====
# =======================
# CSV file OR directory containing a CSV (script will pick first one)
# Provide the path to the integrated summary CSV produced by Step 2
# (windowed_statistics.py), specifically the "Grand_Total_integrated_summary.csv"
# or "Total_integrated_summary.csv" file.
input_path = "<INTEGRATED_SUMMARY_CSV>"
# Example: "/path/to/results/Grand_Total_integrated_summary.csv"

# Date range filter (inclusive). Use 'YYYYMMDD'; set "" to disable either bound.
# Adjust these to match your recording period.
start_date = "<START_DATE>"; end_date = "<END_DATE>"
# Example: start_date = "20241018"; end_date = "20241028"

# Sunrise/Sunset (local hour in [0..24]) for Day/Night splitting and shading.
# Look up local sunrise/sunset times for your recording site and period,
# then set approximate integer hour boundaries.
sun_rise_time = 6    # e.g., 6 for ~6:00 AM local sunrise
sun_set_time  = 20   # e.g., 20 for ~8:00 PM local sunset

# Day/Night shading appearance for diel curve
DAY_FACE_COLOR   = "lightgoldenrodyellow"
NIGHT_FACE_COLOR = "lightgrey"
DAY_ALPHA   = 0.35
NIGHT_ALPHA = 0.28

# Which variables to analyze (leave [] to auto-pick sensible defaults)
manual_columns = [
    "tail1_hz_mean", "tail2_hz_mean",
    "tail3_hz_mean", "tail4_hz_mean",
    "curvature_1_mm_mean_all",
    "Average speed of Head1(BL/s)", "file_sum_Head1_frame_dist_mm",
    "Average speed of MFC5(BL/s)", "file_sum_MFC5_frame_dist_mm",
    "file_sum_Tail1_swim_count", "file_sum_Tail2_swim_count",
    "file_sum_Tail3_swim_count", "file_sum_Tail4_swim_count",
]  # e.g., ["tail1_hz_mean", "Average speed of Head1(BL/s)"]

# Global toggle: show Day/Night average labels on ALL plots
show_day_night_labels = True

# Bubble scaling options (linear only)
bubble_linear_min = 16   # min marker area (points^2) for linear
bubble_linear_max = 720  # max marker area (points^2) for linear

# Output subfolders toggles (kept organized)
make_subfolders = True

# =======================
# === PLOT TOGGLES ======
# =======================
GENERATE_DIEL_CURVES    = True
GENERATE_HEATMAPS       = True
GENERATE_BUBBLE_CHARTS  = True
GENERATE_DAY_NIGHT_BARS = True

# =======================
# === LONG-TERM TESTS ===
# =======================
# Evidence that months-scale generalizes beyond day-to-day noise
GENERATE_LT_STABILITY     = True   # rolling mean ± 95% CI for key metrics
GENERATE_LT_RARE_REGIMES  = True   # cumulative maxima + weekly extreme-bending %
GENERATE_LT_RELIABILITY   = True   # ICC + Spearman–Brown reliability curves
GENERATE_LT_DIEL_INDEX    = True   # per-day Diel Index (Night/Day)

# Parameters for long-term analysis
LT_PRIMARY_TBF_COL          = "freq_mean_hz_mean"           # fallback: avg of tail*_hz_mean
LT_PRIMARY_CURV_COL_MEDIAN  = "curvature_1_mm_median_all"   # fallback: curvature_1_mm_mean_all
LT_ROLLING_WINDOW_DAYS      = 5
LT_BOOTSTRAP_ITER           = 400   # for rolling CI (by day)
LT_RELIABILITY_MAX_K        = None  # if None, will use number of distinct days
LT_RARE_TBF_MAX_COLS        = ["tail1_hz_max","tail2_hz_max","tail3_hz_max","tail4_hz_max"]
LT_RARE_CURV_MAX_COLS       = ["curvature_1_mm_max_all"]    # daily maxima
LT_EXTREME_PCT_COL          = "curvature_1_extreme_percentage"
LT_FRAMES_DENOM_COL         = "bending_total_considered_frames"  # to weight extreme %

# =======================
# === HELPERS ===========
# =======================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def find_csv(path: str) -> str:
    if not path:
        raise ValueError("Please specify the input_path to your CSV file or folder.")
    if os.path.isdir(path):
        csvs = [f for f in os.listdir(path) if f.lower().endswith(".csv")]
        if not csvs:
            raise FileNotFoundError("No CSV file found in the provided directory.")
        if len(csvs) > 1:
            print(f"[Info] Multiple CSV files found in folder. Using the first: {csvs[0]}")
        return os.path.join(path, csvs[0])
    return path

def parse_date_and_hour(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse 'Datetime' like 'YYYYMMDD_time_HH' where HH is 1..24.
    Adds: 'date' (datetime64), 'hour_code' (1..24), 'hour_bin' (0..23), 'xcenter' (bin center).
    """
    if "Datetime" not in df.columns:
        raise ValueError("Required column 'Datetime' not found.")

    dt_str = df["Datetime"].astype(str)
    date_s = dt_str.str.extract(r'(^\d{8})_time_', expand=False)
    hour_s = dt_str.str.extract(r'_time_(\d{1,2})$', expand=False)

    if date_s.isna().all() or hour_s.isna().all():
        raise ValueError("Could not parse 'Datetime'. Expected like 'YYYYMMDD_time_HH'.")

    out = df.copy()
    out["date"] = pd.to_datetime(date_s, format="%Y%m%d", errors="coerce")
    hour_code = pd.to_numeric(hour_s, errors="coerce").astype("Int64")
    out = out.dropna(subset=["date", hour_code.name])
    hour_code = hour_code.astype(int)
    hour_code = np.clip(hour_code, 1, 24)
    out["hour_code"] = hour_code
    out["hour_bin"] = out["hour_code"] - 1
    out["xcenter"] = out["hour_bin"].astype(float) + 0.5
    return out

def apply_date_range(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if start:
        start_dt = pd.to_datetime(start, format="%Y%m%d", errors="coerce")
        if pd.isna(start_dt):
            raise ValueError(f"start_date '{start}' is not in YYYYMMDD format.")
        df = df[df["date"] >= start_dt]
    if end:
        end_dt = pd.to_datetime(end, format="%Y%m%d", errors="coerce")
        if pd.isna(end_dt):
            raise ValueError(f"end_date '{end}' is not in YYYYMMDD format.")
        df = df[df["date"] <= end_dt]
    return df

def get_date_range_string(start: str, end: str) -> str:
    if start and end:
        return f"Date Range: {start} to {end}"
    elif start:
        return f"Date Range: {start} onwards"
    elif end:
        return f"Date Range: up to {end}"
    else:
        return "Date Range: All dates"

def auto_select_columns(df: pd.DataFrame) -> list:
    cols = set()
    exacts = ["tail1_hz_mean"]
    for c in exacts:
        if c in df.columns:
            cols.add(c)
    speed_pattern = re.compile(r'^Average speed of .*\(BL/s\)$')
    for c in df.columns:
        if speed_pattern.match(str(c)):
            cols.add(c)
    return [c for c in df.columns if c in cols]

def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|() ]+', '_', str(name)).strip('_')

def day_night_masks_for_hours(hours: np.ndarray, sunrise: int, sunset: int):
    sunrise = int(np.clip(sunrise, 0, 24))
    sunset  = int(np.clip(sunset, 0, 24))
    if sunrise > sunset:
        sunrise, sunset = sunset, sunrise
    day_mask = (hours >= sunrise) & (hours < sunset)
    night_mask = ~day_mask
    return day_mask, night_mask

def annotate_day_night(ax, day_mean, night_mean, title_suffix=""):
    if not show_day_night_labels:
        return
    lines = []
    if day_mean is not None and np.isfinite(day_mean):
        lines.append(f"Day avg{title_suffix}: {day_mean:.3g}")
    if night_mean is not None and np.isfinite(night_mean):
        lines.append(f"Night avg{title_suffix}: {night_mean:.3g}")
    if not lines:
        return
    txt = "\n".join(lines)
    bbox_props = dict(boxstyle="round,pad=0.35", fc="white", ec="gray", alpha=0.85)
    ax.text(0.98, 0.98, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=bbox_props, zorder=10)

def coerce_numeric(s: pd.Series) -> pd.Series:
    if s.dtype == "O":
        s = s.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")

def aggregate_by_hour(df: pd.DataFrame, col: str) -> pd.DataFrame:
    s = coerce_numeric(df[col])
    agg = (
        pd.DataFrame({"hour_bin": df["hour_bin"], col: s})
        .dropna()
        .groupby("hour_bin")[col]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
        .sort_values("hour_bin")
    )
    full = pd.DataFrame({"hour_bin": range(24)})
    agg = full.merge(agg, on="hour_bin", how="left")
    return agg

def compute_day_night_stats(df: pd.DataFrame, col: str):
    s = coerce_numeric(df[col])
    tmp = df[["date", "hour_bin"]].copy()
    tmp[col] = s
    day_mask, night_mask = day_night_masks_for_hours(tmp["hour_bin"].values, sun_rise_time, sun_set_time)
    tmp["period"] = np.where(day_mask, "Day", "Night")

    def _iqr(x): return np.nanpercentile(x, 75) - np.nanpercentile(x, 25)

    per_day = (
        tmp.dropna(subset=[col])
        .groupby(["date", "period"])[col]
        .agg(mean="mean", median="median", std="std", iqr=_iqr, min="min", max="max", count="count")
        .reset_index()
        .sort_values(["date", "period"])
    )
    full = (
        tmp.dropna(subset=[col])
        .groupby("period")[col]
        .agg(mean="mean", median="median", std="std", iqr=_iqr, min="min", max="max", count="count")
        .reset_index()
        .sort_values("period")
    )
    return per_day, full

def build_date_hour_pivot(df: pd.DataFrame, col: str, aggfunc="mean") -> pd.DataFrame:
    w = df[["date", "hour_bin", col]].copy()
    w[col] = coerce_numeric(w[col])
    w = w.dropna(subset=[col, "date", "hour_bin"])
    piv = (w.groupby(["date", "hour_bin"])[col]
           .agg(aggfunc)
           .unstack("hour_bin")
           .reindex(columns=range(24))
           .sort_index())
    return piv

# ---------- Plotters ----------
def add_day_night_shading(ax, sunrise: float, sunset: float):
    sr = max(0.0, min(24.0, float(min(sunrise, sunset))))
    ss = max(0.0, min(24.0, float(max(sunrise, sunset))))
    if sr > 0:
        ax.axvspan(0, sr, facecolor=NIGHT_FACE_COLOR, alpha=NIGHT_ALPHA, zorder=0)
    if ss > sr:
        ax.axvspan(sr, ss, facecolor=DAY_FACE_COLOR, alpha=DAY_ALPHA, zorder=0)
    if ss < 24:
        ax.axvspan(ss, 24, facecolor=NIGHT_FACE_COLOR, alpha=NIGHT_ALPHA, zorder=0)
    return (
        Patch(facecolor=DAY_FACE_COLOR, alpha=DAY_ALPHA, label="Daytime"),
        Patch(facecolor=NIGHT_FACE_COLOR, alpha=NIGHT_ALPHA, label="Nighttime"),
    )

def plot_diel_curve(grouped: pd.DataFrame, col: str, out_png: str, date_range_str: str):
    x = grouped["hour_bin"].astype(float) + 0.5
    y = grouped["mean"].values
    yerr = grouped["std"].fillna(0.0).values

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    day_patch, night_patch = add_day_night_shading(ax, sun_rise_time, sun_set_time)

    container = ax.errorbar(x, y, yerr=yerr, fmt='-o', capsize=4, zorder=5)
    main_line = container.lines[0]
    main_line.set_label("Mean ± SD")

    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 1))
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel(col)
    ax.set_title(f"{date_range_str}\nDiel Analysis of '{col}'")
    ax.grid(True, alpha=0.4, zorder=1)
    ax.legend(handles=[main_line, day_patch, night_patch], loc="lower right", framealpha=0.9)

    hours = grouped["hour_bin"].values
    day_mask, night_mask = day_night_masks_for_hours(hours, sun_rise_time, sun_set_time)
    day_avg = np.nanmean(grouped.loc[day_mask, "mean"].values) if np.any(day_mask) else None
    night_avg = np.nanmean(grouped.loc[night_mask, "mean"].values) if np.any(night_mask) else None
    annotate_day_night(ax, day_avg, night_avg)

    fig.tight_layout()
    fig.subplots_adjust(top=0.90, right=0.95)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def plot_heatmap(pivot_df: pd.DataFrame, colname: str, out_png: str, date_range_str: str):
    heat = pivot_df.T  # rows=hours, cols=dates
    data = np.ma.masked_invalid(heat.values)
    fig, ax = plt.subplots(figsize=(max(8, heat.shape[1] * 0.35), 7))
    im = ax.imshow(data, aspect="auto", origin="upper")
    ax.set_title(f"{date_range_str}\n{colname} by Date & Hour (mean)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Hour of Day")

    dates = heat.columns.strftime("%Y-%m-%d").tolist()
    step = max(1, len(dates) // 15)
    ax.set_xticks(np.arange(0, len(dates), step))
    ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)], rotation=45, ha="right")
    ax.set_yticks(np.arange(0, 24))
    ax.set_yticklabels([str(h) for h in range(24)])
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(colname)

    if show_day_night_labels:
        hours = np.arange(24)
        day_mask, night_mask = day_night_masks_for_hours(hours, sun_rise_time, sun_set_time)
        day_avg = np.nanmean(heat.values[day_mask, :]) if np.any(day_mask) else None
        night_avg = np.nanmean(heat.values[night_mask, :]) if np.any(night_mask) else None
        annotate_day_night(ax, day_avg, night_avg)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.17, right=0.93, top=0.92)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def _scale_bubble_sizes(Z, min_area=20, max_area=600):
    z = Z.copy().astype(float)
    zmin = np.nanmin(z)
    zmax = np.nanmax(z)
    if not np.isfinite(zmin) or not np.isfinite(zmax) or zmin == zmax:
        return np.ones_like(z) * ((min_area + max_area) / 2.0)
    z_norm = (z - zmin) / (zmax - zmin)  # 0..1
    areas = min_area + z_norm * (max_area - min_area)
    return areas

def plot_bubble_raster(pivot_df: pd.DataFrame, colname: str, out_png: str,
                       date_range_str: str, min_area=20, max_area=600):
    dates = pivot_df.index
    hours = pivot_df.columns  # expected 0..23
    X_idx, Y_idx = np.meshgrid(np.arange(len(dates)), np.arange(len(hours)))
    Z = pivot_df.values.T  # (24, n_dates)

    valid_date_mask = ~np.all(np.isnan(Z), axis=0)
    X_idx = X_idx[:, valid_date_mask]
    Y_idx = Y_idx[:, valid_date_mask]
    Z = Z[:, valid_date_mask]
    dates = dates[valid_date_mask]

    areas = _scale_bubble_sizes(Z, min_area=min_area, max_area=max_area)

    fig, ax = plt.subplots(figsize=(max(8, len(dates) * 0.38), 7))
    sc = ax.scatter(X_idx.flatten(), Y_idx.flatten(),
                    s=areas.flatten(), c=Z.flatten(),
                    cmap=plt.cm.viridis, edgecolors="k",
                    linewidths=0.2, alpha=0.85)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(colname)
    ax.set_title(f"{date_range_str}\n{colname} — Bubble Raster (Linear size)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Hour of Day")

    step = max(1, len(dates) // 15)
    ax.set_xticks(np.arange(0, len(dates), step))
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in dates[::step]], rotation=45, ha="right")
    ax.set_yticks(np.arange(0, len(hours)))
    ax.set_yticklabels([str(h) for h in hours])
    ax.set_ylim(-0.5, len(hours) - 0.5)
    ax.invert_yaxis()
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")

    if show_day_night_labels:
        hrs = np.arange(24)
        day_mask, night_mask = day_night_masks_for_hours(hrs, sun_rise_time, sun_set_time)
        day_avg = np.nanmean(Z[day_mask, :]) if np.any(day_mask) else None
        night_avg = np.nanmean(Z[night_mask, :]) if np.any(night_mask) else None
        annotate_day_night(ax, day_avg, night_avg, title_suffix=" (linear)")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20, right=0.93, top=0.92)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def bar_day_night_overall(full_stats: pd.DataFrame, col: str, out_png: str, date_range_str: str):
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    x = np.arange(2)
    means, labels = [], []
    for p in ["Day", "Night"]:
        row = full_stats[full_stats["period"] == p]
        means.append(float(row["mean"].values[0]) if not row.empty else np.nan)
        labels.append(p)
    ax.bar(x, means, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean")
    ax.set_title(f"{date_range_str}\n{col} — Overall Day vs Night Mean")
    if show_day_night_labels:
        for xi, m in zip(x, means):
            if np.isfinite(m):
                ax.text(xi, m, f"{m:.3g}", ha="center", va="bottom")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

# =======================
# === LONG-TERM HELPERS =
# =======================
def pick_primary_tbf(df: pd.DataFrame) -> pd.Series:
    if LT_PRIMARY_TBF_COL in df.columns:
        return coerce_numeric(df[LT_PRIMARY_TBF_COL])
    # fallback: mean of available tail*_hz_mean
    tail_cols = [c for c in df.columns if re.fullmatch(r"tail\d+_hz_mean", c)]
    if not tail_cols:
        raise ValueError("No TBF column found. Provide freq_mean_hz_mean or tail*_hz_mean.")
    return pd.to_numeric(df[tail_cols].apply(lambda r: coerce_numeric(r).mean(), axis=1), errors="coerce")

def pick_primary_curvature(df: pd.DataFrame) -> pd.Series:
    if LT_PRIMARY_CURV_COL_MEDIAN in df.columns:
        return coerce_numeric(df[LT_PRIMARY_CURV_COL_MEDIAN])
    if "curvature_1_mm_mean_all" in df.columns:
        return coerce_numeric(df["curvature_1_mm_mean_all"])
    fallback = [c for c in df.columns if re.fullmatch(r"curvature_\d+_mm_mean_all", c)]
    if not fallback:
        raise ValueError("No curvature column found. Provide curvature_1_mm_median_all or curvature_1_mm_mean_all.")
    return coerce_numeric(df[fallback[0]])

def daily_aggregate_mean(df: pd.DataFrame, series: pd.Series) -> pd.Series:
    tmp = pd.DataFrame({"date": df["date"], "val": series})
    return tmp.dropna().groupby("date")["val"].mean().sort_index()

def rolling_mean_ci(series: pd.Series, window_days=5, n_boot=400, alpha=0.05):
    """
    Rolling window mean with bootstrap CI by day (resamples days with replacement).
    Returns DataFrame with columns: mean, lo, hi
    """
    s = series.dropna()
    dates = s.index.sort_values()
    means, lo, hi = [], [], []
    idx = []
    for i in range(len(dates)):
        start = dates[max(0, i - window_days + 1)]
        win = s.loc[start:dates[i]]
        # need enough days:
        if len(win) < max(5, min(10, window_days//2)):
            means.append(np.nan); lo.append(np.nan); hi.append(np.nan); idx.append(dates[i]); continue
        win_vals = win.values
        n = len(win_vals)
        boot = []
        for _ in range(n_boot):
            sample = np.random.choice(win_vals, size=n, replace=True)
            boot.append(np.nanmean(sample))
        boot = np.array(boot)
        means.append(np.nanmean(win_vals))
        lo.append(np.nanpercentile(boot, 2.5))
        hi.append(np.nanpercentile(boot, 97.5))
        idx.append(dates[i])
    out = pd.DataFrame({"mean": means, "lo": lo, "hi": hi}, index=pd.to_datetime(idx))
    return out

def compute_icc_oneway(values: np.ndarray, groups: np.ndarray):
    """
    ICC(1,1) one-way random effects with unequal group sizes.
    Returns ICC, MS_between, MS_within, k_bar, g, N
    """
    mask = np.isfinite(values) & pd.notnull(groups)
    x = values[mask]
    g = pd.Categorical(groups[mask])
    groups_idx = g.codes
    N = len(x)
    if N < 3 or g.categories.size < 2:
        return np.nan, np.nan, np.nan, np.nan, g.categories.size, N

    grand = np.nanmean(x)
    SSB = 0.0  # between
    SSW = 0.0  # within
    n_list = []
    for gi, _ in enumerate(g.categories):
        xi = x[groups_idx == gi]
        ni = len(xi)
        if ni == 0:
            continue
        mi = np.nanmean(xi)
        SSB += ni * (mi - grand) ** 2
        SSW += np.nansum((xi - mi) ** 2)
        n_list.append(ni)

    g_num = len(n_list)
    df_between = g_num - 1
    df_within  = N - g_num
    if df_between <= 0 or df_within <= 0:
        return np.nan, np.nan, np.nan, np.nan, g_num, N

    MS_between = SSB / df_between
    MS_within  = SSW / df_within
    k_bar = np.mean(n_list)
    if not np.isfinite(MS_within) or not np.isfinite(MS_between) or not np.isfinite(k_bar):
        return np.nan, MS_between, MS_within, k_bar, g_num, N

    ICC = (MS_between - MS_within) / (MS_between + (k_bar - 1.0) * MS_within)
    return ICC, MS_between, MS_within, k_bar, g_num, N

def spearman_brown(icc, k_array):
    if not np.isfinite(icc) or icc <= -1.0:
        return np.full_like(k_array, np.nan, dtype=float)
    return (k_array * icc) / (1.0 + (k_array - 1.0) * icc)

def wilson_ci(success, total, z=1.96):
    """Wilson proportion CI (returns p_hat, lo, hi)."""
    total = max(1.0, float(total))
    p = float(success) / total
    denom = 1.0 + (z**2)/total
    center = p + (z**2)/(2*total)
    rad = z * math.sqrt((p*(1-p))/total + (z**2)/(4*total**2))
    lo = (center - rad) / denom
    hi = (center + rad) / denom
    return p, max(0.0, lo), min(1.0, hi)

# ---------- NEW: Save helpers for long-term CSVs ----------
def save_series(path: str, series: pd.Series, value_name: str):
    if series is None:
        return
    df = pd.DataFrame({value_name: series})
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df.to_csv(path)

def save_df(path: str, df: pd.DataFrame):
    if df is None:
        return
    df.to_csv(path, index=True)

# ---------- Long-term Plots ----------
def plot_stability(series: pd.Series, rolling_df: pd.DataFrame, title, ylabel, out_png):
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ax.plot(series.index, series.values, ".", alpha=0.4, label="Daily value")
    ax.fill_between(rolling_df.index, rolling_df["lo"], rolling_df["hi"], alpha=0.25, label="95% CI (bootstrap)")
    ax.plot(rolling_df.index, rolling_df["mean"], "-", lw=2.0, label=f"{LT_ROLLING_WINDOW_DAYS}-day rolling mean")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150); plt.close(fig)

# ===== NEW: Diel Index (Night/Day) ===========================================
def diel_index_per_day(df: pd.DataFrame, series_row: pd.Series) -> pd.DataFrame:
    """
    Compute per-day Day mean, Night mean, and Diel Index = Night/Day.
    Returns DataFrame indexed by date with columns: day_mean, night_mean, diel_index.
    """
    tmp = pd.DataFrame({"date": df["date"], "hour_bin": df["hour_bin"], "val": coerce_numeric(series_row)})
    tmp = tmp.dropna(subset=["date", "hour_bin", "val"])
    day_mask, night_mask = day_night_masks_for_hours(tmp["hour_bin"].values, sun_rise_time, sun_set_time)
    tmp["period"] = np.where(day_mask, "Day", "Night")

    agg = tmp.groupby(["date", "period"])["val"].mean().unstack("period")
    # Ensure columns exist
    if "Day" not in agg.columns:   agg["Day"] = np.nan
    if "Night" not in agg.columns: agg["Night"] = np.nan

    out = pd.DataFrame(index=agg.index.sort_values())
    out["day_mean"]   = agg["Day"]
    out["night_mean"] = agg["Night"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["diel_index"] = np.where(np.isfinite(out["day_mean"]) & (np.abs(out["day_mean"]) > 1e-12),
                                     out["night_mean"] / out["day_mean"], np.nan)
    return out

def plot_diel_index(di_df: pd.DataFrame, title: str, out_png: str):
    s = di_df["diel_index"]
    if s.notna().sum() == 0:
        return
    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    ax.plot(s.index, s.values, "-o", alpha=0.9, label="Diel Index (Night/Day)")
    ax.axhline(1.0, linestyle="--", alpha=0.6, label="Night = Day")
    ax.set_ylabel("Diel Index (Night / Day)")
    ax.set_xlabel("Date")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150); plt.close(fig)

def plot_cumulative_max(daily_max: pd.Series, title, ylabel, out_png):
    s = daily_max.sort_index()
    cummax = s.cummax()
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.plot(s.index, s.values, ".", alpha=0.3, label="Daily max")
    ax.plot(cummax.index, cummax.values, "-", lw=2.0, label="Cumulative max")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150); plt.close(fig)

# ===== NEW: Weekly extreme-bending DF + plot (saves handled in MAIN) =========
def compute_weekly_extreme_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted weekly extreme-bending % with Wilson CIs.
    Uses LT_EXTREME_PCT_COL (%) and LT_FRAMES_DENOM_COL (counts).
    Returns DataFrame indexed by week with columns: p_hat, lo, hi, n_frames
    """
    if LT_EXTREME_PCT_COL not in df.columns or LT_FRAMES_DENOM_COL not in df.columns:
        return None

    pct = coerce_numeric(df[LT_EXTREME_PCT_COL])     # 0..100
    denom = coerce_numeric(df[LT_FRAMES_DENOM_COL])  # frames
    ok = df["date"].notna() & np.isfinite(pct) & np.isfinite(denom) & (denom > 0)
    if ok.sum() == 0:
        return None

    tmp = pd.DataFrame({
        "date": df.loc[ok, "date"],
        "extreme_count": pct.loc[ok] / 100.0 * denom.loc[ok],
        "frames": denom.loc[ok]
    })
    tmp["week"] = tmp["date"].dt.to_period("W-MON").apply(lambda p: p.start_time)
    agg = tmp.groupby("week")[["extreme_count","frames"]].sum().sort_index()

    rows = []
    for w, r in agg.iterrows():
        p, lo, hi = wilson_ci(r["extreme_count"], r["frames"])
        rows.append({"week": w, "p_hat": p, "lo": lo, "hi": hi, "n_frames": r["frames"]})
    out = pd.DataFrame(rows).set_index("week").sort_index()
    return out

def plot_weekly_extreme(weekly_df: pd.DataFrame, out_png: str, date_range_str: str):
    if weekly_df is None or weekly_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.plot(weekly_df.index, weekly_df["p_hat"]*100.0, "-o", label="Extreme-bending %")
    ax.fill_between(weekly_df.index, weekly_df["lo"]*100.0, weekly_df["hi"]*100.0,
                    alpha=0.25, label="95% CI (Wilson)")
    ax.set_title(f"{date_range_str}\nWeekly Extreme-bending Percentage (weighted by frames)")
    ax.set_ylabel("Extreme-bending (%)")
    ax.set_xlabel("Week")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150); plt.close(fig)

# ===== NEW: Reliability plotting returns DF so we can save CSV ================
def icc_from_hourly(df: pd.DataFrame, series_row: pd.Series, label="metric"):
    s = series_row
    tmp = pd.DataFrame({"date": df["date"], "hour_bin": df["hour_bin"], "val": s})
    tmp["val"] = coerce_numeric(tmp["val"])
    tmp = tmp.dropna(subset=["val","date","hour_bin"])
    values = tmp["val"].values
    groups = tmp["date"].values
    return compute_icc_oneway(values, groups)

def plot_reliability_from_icc(icc_value, n_days, title, out_png):
    """
    Returns DataFrame with columns: k, reliability, ICC, n_days
    """
    if not np.isfinite(icc_value) or icc_value <= -1.0 or n_days < 2:
        return None
    max_k = n_days if LT_RELIABILITY_MAX_K is None else min(n_days, int(LT_RELIABILITY_MAX_K))
    k = np.arange(1, max_k+1, dtype=float)
    rel = spearman_brown(icc_value, k)

    # Plot
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(k, rel, "-o")
    ax.set_xlabel("# days averaged (k)")
    ax.set_ylabel("Reliability (Spearman–Brown)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.set_title(title + f"\nICC (day grouping) = {icc_value:.3f}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150); plt.close(fig)

    df_rel = pd.DataFrame({"k": k, "reliability": rel})
    df_rel["ICC"] = float(icc_value)
    df_rel["n_days"] = int(n_days)
    return df_rel

# =======================
# === MAIN ==============
# =======================
data_file = find_csv(input_path)
df = pd.read_csv(data_file, sep=",", thousands=",")

# Parse and filter
df = parse_date_and_hour(df)
df = apply_date_range(df, start_date, end_date)
if df.empty:
    raise SystemExit("[Info] No rows remain after applying the date range filter.")

date_range_str = get_date_range_string(start_date, end_date)

# Decide columns
if manual_columns:
    candidate_cols = [c for c in manual_columns if c in df.columns]
    missing = [c for c in manual_columns if c not in df.columns]
    if missing:
        print(f"[Warn] These manual columns were not found and will be skipped: {missing}")
else:
    candidate_cols = auto_select_columns(df)

if not candidate_cols:
    raise SystemExit("[Info] No matching analysis columns found in the dataframe.")

# Prepare output directory (shortened to avoid Windows 260 char limit)
base_dir = os.path.dirname(data_file)
if start_date and end_date:
    root_dir = os.path.join(base_dir, f"Diel_{start_date}_{end_date}_({sun_rise_time}-{sun_set_time})")
elif start_date:
    root_dir = os.path.join(base_dir, f"Diel_{start_date}_on")
elif end_date:
    root_dir = os.path.join(base_dir, f"Diel_to_{end_date}")
else:
    root_dir = os.path.join(base_dir, "Diel_analysis")
os.makedirs(root_dir, exist_ok=True)

# Subfolders
def subdir(name):
    p = os.path.join(root_dir, name) if make_subfolders else root_dir
    os.makedirs(p, exist_ok=True)
    return p

# Always create stats and pivots directories
dir_stats  = subdir("stats")
dir_pivots = subdir("pivots")
dir_curves = subdir("curves")   if GENERATE_DIEL_CURVES   else None
dir_heat   = subdir("heatmaps") if GENERATE_HEATMAPS      else None
dir_bubble = subdir("bubbles")  if GENERATE_BUBBLE_CHARTS else None

# Long-term folder structure
dir_lt        = subdir("longterm")
dir_lt_stab   = ensure_dir(os.path.join(dir_lt, "stability"))      if GENERATE_LT_STABILITY   else None
dir_lt_rare   = ensure_dir(os.path.join(dir_lt, "rare"))           if GENERATE_LT_RARE_REGIMES else None
dir_lt_rel    = ensure_dir(os.path.join(dir_lt, "reliability"))    if GENERATE_LT_RELIABILITY else None
dir_lt_diel   = ensure_dir(os.path.join(dir_lt, "diel_index"))     if GENERATE_LT_DIEL_INDEX  else None

print("[Info] Selected columns for analysis:")
for c in candidate_cols:
    print("  -", c)
print(f"\n[Info] Plot generation settings:")
print(f"  - Diel Curves: {'ON' if GENERATE_DIEL_CURVES else 'OFF'}")
print(f"  - Heatmaps: {'ON' if GENERATE_HEATMAPS else 'OFF'}")
print(f"  - Bubble Charts: {'ON' if GENERATE_BUBBLE_CHARTS else 'OFF'}")
print(f"  - Day/Night Bars: {'ON' if GENERATE_DAY_NIGHT_BARS else 'OFF'}")

# For combined hourly means across variables
combined_means = pd.DataFrame({"hour_bin": range(24)})
combined_means["xcenter"] = combined_means["hour_bin"].astype(float) + 0.5

# Build and cache pivots
pivot_df_by_col = {}

# Main processing loop with progress bar
print("\n[Info] Processing columns...")
for col in tqdm(candidate_cols, desc="Analyzing variables", unit="var"):
    # 1) Diel curve (mean±SD + shading) + save hourly aggregated CSV
    grouped = aggregate_by_hour(df, col)
    safe_name = sanitize_filename(col)

    out_csv = os.path.join(dir_stats, f"{safe_name}_hourly.csv")
    grouped.to_csv(out_csv, index=False)

    if GENERATE_DIEL_CURVES and dir_curves:
        out_curve = os.path.join(dir_curves, f"{safe_name}_diel.png")
        plot_diel_curve(grouped, col, out_curve, date_range_str)

    # 2) Per-day + Full-range Day/Night stats (CSV) + Overall bar plot
    per_day_stats, full_stats = compute_day_night_stats(df, col)
    per_day_csv = os.path.join(dir_stats, f"{safe_name}_perday.csv")
    full_csv    = os.path.join(dir_stats, f"{safe_name}_full.csv")
    per_day_stats.to_csv(per_day_csv, index=False)
    full_stats.to_csv(full_csv, index=False)

    if GENERATE_DAY_NIGHT_BARS:
        out_bar = os.path.join(dir_stats, f"{safe_name}_bar.png")
        bar_day_night_overall(full_stats, col, out_bar, date_range_str)

    # 3) Date×Hour pivot for heatmap/bubbles
    pivot_df = build_date_hour_pivot(df, col, aggfunc="mean")
    pivot_csv = os.path.join(dir_pivots, f"{safe_name}_pivot.csv")
    pivot_df.to_csv(pivot_csv)
    pivot_df_by_col[col] = pivot_df

    # 4) Heatmap
    if GENERATE_HEATMAPS and dir_heat:
        out_heat = os.path.join(dir_heat, f"{safe_name}_heat.png")
        plot_heatmap(pivot_df, col, out_heat, date_range_str)

    # 5) Bubble raster
    if GENERATE_BUBBLE_CHARTS and dir_bubble:
        out_bubble_lin = os.path.join(dir_bubble, f"{safe_name}_bubble.png")
        plot_bubble_raster(pivot_df, col, out_bubble_lin, date_range_str,
                           min_area=bubble_linear_min, max_area=bubble_linear_max)

    # Merge hourly mean for combined table
    combined_means = combined_means.merge(
        grouped[["hour_bin", "mean"]].rename(columns={"mean": f"{col}__mean"}),
        on="hour_bin", how="left"
    )

# Save combined hourly means
combined_csv = os.path.join(dir_stats, "combined_hourly_means.csv")
combined_means.to_csv(combined_csv, index=False)

# =======================
# === LONG-TERM: NEW  ===
# =======================
print("\n[Info] Running long-term evidence pack...")

# ---- pick primary series (TBF & Curvature) ----
try:
    s_tbf_row = pick_primary_tbf(df)
except Exception as e:
    s_tbf_row = None
    print(f"[Warn] TBF primary not available: {e}")

try:
    s_curv_row = pick_primary_curvature(df)
except Exception as e:
    s_curv_row = None
    print(f"[Warn] Curvature primary not available: {e}")

# daily aggregates
daily_tbf  = daily_aggregate_mean(df, s_tbf_row)   if s_tbf_row is not None else None
daily_curv = daily_aggregate_mean(df, s_curv_row)  if s_curv_row is not None else None

# 1) STABILITY: rolling mean ± CI  (SAVE processed data)
if GENERATE_LT_STABILITY:
    if daily_tbf is not None:
        roll_tbf  = rolling_mean_ci(daily_tbf, window_days=LT_ROLLING_WINDOW_DAYS, n_boot=LT_BOOTSTRAP_ITER)
        # Save processed
        save_series(os.path.join(dir_lt_stab, "tbf_daily_mean.csv"), daily_tbf, "daily_mean_hz")
        save_df(os.path.join(dir_lt_stab, "tbf_rolling_mean_CI.csv"), roll_tbf)
        # Plot
        plot_stability(daily_tbf, roll_tbf,
                       title=f"{date_range_str}\nRolling Stability — Tail-beat Frequency (daily mean, {LT_ROLLING_WINDOW_DAYS}-day window)",
                       ylabel="Hz",
                       out_png=os.path.join(dir_lt_stab, "stability_tbf.png"))
    if daily_curv is not None:
        roll_curv = rolling_mean_ci(daily_curv, window_days=LT_ROLLING_WINDOW_DAYS, n_boot=LT_BOOTSTRAP_ITER)
        # Save processed
        save_series(os.path.join(dir_lt_stab, "curvature_daily_mean.csv"), daily_curv, "daily_mean_curvature_mm")
        save_df(os.path.join(dir_lt_stab, "curvature_rolling_mean_CI.csv"), roll_curv)
        # Plot
        plot_stability(daily_curv, roll_curv,
                       title=f"{date_range_str}\nRolling Stability — Curvature (daily median/mean, {LT_ROLLING_WINDOW_DAYS}-day window)",
                       ylabel="Curvature (mm)",
                       out_png=os.path.join(dir_lt_stab, "stability_curvature.png"))

# 2) RARE REGIMES: cumulative maxima + weekly extreme-bending %  (SAVE processed data)
def daily_max_from_cols(df: pd.DataFrame, cols: list, label: str):
    avail = [c for c in cols if c in df.columns]
    if not avail:
        return None
    tmp = df[["date"]+avail].copy()
    for c in avail:
        tmp[c] = coerce_numeric(tmp[c])
    tmp["row_max"] = tmp[avail].max(axis=1, numeric_only=True)
    return tmp.dropna(subset=["row_max"]).groupby("date")["row_max"].max().sort_index()

if GENERATE_LT_RARE_REGIMES:
    # TBF cumulative max
    dm_tbf = daily_max_from_cols(df, LT_RARE_TBF_MAX_COLS, "TBF max")
    if dm_tbf is not None:
        # Save daily max & cummax
        df_out = pd.DataFrame({"daily_max_hz": dm_tbf})
        df_out["cummax_hz"] = df_out["daily_max_hz"].cummax()
        df_out.index.name = "date"
        df_out.to_csv(os.path.join(dir_lt_rare, "tbf_daily_max_and_cummax.csv"))
        # Plot
        plot_cumulative_max(dm_tbf,
            title=f"{date_range_str}\nCumulative Maximum — Tail-beat Frequency",
            ylabel="Hz (daily max)",
            out_png=os.path.join(dir_lt_rare, "cumulative_max_tbf.png"))
    # Curvature cumulative max
    dm_curv = daily_max_from_cols(df, LT_RARE_CURV_MAX_COLS, "Curvature max")
    if dm_curv is not None:
        df_out = pd.DataFrame({"daily_max_curvature_mm": dm_curv})
        df_out["cummax_curvature_mm"] = df_out["daily_max_curvature_mm"].cummax()
        df_out.index.name = "date"
        df_out.to_csv(os.path.join(dir_lt_rare, "curvature_daily_max_and_cummax.csv"))
        # Plot
        plot_cumulative_max(dm_curv,
            title=f"{date_range_str}\nCumulative Maximum — Curvature",
            ylabel="Curvature (mm, daily max)",
            out_png=os.path.join(dir_lt_rare, "cumulative_max_curvature.png"))
    # Weekly extreme-bending % (save DF with Wilson CIs)
    weekly_ext = compute_weekly_extreme_df(df)
    if weekly_ext is not None:
        weekly_ext.to_csv(os.path.join(dir_lt_rare, "weekly_extreme_bending.csv"))
        plot_weekly_extreme(weekly_ext, out_png=os.path.join(dir_lt_rare, "weekly_extreme_bending.png"),
                            date_range_str=date_range_str)

# 3) RELIABILITY: ICC + Spearman–Brown vs #days  (SAVE processed data)
def icc_from_hourly_values(df: pd.DataFrame, series_row: pd.Series):
    s = series_row
    tmp = pd.DataFrame({"date": df["date"], "hour_bin": df["hour_bin"], "val": s})
    tmp["val"] = coerce_numeric(tmp["val"])
    tmp = tmp.dropna(subset=["val","date","hour_bin"])
    values = tmp["val"].values
    groups = tmp["date"].values
    return compute_icc_oneway(values, groups)

if GENERATE_LT_RELIABILITY:
    n_days = df["date"].nunique()

    # 1) TBF
    if s_tbf_row is not None:
        icc_tbf, MSb, MSc, kbar, gnum, N = icc_from_hourly_values(df, s_tbf_row)
        df_rel = plot_reliability_from_icc(icc_tbf, n_days,
            title=f"{date_range_str}\nReliability vs #days — TBF (from ICC by day)",
            out_png=os.path.join(dir_lt_rel, "reliability_tbf.png"))
        # Save reliability curve + ICC summary
        if df_rel is not None:
            df_rel.to_csv(os.path.join(dir_lt_rel, "reliability_curve_tbf.csv"), index=False)
            pd.DataFrame([{
                "ICC": icc_tbf, "MS_between": MSb, "MS_within": MSc,
                "k_bar": kbar, "groups": gnum, "N": N, "n_days": n_days
            }]).to_csv(os.path.join(dir_lt_rel, "reliability_tbf_icc_summary.csv"), index=False)

    # 2) Curvature
    if s_curv_row is not None:
        icc_curv, MSb, MSc, kbar, gnum, N = icc_from_hourly_values(df, s_curv_row)
        df_rel = plot_reliability_from_icc(icc_curv, n_days,
            title=f"{date_range_str}\nReliability vs #days — Curvature (from ICC by day)",
            out_png=os.path.join(dir_lt_rel, "reliability_curvature.png"))
        if df_rel is not None:
            df_rel.to_csv(os.path.join(dir_lt_rel, "reliability_curve_curvature.csv"), index=False)
            pd.DataFrame([{
                "ICC": icc_curv, "MS_between": MSb, "MS_within": MSc,
                "k_bar": kbar, "groups": gnum, "N": N, "n_days": n_days
            }]).to_csv(os.path.join(dir_lt_rel, "reliability_curvature_icc_summary.csv"), index=False)

    # 3) (Optional) MFC5 average speed, if present
    col_mfc5_speed = "Average speed of MFC5(BL/s)"
    if col_mfc5_speed in df.columns:
        s_mfc5 = coerce_numeric(df[col_mfc5_speed])
        icc_mfc5, MSb, MSc, kbar, gnum, N = icc_from_hourly_values(df, s_mfc5)
        df_rel = plot_reliability_from_icc(icc_mfc5, n_days,
            title=f"{date_range_str}\nReliability vs #days — MFC5 Average Speed (from ICC by day)",
            out_png=os.path.join(dir_lt_rel, "reliability_mfc5_speed.png"))
        if df_rel is not None:
            df_rel.to_csv(os.path.join(dir_lt_rel, "reliability_curve_mfc5_speed.csv"), index=False)
            pd.DataFrame([{
                "ICC": icc_mfc5, "MS_between": MSb, "MS_within": MSc,
                "k_bar": kbar, "groups": gnum, "N": N, "n_days": n_days
            }]).to_csv(os.path.join(dir_lt_rel, "reliability_mfc5_speed_icc_summary.csv"), index=False)

# 4) DIEL INDEX (Night/Day) — per day
if GENERATE_LT_DIEL_INDEX:
    # TBF
    if s_tbf_row is not None:
        di_tbf = diel_index_per_day(df, s_tbf_row)
        # Save to stats (existing) AND longterm/diel_index (new)
        di_tbf.to_csv(os.path.join(dir_stats, "TBF_diel_index_perday.csv"), index=True)
        di_tbf.to_csv(os.path.join(dir_lt_diel, "diel_index_tbf.csv"), index=True)
        plot_diel_index(di_tbf,
            title=f"{date_range_str}\nDiel Index (Night/Day) — TBF",
            out_png=os.path.join(dir_lt_diel, "diel_index_tbf.png"))
    # Curvature
    if s_curv_row is not None:
        di_curv = diel_index_per_day(df, s_curv_row)
        di_curv.to_csv(os.path.join(dir_stats, "Curvature_diel_index_perday.csv"), index=True)
        di_curv.to_csv(os.path.join(dir_lt_diel, "diel_index_curvature.csv"), index=True)
        plot_diel_index(di_curv,
            title=f"{date_range_str}\nDiel Index (Night/Day) — Curvature",
            out_png=os.path.join(dir_lt_diel, "diel_index_curvature.png"))
    # Optional: MFC5 speed
    col_mfc5_speed = "Average speed of MFC5(BL/s)"
    if col_mfc5_speed in df.columns:
        di_mfc5 = diel_index_per_day(df, coerce_numeric(df[col_mfc5_speed]))
        di_mfc5.to_csv(os.path.join(dir_stats, "MFC5_speed_diel_index_perday.csv"), index=True)
        di_mfc5.to_csv(os.path.join(dir_lt_diel, "diel_index_mfc5_speed.csv"), index=True)
        plot_diel_index(di_mfc5,
            title=f"{date_range_str}\nDiel Index (Night/Day) — MFC5 Average Speed",
            out_png=os.path.join(dir_lt_diel, "diel_index_mfc5_speed.png"))

print(f"\n[Done] Diel + Long-term analysis complete.")
print(f"Date range applied: {start_date or 'ALL'} to {end_date or 'ALL'}")
print(f"Sunrise={sun_rise_time}, Sunset={sun_set_time}")
print(f"Outputs in: {root_dir}")
