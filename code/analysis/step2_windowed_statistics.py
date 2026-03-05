#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windowed Swimming Statistics (Step 2)

Reads the per-video tracking-analysis CSVs produced by Step 1
(pose_tracking_analysis.py) and computes time-windowed statistics:

  - Tail-beat frequencies (Hz) per window.
  - Curvature statistics (mean, median, std) per window.
  - Body-part frame distances (summed per window).
  - Bending-class distribution across the entire recording.
  - Average swimming speed in body lengths per second (BL/s).
  - Per-file and cross-file integrated summaries.

Supports sweeping multiple root directories in one run, with per-root
and grand-total outputs.

Input
-----
Each root directory should contain *_tracking_analysis.csv files
(the output of Step 1). File discovery is non-recursive (top-level only).

Output
------
Under each root:
  <root.name>_swim_freq_curv_results/per_file/  -- per-file windowed CSVs.
  <root.name>_swim_freq_curv_results/logs/       -- processing logs.
  Total_summary_swim_freq_curv_stats_O_skipped_frame.csv
  Total_integrated_summary.csv
Grand total across all roots is saved under the first root result folder.
"""

from __future__ import annotations
import os, sys, re, logging, math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import pandas as pd

from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ============================ GLOBAL PARAMETERS ============================

ROOT_OVERRIDES: List[str] = [
    "<STEP1_OUTPUT_DIR>",
    # Example: "/path/to/experiment_01/tracking_results",
    # Each directory should contain *_tracking_analysis.csv files
    # produced by pose_tracking_analysis.py (Step 1).
    # Add more paths to process multiple recording sessions:
    # "/path/to/experiment_02/tracking_results",
]

FPS: float = 20.0
WINDOW_SECONDS: float = 10.0

# Body length for BL/s conversion (mm)
BL_MM: float = 490.0

# File discovery (NON-RECURSIVE)
FILE_GLOB_TOPLEVEL: str = "*.csv"
FILE_INCLUDE_SUBSTRING: Optional[str] = "_tracking_analysis"  # case-insensitive; set None to disable

# Result structure
RESULT_DIRNAME: str = "Final_total_summary_result"
PER_FILE_DIRNAME: str = "per_file"
LOG_DIRNAME: str = "logs"

# Parallelism & progress
USE_MULTIPROCESSING: bool = True
CPU_USAGE: float = 0.60
SHOW_PROGRESS: bool = True

# Reference curvature for skipping
REFERENCE_CURVATURE_FOR_SKIP: str = "auto"

# Distance columns to sum per window (legacy behavior — unchanged)
DISTANCE_COLS = [
    "Head1_total_distance_px", "Head1_total_distance_mm",
    "Head2_total_distance_px", "Head2_total_distance_mm",
    "MFC1_total_distance_px",  "MFC1_total_distance_mm",
    "MFC2_total_distance_px",  "MFC2_total_distance_mm",
    "MFC3_total_distance_px",  "MFC3_total_distance_mm",
    "MFC4_total_distance_px",  "MFC4_total_distance_mm",
    "MFC5_total_distance_px",  "MFC5_total_distance_mm",
    "Tail1_total_distance_px", "Tail1_total_distance_mm",
    "Tail2_total_distance_px", "Tail2_total_distance_mm",
    "Tail3_total_distance_px", "Tail3_total_distance_mm",
    "Tail4_total_distance_px", "Tail4_total_distance_mm",
]

# =============================== LOGGING ===================================

def make_logger(log_folder: Path) -> logging.Logger:
    log_folder.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_folder / f"run_{ts}.log"
    logger = logging.getLogger("swimfreq")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

# ================================ UTILS ====================================

def resolve_home_from_env() -> Path:
    hp = os.environ.get("HOMEPATH", "").strip()
    hd = os.environ.get("HOMEDRIVE", "").strip()
    if hp:
        if hp.startswith("\\") or hp.startswith("/"):
            if hd:
                return Path(hd + hp)
        return Path(hp)
    return Path.home()

def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    for enc in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return None

def curvature_columns(df: pd.DataFrame) -> List[str]:
    patt = re.compile(r"^curvature_(\d+)_mm$")
    cols = [c for c in df.columns if patt.match(c)]
    cols.sort(key=lambda c: int(re.findall(r"\d+", c)[0]))
    return cols

# --- NEW: find curvature radius & length columns (stat-type) ----------------
def curvature_radius_columns(df: pd.DataFrame) -> List[str]:
    patt = re.compile(r"^curvature_\d+_radius_(px|mm)$", re.IGNORECASE)
    return [c for c in df.columns if patt.match(c)]

def length_series_columns(df: pd.DataFrame) -> List[str]:
    patt = re.compile(r"^(mfc_length|body_length)_(px|mm)$", re.IGNORECASE)
    return [c for c in df.columns if patt.match(c)]

# --- NEW: find non-cumulative sums-type columns -----------------------------
def frame_distance_columns(df: pd.DataFrame) -> List[str]:
    patt = re.compile(r"^(Head\d+|MFC\d+|Tail\d+)_frame_dist_(px|mm)$", re.IGNORECASE)
    return [c for c in df.columns if patt.match(c)]

def tail_travel_distance_columns(df: pd.DataFrame) -> List[str]:
    patt = re.compile(r"^Tail\d+_travel_distance_(px|mm)$", re.IGNORECASE)
    return [c for c in df.columns if patt.match(c)]

def tail_raw_swimcount_columns(df: pd.DataFrame) -> List[str]:
    patt = re.compile(r"^Tail\d+_swim_count$", re.IGNORECASE)
    return [c for c in df.columns if patt.match(c)]

def pick_reference_curvature(df: pd.DataFrame) -> Optional[str]:
    if REFERENCE_CURVATURE_FOR_SKIP and REFERENCE_CURVATURE_FOR_SKIP != "auto":
        return REFERENCE_CURVATURE_FOR_SKIP if REFERENCE_CURVATURE_FOR_SKIP in df.columns else None
    for k in (3, 2, 1):
        c = f"curvature_{k}_mm"
        if c in df.columns:
            return c
    cols = curvature_columns(df)
    return cols[0] if cols else None

def pick_curvature_for_peak(df: pd.DataFrame, ref_curv: Optional[str]) -> Optional[str]:
    if "curvature_1_mm" in df.columns:
        return "curvature_1_mm"
    if ref_curv and ref_curv in df.columns:
        return ref_curv
    cols = curvature_columns(df)
    return cols[0] if cols else None

def has_required_signals(df: pd.DataFrame) -> bool:
    tails = [f"Tail{i}_swim_count" for i in range(1, 5)]
    curvs = curvature_columns(df)
    return any(c in df.columns for c in tails) or len(curvs) > 0

def numeric_series(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    if col not in df.columns:
        return None
    return pd.to_numeric(df[col], errors="coerce")

def _looks_cumulative(s: pd.Series, tol: float = 1e-6) -> bool:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.size < 5:
        return False
    d = np.diff(x.values)
    if (d < -tol).sum() > 0:
        return False
    nonneg_ratio = (d >= -tol).mean() if d.size > 0 else 0.0
    return nonneg_ratio >= 0.80

def pick_swim_series(df: pd.DataFrame, tail_name: str) -> Tuple[Optional[pd.Series], Optional[str], bool]:
    cand_cum = f"{tail_name}_cumulative_swim_count"
    cand = f"{tail_name}_swim_count"
    s_cum = numeric_series(df, cand_cum)
    s = numeric_series(df, cand)
    if s_cum is not None and _looks_cumulative(s_cum):
        return s_cum, cand_cum, True
    if s is not None:
        if _looks_cumulative(s):
            return s, cand, True
        else:
            return s, cand, False
    return None, None, False

def seconds_to_mmss(sec: float) -> Optional[str]:
    if not (isinstance(sec, (int, float)) and np.isfinite(sec)):
        return None
    sec = float(max(0.0, sec))
    m = int(sec // 60)
    s = sec - m * 60
    return f"{m:02d}:{s:06.3f}".replace(",", ".")

# ============================= ROUNDING / CAST ==============================

def _is_curvature_col(c: str) -> bool:
    return bool(re.match(r"^curvature_\d+_mm", c))

def _is_hz_col(c: str) -> bool:
    return "hz" in c.lower()

def _is_pct_col(c: str) -> bool:
    s = c.lower()
    return s.endswith("_pct") or ("percentage" in s) or s.endswith("_rate_pct")

def _is_time_col(c: str) -> bool:
    return c.lower().endswith("_time_s") or c.lower() == "duration_s"

def _is_distance_col(c: str) -> bool:
    s = c.lower()
    # treat new sum_ and file_sum_ distance-like aggregates the same
    return s.endswith("_total_distance_px") or s.endswith("_total_distance_mm") or s.startswith("sum_") or s.startswith("file_sum_")

def _int_columns_window(df_cols: Iterable[str]) -> List[str]:
    base = {
        "window_idx", "start_frame", "end_frame", "n_tails_used",
        "frames_in_window", "frames_non_skipped_window", "frames_skipped_window",
        "file_skipped_frames"
    }
    base |= {c for c in df_cols if c.endswith("_n")}
    return [c for c in df_cols if c in base]

def _int_columns_summary(df_cols: Iterable[str]) -> List[str]:
    base = {
        "n_windows", "n_frames", "skipped_frames",
        "bending_total_valid_frames", "bending_total_considered_frames",
        "max_freq_window_idx", "max_freq_start_frame", "max_freq_end_frame",
        "max_curvature_frame_idx"
    }
    return [c for c in df_cols if c in base]

def round_and_cast(df: pd.DataFrame, is_summary: bool = False) -> pd.DataFrame:
    out = df.copy()
    int_cols = _int_columns_summary(out.columns) if is_summary else _int_columns_window(out.columns)
    for c in int_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").round(0).astype("Int64")
    for c in out.columns:
        if c in int_cols:
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            if _is_curvature_col(c):
                out[c] = out[c].round(5)
            elif _is_hz_col(c) or _is_pct_col(c):
                out[c] = out[c].round(2)
            elif _is_time_col(c):
                out[c] = out[c].round(3)
            elif _is_distance_col(c) or c in ("fps", "window_seconds"):
                out[c] = out[c].round(2)
            else:
                out[c] = out[c].round(3)
    return out

# ============================ CORE COMPUTATIONS =============================

def window_indices(n_frames: int, win_size: int) -> List[Tuple[int, int, int]]:
    out = []
    if n_frames <= 0 or win_size <= 0:
        return out
    k = 0
    i = 0
    while i < n_frames:
        j = min(i + win_size, n_frames)
        out.append((k, i, j))
        k += 1
        i = j
    return out

def detect_skipped_frames_by_curvature(df: pd.DataFrame, ref_curv_col: Optional[str]) -> pd.Series:
    if not ref_curv_col or ref_curv_col not in df.columns:
        return pd.Series(False, index=df.index)
    return pd.to_numeric(df[ref_curv_col], errors="coerce").isna()

def increments_cumulative_adjacent(s: pd.Series, skip_mask: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    d = s.diff()
    prev_non_skipped = (~skip_mask.shift(1, fill_value=True))
    valid_pairs = (~skip_mask) & prev_non_skipped & s.notna() & s.shift(1).notna()
    inc = d.where(valid_pairs, 0.0).clip(lower=0.0).fillna(0.0)
    return inc

def increments_instantaneous_frame(s: pd.Series, skip_mask: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    inc = x.where(~skip_mask, 0.0).fillna(0.0).clip(lower=0.0)
    return inc

def _series_stats_over_window(values: pd.Series, mask: pd.Series) -> Dict[str, float]:
    v = pd.to_numeric(values, errors="coerce")
    v = v.where(mask, np.nan).dropna().values
    if v.size > 0:
        return {
            "mean": float(np.mean(v)),
            "median": float(np.median(v)),
            "std": float(np.std(v)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "n": int(v.size),
        }
    return {"mean": np.nan, "median": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "n": 0}

def compute_window_table(
    df: pd.DataFrame,
    source_path: Path,
    fps: float,
    window_seconds: float,
    skip_mask: pd.Series
) -> pd.DataFrame:
    n = len(df)
    win_size = max(1, int(round(fps * window_seconds)))
    windows = window_indices(n, win_size)

    tails = [f"Tail{i}" for i in range(1, 5)]
    tail_incs = {}
    for t in tails:
        s, _, is_cum = pick_swim_series(df, t)
        if s is None:
            tail_incs[t] = None
        else:
            tail_incs[t] = increments_cumulative_adjacent(s, skip_mask) if is_cum else increments_instantaneous_frame(s, skip_mask)

    # --- Existing curvature stats (unchanged) ---
    curv_cols = curvature_columns(df)
    curv_num: Dict[str, pd.Series] = {c: pd.to_numeric(df[c], errors="coerce") for c in curv_cols}

    # --- NEW: curvature radius & length series for stats ---
    radius_cols = curvature_radius_columns(df)
    length_cols = length_series_columns(df)
    radius_num: Dict[str, pd.Series] = {c: pd.to_numeric(df[c], errors="coerce") for c in radius_cols}
    length_num: Dict[str, pd.Series] = {c: pd.to_numeric(df[c], errors="coerce") for c in length_cols}

    # --- Existing distance sums (unchanged) ---
    dist_cols_present = [c for c in DISTANCE_COLS if c in df.columns]
    dist_num: Dict[str, pd.Series] = {c: pd.to_numeric(df[c], errors="coerce").fillna(0.0) for c in dist_cols_present}

    # --- NEW sums: frame distances / tail travel / raw swim_count (non-cumulative only) ---
    frame_dist_cols_present = [c for c in frame_distance_columns(df) if "cum" not in c.lower()]
    travel_cols_present = [c for c in tail_travel_distance_columns(df) if "cum" not in c.lower()]
    raw_swim_cols_present = tail_raw_swimcount_columns(df)  # instantaneous; sum over window

    frame_dist_num: Dict[str, pd.Series] = {c: pd.to_numeric(df[c], errors="coerce").fillna(0.0) for c in frame_dist_cols_present}
    travel_num: Dict[str, pd.Series] = {c: pd.to_numeric(df[c], errors="coerce").fillna(0.0) for c in travel_cols_present}
    raw_swim_num: Dict[str, pd.Series] = {c: pd.to_numeric(df[c], errors="coerce").fillna(0.0) for c in raw_swim_cols_present}

    file_skipped_frames = int(skip_mask.sum())
    file_skipped_rate_pct = (file_skipped_frames / n * 100.0) if n > 0 else 0.0

    rows = []
    for (widx, s_idx, e_idx) in windows:
        frames_in_window = (e_idx - s_idx)
        window_slice = slice(s_idx, e_idx)
        mask_win = ~skip_mask.iloc[window_slice]
        non_skipped = int(mask_win.sum())
        skipped = frames_in_window - non_skipped
        skipped_ratio = (skipped / frames_in_window) if frames_in_window > 0 else 0.0

        dur = max(1e-9, non_skipped / float(fps))
        high_skip_freq = skipped_ratio > 0.70

        t_counts, t_hz = {}, {}
        t_hz_vals = []

        if high_skip_freq:
            for t in tails:
                t_counts[t] = np.nan
                t_hz[t] = np.nan
            freq_mean = freq_med = freq_min = freq_max = np.nan
            n_used = 0
        else:
            for t in tails:
                inc_series = tail_incs.get(t)
                if inc_series is None:
                    t_counts[t] = np.nan
                    t_hz[t] = np.nan
                    continue
                count = float(pd.to_numeric(inc_series.iloc[window_slice], errors="coerce").where(mask_win, 0.0).sum())
                freq = (count / 2.0) / dur if dur > 0 else np.nan
                t_counts[t] = count
                t_hz[t] = freq
                if np.isfinite(freq):
                    t_hz_vals.append(freq)

            if t_hz_vals:
                freq_mean = float(np.mean(t_hz_vals))
                freq_med  = float(np.median(t_hz_vals))
                freq_min  = float(np.min(t_hz_vals))
                freq_max  = float(np.max(t_hz_vals))
                n_used    = int(len(t_hz_vals))
            else:
                freq_mean = freq_med = freq_min = freq_max = np.nan
                n_used = 0

        # Curvature stats (existing)
        curv_stats: Dict[str, float] = {}
        for c in curv_cols:
            stats = _series_stats_over_window(curv_num[c].iloc[window_slice], mask_win)
            prefix = c
            curv_stats.update({
                f"{prefix}_mean":   stats["mean"],
                f"{prefix}_median": stats["median"],
                f"{prefix}_std":    stats["std"],
                f"{prefix}_min":    stats["min"],
                f"{prefix}_max":    stats["max"],
                f"{prefix}_n":      stats["n"],
            })

        # NEW: curvature radius stats
        radius_stats: Dict[str, float] = {}
        for c in radius_cols:
            stats = _series_stats_over_window(radius_num[c].iloc[window_slice], mask_win)
            radius_stats.update({
                f"{c}_mean":   stats["mean"],
                f"{c}_median": stats["median"],
                f"{c}_std":    stats["std"],
                f"{c}_min":    stats["min"],
                f"{c}_max":    stats["max"],
                f"{c}_n":      stats["n"],
            })

        # NEW: length stats (mfc_length/body_length)
        length_stats: Dict[str, float] = {}
        for c in length_cols:
            stats = _series_stats_over_window(length_num[c].iloc[window_slice], mask_win)
            length_stats.update({
                f"{c}_mean":   stats["mean"],
                f"{c}_median": stats["median"],
                f"{c}_std":    stats["std"],
                f"{c}_min":    stats["min"],
                f"{c}_max":    stats["max"],
                f"{c}_n":      stats["n"],
            })

        # Distance sums (existing)
        dist_stats: Dict[str, float] = {}
        for c in dist_cols_present:
            vals = dist_num[c].iloc[window_slice]
            dist_stats[f"sum_{c}"] = float(vals.where(mask_win, 0.0).sum())

        # NEW sums: frame distances / travel distances / raw swim counts
        for c in frame_dist_cols_present:
            vals = frame_dist_num[c].iloc[window_slice]
            dist_stats[f"sum_{c}"] = float(vals.where(mask_win, 0.0).sum())

        for c in travel_cols_present:
            vals = travel_num[c].iloc[window_slice]
            dist_stats[f"sum_{c}"] = float(vals.where(mask_win, 0.0).sum())

        for c in raw_swim_cols_present:
            vals = raw_swim_num[c].iloc[window_slice]
            dist_stats[f"sum_{c}"] = float(vals.where(mask_win, 0.0).sum())

        row = {
            "source_path": str(source_path.resolve()),
            "file": source_path.stem,
            "fps": fps,
            "window_seconds": window_seconds,
            "window_idx": widx,
            "start_frame": s_idx,
            "end_frame": e_idx,
            "start_time_s": s_idx / fps,
            "end_time_s": e_idx / fps,
            "mid_time_s": (0.5 * (s_idx + e_idx)) / fps,
            "duration_s": dur,
            "tail1_count": t_counts.get("Tail1", np.nan),
            "tail2_count": t_counts.get("Tail2", np.nan),
            "tail3_count": t_counts.get("Tail3", np.nan),
            "tail4_count": t_counts.get("Tail4", np.nan),
            "tail1_hz": t_hz.get("Tail1", np.nan),
            "tail2_hz": t_hz.get("Tail2", np.nan),
            "tail3_hz": t_hz.get("Tail3", np.nan),
            "tail4_hz": t_hz.get("Tail4", np.nan),
            "freq_mean_hz": freq_mean,
            "freq_median_hz": freq_med,
            "freq_min_hz": freq_min,
            "freq_max_hz": freq_max,
            "n_tails_used": n_used,
            "frames_in_window": frames_in_window,
            "frames_non_skipped_window": non_skipped,
            "frames_skipped_window": skipped,
            "skipped_rate_window_pct": (skipped_ratio * 100.0) if frames_in_window > 0 else 0.0,
            "file_skipped_frames": file_skipped_frames,
            "file_skipped_rate_pct": file_skipped_rate_pct,
        }
        row.update(curv_stats)
        row.update(radius_stats)   # NEW
        row.update(length_stats)   # NEW
        row.update(dist_stats)
        rows.append(row)

    return pd.DataFrame(rows)

def describe_block(win_df: pd.DataFrame, col: str) -> Dict[str, float]:
    x = pd.to_numeric(win_df[col], errors="coerce").dropna().values
    if x.size == 0:
        return {f"{col}_mean": np.nan, f"{col}_median": np.nan, f"{col}_std": np.nan,
                f"{col}_min": np.nan, f"{col}_max": np.nan, f"{col}_p10": np.nan, f"{col}_p90": np.nan,
                f"{col}_nonzero_ratio": np.nan}
    return {
        f"{col}_mean": float(np.mean(x)),
        f"{col}_median": float(np.median(x)),
        f"{col}_std": float(np.std(x)),
        f"{col}_min": float(np.min(x)),
        f"{col}_max": float(np.max(x)),
        f"{col}_p10": float(np.percentile(x, 10)),
        f"{col}_p90": float(np.percentile(x, 90)),
        f"{col}_nonzero_ratio": float(np.mean(x > 0.0)),
    }

def bending_distribution_curv1(df: pd.DataFrame, skip_mask: pd.Series) -> Dict[str, float]:
    col = "curvature_1_class"
    considered_mask = (~skip_mask)
    total_considered = int(considered_mask.sum())
    if col not in df.columns or total_considered == 0:
        return {
            "curvature_1_minimal_percentage": np.nan,
            "curvature_1_normal_percentage":  np.nan,
            "curvature_1_extreme_percentage": np.nan,
            "curvature_1_invalid_percentage": np.nan,
            "bending_total_valid_frames": 0,
            "bending_total_considered_frames": total_considered,
            "count_minimal": 0, "count_normal": 0, "count_extreme": 0, "count_invalid": total_considered
        }
    s_considered = df.loc[considered_mask, col]
    valid_mask = s_considered.notna()
    total_valid = int(valid_mask.sum())
    invalid = total_considered - total_valid

    vals = s_considered[valid_mask].astype(str).str.strip().str.lower()
    minimal = int((vals == "minimal").sum())
    normal  = int((vals == "normal").sum())
    extreme = int((vals == "extreme").sum())

    return {
        "curvature_1_minimal_percentage": minimal / total_considered * 100.0 if total_considered else np.nan,
        "curvature_1_normal_percentage":  normal  / total_considered * 100.0 if total_considered else np.nan,
        "curvature_1_extreme_percentage": extreme / total_considered * 100.0 if total_considered else np.nan,
        "curvature_1_invalid_percentage": invalid / total_considered * 100.0 if total_considered else np.nan,
        "bending_total_valid_frames": total_valid,
        "bending_total_considered_frames": total_considered,
        "count_minimal": minimal, "count_normal": normal, "count_extreme": extreme, "count_invalid": invalid
    }

def simplify_perfile_stem(stem: str) -> str:
    m = re.search(r"(T\d{8}_time_\d+)", stem)
    front = m.group(1) if m else None
    i = stem.lower().find("tracking_analysis")
    tail = stem[i:] if i >= 0 else None
    if front and tail:
        return f"{front}_{tail}"
    return stem

# --- NEW: parse Datetime from simplified file name --------------------------
def extract_datetime_from_stem(stem: str) -> Optional[str]:
    """
    Example:
      'T20241018_time_18_tracking_analysis' -> '20241018_time_18'
    """
    m = re.search(r"T(\d{8}_time_\d+)", stem)
    if m:
        return m.group(1)
    return None

# ============================ SUMMARY HELPERS ===============================

def _weighted_summary_from_window_means(win_df: pd.DataFrame, mean_col: str) -> Dict[str, float]:
    """Compute weighted mean (by frames_non_skipped_window) and other stats from window *_mean column."""
    base = mean_col[:-5]  # strip "_mean"
    vals = pd.to_numeric(win_df[mean_col], errors="coerce")
    nvec = pd.to_numeric(win_df["frames_non_skipped_window"], errors="coerce").fillna(0)
    if vals.notna().any() and nvec.sum() > 0:
        w = nvec.where(vals.notna(), 0)
        mean_all = float((vals.fillna(0) * w).sum() / max(1, w.sum()))
    else:
        mean_all = np.nan
    med_all = float(np.nanmedian(vals)) if vals.notna().any() else np.nan
    std_all = float(np.nanstd(vals)) if vals.notna().any() else np.nan
    min_all = float(np.nanmin(vals)) if vals.notna().any() else np.nan
    max_all = float(np.nanmax(vals)) if vals.notna().any() else np.nan
    valid_ratio = float(vals.notna().mean()) if len(vals) > 0 else np.nan
    return {
        f"{base}_mean_all": mean_all,
        f"{base}_median_all": med_all,
        f"{base}_std_all": std_all,
        f"{base}_min_all": min_all,
        f"{base}_max_all": max_all,
        f"{base}_valid_ratio": valid_ratio,
    }

def compute_file_summary(
    win_df: pd.DataFrame,
    file_base_short: str,
    n_frames_in_file: int,
    file_skipped_frames: int,
    df_original: pd.DataFrame,
    skip_mask: pd.Series,
    fps: float
) -> pd.DataFrame:
    tails = ["tail1_hz", "tail2_hz", "tail3_hz", "tail4_hz"]
    metrics = {
        "file": file_base_short,
        # NEW: Datetime parsed from 'file'
        "Datetime": extract_datetime_from_stem(file_base_short),
        "fps": float(win_df["fps"].iloc[0]) if "fps" in win_df else FPS,
        "window_seconds": float(win_df["window_seconds"].iloc[0]) if "window_seconds" in win_df else WINDOW_SECONDS,
        "n_windows": int(len(win_df)),
        "n_frames": int(n_frames_in_file),
        "skipped_frames": int(file_skipped_frames),
        "skipped_rate_pct": float(file_skipped_frames / n_frames_in_file * 100.0) if n_frames_in_file > 0 else 0.0,
    }
    for t in tails:
        metrics.update(describe_block(win_df, t))
    metrics.update(describe_block(win_df, "freq_mean_hz"))

    # tail frequency share %
    tail_means = [metrics.get(f"{t}_mean", np.nan) for t in tails]
    means_arr = np.array([m for m in tail_means if np.isfinite(m)], dtype=float)
    denom = float(means_arr.sum()) if means_arr.size > 0 else np.nan
    for i, t in enumerate(tails, start=1):
        m = metrics.get(f"{t}_mean", np.nan)
        metrics[f"tail{i}_freq_pct"] = float(m / denom * 100.0) if (np.isfinite(m) and np.isfinite(denom) and denom > 0) else np.nan

    # curvature overall from window means (existing)
    curv_mean_cols = [c for c in win_df.columns if re.match(r"^curvature_\d+_mm_mean$", c)]
    for cm in curv_mean_cols:
        metrics.update(_weighted_summary_from_window_means(win_df, cm))

    # NEW: curvature radius & length overall from window means
    radius_mean_cols = [c for c in win_df.columns if re.match(r"^curvature_\d+_radius_(px|mm)_mean$", c)]
    length_mean_cols = [c for c in win_df.columns if re.match(r"^(mfc_length|body_length)_(px|mm)_mean$", c)]
    for cm in radius_mean_cols + length_mean_cols:
        metrics.update(_weighted_summary_from_window_means(win_df, cm))

    # bending (curvature_1_class)
    bd = bending_distribution_curv1(df_original, skip_mask)
    metrics.update({k: v for k, v in bd.items() if not k.startswith("count_")})

    # distance totals (existing + NEW because they start with "sum_")
    for c in win_df.columns:
        if c.startswith("sum_"):
            metrics[f"file_{c}"] = float(pd.to_numeric(win_df[c], errors="coerce").fillna(0).sum())

    # --- NEW: Average speed (BL/s) calculations ---
    # Duration is computed using non-skipped frames only
    non_skipped_frames = max(0, n_frames_in_file - file_skipped_frames)
    duration_s = (non_skipped_frames / float(fps)) if (fps and fps > 0) else np.nan
    duration_s = duration_s if duration_s and duration_s > 0 else np.nan

    # For frame distances (mm)
    if np.isfinite(duration_s):
        # Average speed of X(BL/s) for Head/MFC/Tail frame distances in mm
        patt_fd = re.compile(r"^file_sum_(Head\d+|MFC\d+|Tail\d+)_frame_dist_mm$")
        for k in list(metrics.keys()):
            m = patt_fd.match(k)
            if m and np.isfinite(metrics.get(k, np.nan)):
                part = m.group(1)
                speed_bls = (metrics[k] / duration_s) / BL_MM if BL_MM > 0 else np.nan
                metrics[f"Average speed of {part}(BL/s)"] = float(speed_bls)

        # Average speed(reference line) for Tail travel distances in mm
        patt_travel = re.compile(r"^file_sum_(Tail\d+)_travel_distance_mm$")
        for k in list(metrics.keys()):
            m = patt_travel.match(k)
            if m and np.isfinite(metrics.get(k, np.nan)):
                part = m.group(1)
                speed_bls = (metrics[k] / duration_s) / BL_MM if BL_MM > 0 else np.nan
                metrics[f"Average speed(reference line) of {part}(BL/s)"] = float(speed_bls)

    # Max Hz window
    freq_series = pd.to_numeric(win_df["freq_mean_hz"], errors="coerce")
    if freq_series.notna().any():
        idx = int(freq_series.idxmax())
        row = win_df.loc[idx]
        metrics["max_freq_window_idx"] = int(row["window_idx"])
        metrics["max_freq_hz"] = float(row["freq_mean_hz"])
        metrics["max_freq_start_frame"] = int(row["start_frame"])
        metrics["max_freq_end_frame"] = int(row["end_frame"])
        metrics["max_freq_timestamp_s"] = float(row["mid_time_s"])
        mmss = seconds_to_mmss(metrics["max_freq_timestamp_s"])
        metrics["max_freq_timestamp_mmss"] = mmss if mmss is not None else np.nan
    else:
        metrics["max_freq_window_idx"] = np.nan
        metrics["max_freq_hz"] = np.nan
        metrics["max_freq_start_frame"] = np.nan
        metrics["max_freq_end_frame"] = np.nan
        metrics["max_freq_timestamp_s"] = np.nan
        metrics["max_freq_timestamp_mmss"] = np.nan

    # Max curvature frame (prefer curvature_1_mm)
    ref_curv_for_peak = pick_curvature_for_peak(df_original, None)
    if ref_curv_for_peak and ref_curv_for_peak in df_original.columns:
        s = pd.to_numeric(df_original[ref_curv_for_peak], errors="coerce")
        if s.notna().any():
            peak_pos = int(np.nanargmax(s.values))
            peak_idx = int(s.index[peak_pos])
            metrics["max_curvature_column"] = ref_curv_for_peak
            metrics["max_curvature_value_mm"] = float(s.iloc[peak_pos])
            metrics["max_curvature_frame_idx"] = peak_idx
            metrics["max_curvature_timestamp_s"] = float(peak_idx / fps)
            mmss = seconds_to_mmss(metrics["max_curvature_timestamp_s"])
            metrics["max_curvature_timestamp_mmss"] = mmss if mmss is not None else np.nan
        else:
            metrics["max_curvature_column"] = ref_curv_for_peak
            metrics["max_curvature_value_mm"] = np.nan
            metrics["max_curvature_frame_idx"] = np.nan
            metrics["max_curvature_timestamp_s"] = np.nan
            metrics["max_curvature_timestamp_mmss"] = np.nan
    else:
        metrics["max_curvature_column"] = np.nan
        metrics["max_curvature_value_mm"] = np.nan
        metrics["max_curvature_frame_idx"] = np.nan
        metrics["max_curvature_timestamp_s"] = np.nan
        metrics["max_curvature_timestamp_mmss"] = np.nan

    return pd.DataFrame([metrics])

# ============================ WORKER (per file) =============================

def process_one_file(args: Tuple[str, str, str, str, float, float]) -> Dict:
    p_str, per_file_dir_str, result_root_str, log_dir_str, fps, win_secs = args
    p = Path(p_str)
    per_file_dir = Path(per_file_dir_str)

    df = safe_read_csv(p)
    if df is None or not has_required_signals(df):
        return {"ok": False, "path": p_str, "result_root": result_root_str}

    ref_curv = pick_reference_curvature(df)
    skip_mask = detect_skipped_frames_by_curvature(df, ref_curv)

    file_skipped_frames = int(skip_mask.sum())
    n_frames = len(df)

    win_df = compute_window_table(
        df=df,
        source_path=p,
        fps=fps,
        window_seconds=win_secs,
        skip_mask=skip_mask
    )
    if win_df.empty:
        return {"ok": False, "path": p_str, "result_root": result_root_str}

    base_short = simplify_perfile_stem(p.stem)
    win_csv = per_file_dir / f"{base_short}_swim_freq_curv_O_skipped_frame.csv"

    win_df_fmt = round_and_cast(win_df, is_summary=False)
    win_df_fmt.to_csv(win_csv, index=False, encoding="utf-8-sig")

    summ_df = compute_file_summary(
        win_df=win_df,
        file_base_short=base_short,
        n_frames_in_file=n_frames,
        file_skipped_frames=file_skipped_frames,
        df_original=df,
        skip_mask=skip_mask,
        fps=fps
    )
    summ_df_fmt = round_and_cast(summ_df, is_summary=True)
    sum_csv = per_file_dir / f"{base_short}_swim_freq_curv_summary_O_skipped_frame.csv"
    summ_df_fmt.to_csv(sum_csv, index=False, encoding="utf-8-sig")

    bd = bending_distribution_curv1(df, skip_mask)
    bending_counts = {
        "minimal": int(bd.get("count_minimal", 0)),
        "normal":  int(bd.get("count_normal", 0)),
        "extreme": int(bd.get("count_extreme", 0)),
        "valid":   int(bd.get("bending_total_valid_frames", 0)),
        "considered": int(bd.get("bending_total_considered_frames", 0)),
        "invalid": int(bd.get("count_invalid", 0))
    }

    return {
        "ok": True,
        "path": p_str,
        "result_root": result_root_str,
        "win_csv": str(win_csv),
        "sum_csv": str(sum_csv),
        "bending_counts": bending_counts
    }

# ================================ MAIN =====================================

def main():
    # Roots
    if ROOT_OVERRIDES:
        roots = [Path(r).expanduser().resolve() for r in ROOT_OVERRIDES]
    else:
        roots = [resolve_home_from_env().resolve()]

    if not roots:
        print("No roots to sweep.")
        return

    def result_dir_for(root: Path) -> Path:
        return root / f"{root.name}_{RESULT_DIRNAME}"

    # Logger under first root
    first_root = roots[0]
    first_result_root = result_dir_for(first_root)
    first_log_dir = first_result_root / LOG_DIRNAME
    first_result_root.mkdir(parents=True, exist_ok=True)
    first_log_dir.mkdir(parents=True, exist_ok=True)
    logger = make_logger(first_log_dir)

    logger.info("Sweeping ALL roots (non-recursive):")
    for r in roots:
        logger.info(f"  - {r}")

    logger.info(f"Top-level pattern: {FILE_GLOB_TOPLEVEL} (non-recursive)")
    logger.info(f"Include substring: {FILE_INCLUDE_SUBSTRING!r}")
    logger.info(f"FPS={FPS}, window_seconds={WINDOW_SECONDS}")
    logger.info(f"Multiprocessing: {USE_MULTIPROCESSING} (CPU_USAGE={CPU_USAGE})")
    logger.info(f"Reference curvature for skip: {REFERENCE_CURVATURE_FOR_SKIP}")
    logger.info(f"Body length for BL/s (mm): {BL_MM}")

    # Discover tasks, and prepare per-root containers
    tasks: List[Tuple[str, str, str, str, float, float]] = []
    root_result_dirs: Dict[str, Path] = {}  # root path str -> result dir Path
    for root in roots:
        result_root = result_dir_for(root)
        per_file_dir = result_root / PER_FILE_DIRNAME
        log_dir = result_root / LOG_DIRNAME
        result_root.mkdir(parents=True, exist_ok=True)
        per_file_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        root_result_dirs[str(result_root)] = result_root

        all_candidates = sorted([str(p) for p in root.glob(FILE_GLOB_TOPLEVEL)])
        logger.info(f"[{root}] {len(all_candidates)} CSV candidates in top-level.")

        if FILE_INCLUDE_SUBSTRING:
            inc = FILE_INCLUDE_SUBSTRING.lower()
            csvs = [p for p in all_candidates if inc in Path(p).name.lower()]
            logger.info(f"[{root}] {len(csvs)} match include substring {FILE_INCLUDE_SUBSTRING!r} in filename.")
        else:
            csvs = all_candidates
            logger.info(f"[{root}] include substring disabled; using all top-level candidates.")

        for p in csvs:
            tasks.append((p, str(per_file_dir), str(result_root), str(log_dir), float(FPS), float(WINDOW_SECONDS)))

    if not tasks:
        logger.warning("No CSVs to process after filtering.")
        print("\n=== Done (no tasks) ===")
        return

    # Run workers
    if USE_MULTIPROCESSING and CPU_USAGE > 0:
        procs = max(1, int(math.floor(cpu_count() * CPU_USAGE)))
    else:
        procs = 1

    results: List[Dict] = []
    if procs == 1:
        it = tqdm(tasks, total=len(tasks), desc="Processing files", unit="file") if SHOW_PROGRESS else tasks
        for t in it:
            results.append(process_one_file(t))
    else:
        with Pool(processes=procs) as pool:
            imap = pool.imap_unordered(process_one_file, tasks, chunksize=1)
            if SHOW_PROGRESS:
                for r in tqdm(imap, total=len(tasks), desc=f"Processing files (p={procs})", unit="file"):
                    results.append(r)
            else:
                for r in imap:
                    results.append(r)

    # Aggregate per-root artifacts
    per_root_win: Dict[str, List[str]] = {}
    per_root_sum: Dict[str, List[str]] = {}
    per_root_bend: Dict[str, Dict[str, int]] = {}
    grand_win: List[str] = []
    grand_sum: List[str] = []
    grand_bend = {"minimal": 0, "normal": 0, "extreme": 0, "valid": 0, "considered": 0, "invalid": 0}

    processed = 0
    for r in results:
        rr = r.get("result_root")
        if rr is None:
            continue
        per_root_win.setdefault(rr, [])
        per_root_sum.setdefault(rr, [])
        per_root_bend.setdefault(rr, {"minimal": 0, "normal": 0, "extreme": 0, "valid": 0, "considered": 0, "invalid": 0})

        if not r.get("ok", False):
            continue
        processed += 1
        if r.get("win_csv"):
            per_root_win[rr].append(r["win_csv"])
            grand_win.append(r["win_csv"])
        if r.get("sum_csv"):
            per_root_sum[rr].append(r["sum_csv"])
            grand_sum.append(r["sum_csv"])
        bc = r.get("bending_counts", {})
        for k in per_root_bend[rr]:
            per_root_bend[rr][k] += int(bc.get(k, 0))
            grand_bend[k] += int(bc.get(k, 0))

    logger.info(f"Processed {processed} files with usable signals.")

    # ---- helper to build integrated outputs from lists of CSVs ----
    def build_integrated_outputs(win_csv_paths: List[str], sum_csv_paths: List[str], bending_totals: Dict[str, int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        total_win_list = []
        it = tqdm(win_csv_paths, total=len(win_csv_paths), desc="Loading per-file windows [O]", unit="file") if SHOW_PROGRESS else win_csv_paths
        for wpath in it:
            try:
                wdf = pd.read_csv(wpath, encoding="utf-8-sig")
                total_win_list.append(wdf)
            except Exception:
                continue

        freq_df = pd.DataFrame()
        curv_df = pd.DataFrame()
        series_df = pd.DataFrame()  # NEW global-series table
        bend_df = pd.DataFrame()

        if total_win_list:
            total_win = pd.concat(total_win_list, ignore_index=True)

            def block_stats(col: str) -> Dict[str, float]:
                x = pd.to_numeric(total_win[col], errors="coerce").dropna().values
                if x.size == 0:
                    return {f"{col}_mean": np.nan, f"{col}_median": np.nan, f"{col}_std": np.nan,
                            f"{col}_min": np.nan, f"{col}_max": np.nan, f"{col}_p10": np.nan, f"{col}_p90": np.nan,
                            f"{col}_nonzero_ratio": np.nan}
                return {
                    f"{col}_mean": float(np.mean(x)),
                    f"{col}_median": float(np.median(x)),
                    f"{col}_std": float(np.std(x)),
                    f"{col}_min": float(np.min(x)),
                    f"{col}_max": float(np.max(x)),
                    f"{col}_p10": float(np.percentile(x, 10)),
                    f"{col}_p90": float(np.percentile(x, 90)),
                    f"{col}_nonzero_ratio": float(np.mean(x > 0.0)),
                }

            total_frames_est = int(pd.to_numeric(total_win["frames_in_window"], errors="coerce").fillna(0).sum())
            total_skipped_frames = int(pd.to_numeric(total_win["frames_skipped_window"], errors="coerce").fillna(0).sum())
            total_skipped_rate_pct = (total_skipped_frames / total_frames_est * 100.0) if total_frames_est > 0 else 0.0

            tails_cols = ["tail1_hz", "tail2_hz", "tail3_hz", "tail4_hz"]
            tail_means = {}
            for i, col in enumerate(tails_cols, start=1):
                tail_means[i] = pd.to_numeric(total_win[col], errors="coerce").dropna().mean()
            denom = float(np.nansum(list(tail_means.values()))) if len(tail_means) > 0 else np.nan
            tail_share = {}
            for i in range(1, 4+1):
                m = tail_means.get(i, np.nan)
                tail_share[f"tail{i}_freq_pct"] = float(m / denom * 100.0) if (np.isfinite(m) and np.isfinite(denom) and denom > 0) else np.nan

            freq_section = {
                "section": "global_frequency",
                "variant": "O",
                "total_windows": int(len(total_win)),
                "total_frames_est": total_frames_est,
                "total_skipped_frames": total_skipped_frames,
                "total_skipped_rate_pct": total_skipped_rate_pct,
            }
            for col in tails_cols + ["freq_mean_hz"]:
                freq_section.update(block_stats(col))

            freq_df = round_and_cast(pd.DataFrame([freq_section]), is_summary=False)

            # Curvature weighted by frames_non_skipped_window (existing)
            curv_mean_cols = [c for c in total_win.columns if re.match(r"^curvature_\d+_mm_mean$", c)]
            curv_rows = []
            for cm in curv_mean_cols:
                base = cm[:-5]
                vals = pd.to_numeric(total_win[cm], errors="coerce")
                nvec = pd.to_numeric(total_win["frames_non_skipped_window"], errors="coerce").fillna(0)
                if vals.notna().any() and nvec.sum() > 0:
                    w = nvec.where(vals.notna(), 0)
                    mean_all = float((vals.fillna(0) * w).sum() / max(1, w.sum()))
                else:
                    mean_all = np.nan
                med_all = float(np.nanmedian(vals)) if vals.notna().any() else np.nan
                std_all = float(np.nanstd(vals)) if vals.notna().any() else np.nan
                min_all = float(np.nanmin(vals)) if vals.notna().any() else np.nan
                max_all = float(np.nanmax(vals)) if vals.notna().any() else np.nan
                valid_ratio = float(vals.notna().mean()) if len(vals) > 0 else np.nan
                curv_rows.append({
                    "section": "global_curvature",
                    "variant": "O",
                    "curvature": base.replace("_mm", ""),
                    "mean": mean_all, "median": med_all, "std": std_all,
                    "min": min_all, "max": max_all, "valid_ratio": valid_ratio,
                    "n_valid": int(vals.notna().sum())
                })
            curv_df = round_and_cast(pd.DataFrame(curv_rows), is_summary=False) if curv_rows else pd.DataFrame()

            # NEW: global series for curvature radii & lengths (weighted)
            series_mean_cols = []
            series_mean_cols += [c for c in total_win.columns if re.match(r"^curvature_\d+_radius_(px|mm)_mean$", c)]
            series_mean_cols += [c for c in total_win.columns if re.match(r"^(mfc_length|body_length)_(px|mm)_mean$", c)]

            series_rows = []
            for cm in series_mean_cols:
                base = cm[:-5]
                vals = pd.to_numeric(total_win[cm], errors="coerce")
                nvec = pd.to_numeric(total_win["frames_non_skipped_window"], errors="coerce").fillna(0)
                if vals.notna().any() and nvec.sum() > 0:
                    w = nvec.where(vals.notna(), 0)
                    mean_all = float((vals.fillna(0) * w).sum() / max(1, w.sum()))
                else:
                    mean_all = np.nan
                med_all = float(np.nanmedian(vals)) if vals.notna().any() else np.nan
                std_all = float(np.nanstd(vals)) if vals.notna().any() else np.nan
                min_all = float(np.nanmin(vals)) if vals.notna().any() else np.nan
                max_all = float(np.nanmax(vals)) if vals.notna().any() else np.nan
                valid_ratio = float(vals.notna().mean()) if len(vals) > 0 else np.nan
                series_rows.append({
                    "section": "global_series",
                    "variant": "O",
                    "series": base,   # keep the full series name
                    "mean": mean_all, "median": med_all, "std": std_all,
                    "min": min_all, "max": max_all, "valid_ratio": valid_ratio,
                    "n_valid": int(vals.notna().sum())
                })
            series_df = round_and_cast(pd.DataFrame(series_rows), is_summary=False) if series_rows else pd.DataFrame()

            # Bending distribution integrated via direct counts
            considered = bending_totals["considered"]
            valid = bending_totals["valid"]
            invalid = bending_totals["invalid"]
            minimal = bending_totals["minimal"]
            normal  = bending_totals["normal"]
            extreme = bending_totals["extreme"]

            if considered > 0:
                bend_min_pct = minimal / considered * 100.0
                bend_norm_pct = normal  / considered * 100.0
                bend_ext_pct = extreme / considered * 100.0
                bend_inv_pct = invalid  / considered * 100.0
            else:
                bend_min_pct = bend_norm_pct = bend_ext_pct = bend_inv_pct = np.nan

            bend_df = round_and_cast(pd.DataFrame([{
                "section": "global_bending_distribution",
                "variant": "O",
                "curvature_1_minimal_percentage": bend_min_pct,
                "curvature_1_normal_percentage":  bend_norm_pct,
                "curvature_1_extreme_percentage": bend_ext_pct,
                "curvature_1_invalid_percentage": bend_inv_pct,
                "bending_total_valid_frames": valid,
                "bending_total_considered_frames": considered
            }]), is_summary=False)

        # Integrated per-file summaries
        total_summ_list = []
        it2 = tqdm(sum_csv_paths, total=len(sum_csv_paths), desc="Loading per-file summaries", unit="file") if SHOW_PROGRESS else sum_csv_paths
        for spath in it2:
            try:
                sdf = pd.read_csv(spath, encoding="utf-8-sig")
                total_summ_list.append(sdf)
            except Exception:
                continue
        integrated_fmt = round_and_cast(pd.concat(total_summ_list, ignore_index=True), is_summary=True) if total_summ_list else pd.DataFrame()

        # Concatenate stats tables (order preserved; leaves original ones intact)
        pieces = [df for df in [freq_df, curv_df, series_df, bend_df] if not df.empty]
        total_stats = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
        return total_stats, integrated_fmt

    # ---- Build per-root integrated outputs ----
    for rr, result_root in root_result_dirs.items():
        stats_df, integ_df = build_integrated_outputs(per_root_win.get(rr, []), per_root_sum.get(rr, []), per_root_bend.get(rr, {"minimal":0,"normal":0,"extreme":0,"valid":0,"considered":0,"invalid":0}))
        if not stats_df.empty:
            p = result_root / "Total_summary_swim_freq_curv_stats_O_skipped_frame.csv"
            stats_df.to_csv(p, index=False, encoding="utf-8-sig")
            logger.info(f"[Per-root] Integrated stats → {p}")
        if not integ_df.empty:
            p2 = result_root / "Total_integrated_summary.csv"
            integ_df.to_csv(p2, index=False, encoding="utf-8-sig")
            logger.info(f"[Per-root] Integrated summaries → {p2}")

    # ---- Grand Total across ALL roots (optional, under first root) ----
    grand_stats_df, grand_integ_df = build_integrated_outputs(grand_win, grand_sum, grand_bend)
    if not grand_stats_df.empty:
        gp = first_result_root / "Grand_Total_summary_swim_freq_curv_stats_O_skipped_frame.csv"
        grand_stats_df.to_csv(gp, index=False, encoding="utf-8-sig")
        logger.info(f"[Grand] Integrated stats → {gp}")
    if not grand_integ_df.empty:
        gp2 = first_result_root / "Grand_Total_integrated_summary.csv"
        grand_integ_df.to_csv(gp2, index=False, encoding="utf-8-sig")
        logger.info(f"[Grand] Integrated summaries → {gp2}")

    print("\n=== Done ===")
    print("Per-file outputs: each root’s '<root.name>_swim_freq_curv_results/per_file/'.")
    print("Per-root integrated outputs: each root’s '<root.name>_swim_freq_curv_results/'.")
    print(f"Grand-total outputs: {first_result_root}")

if __name__ == "__main__":
    main()
