# -*- coding: utf-8 -*-
"""
Fish Pose-Tracking Analysis (Step 1)

Per-frame analysis of DeepLabCut (DLC) pose-estimation output CSVs.
For each video's CSV the script computes:
  - Cumulative body-part travel distances (px and mm).
  - Spinal curvature via algebraic circle fitting (Pratt, HyperLS, or
    three-point methods) for user-defined keypoint subsets.
  - Bending classification (Minimal / Normal / Extreme / Invalid).
  - Tail-beat (swimming stroke) counts and travel distances relative
    to a head–body reference line.
  - Body and mid-body segment lengths per frame.
  - Per-video summary statistics exported as CSV.

Features
--------
- Multiprocessing-safe: uses ``spawn`` context to avoid BLAS/OpenCV
  thread contention.  A background thread monitors per-file progress.
- Multi-base: can process multiple input directories sequentially in a
  single run, each producing its own output folder.

Input
-----
Each ``BASE_PATH`` directory must contain DLC inference CSV files
(one per video).  These CSVs are the standard DLC output with a
multi-level header: ``scorer / bodypart / x|y|likelihood``.

Output
------
Under each base path:
  ``<timestamp>_result_csv/``  — per-video tracking analysis CSVs and
                                 summaries, plus a ``Total_summary.csv``.
  ``<timestamp>_result_log/``  — per-video processing logs.
"""

import os
import re
import cv2
import time
import math
import logging
import warnings
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from circle_fit import prattSVD as _prattSVD
from circle_fit import hyperSVD as _hyperSVD
from circle_fit import hyperLSQ as _hyperLSQ
from multiprocessing import cpu_count, get_context
import queue
import threading
import signal
from typing import Dict, List, Tuple, Any

# ----------------------------------------------------------------------------
# Constants and Configuration (Global Parameters)
# ----------------------------------------------------------------------------
# Likelihood threshold for body part detection
LIKELIHOOD_THRESHOLD = 0.6

# Default frames per second (FPS) for video
DEFAULT_FPS = 20

# ----------------------------------------------------------------------------
# MULTI-BASE CHANGE: Provide one or more base paths here.
#   - You can keep a single item in the list if you want the old behavior.
#   - Each base path will be processed independently with its own outputs.
# ----------------------------------------------------------------------------
# Each path should point to a directory that contains DLC inference CSV files
# (the standard DLC output with multi-level header: scorer/bodypart/x|y|likelihood).
# These CSVs are typically generated alongside the source videos after running
# deeplabcut.analyze_videos() with save_as_csv=True (see dlc_inference.py).
BASE_PATHS = [
    "<DLC_OUTPUT_DIR>",
    # Example: "/path/to/experiment_01/inference_results",
    # Add more paths to process multiple sessions in one run:
    # "/path/to/experiment_02/inference_results",
]

# List of all body parts as defined in the DeepLabCut (DLC) model output
body_part_names = [
    "Head1", "Head2",
    "MFC1", "MFC2", "MFC3", "MFC4", "MFC5",
    "Tail1", "Tail2", "Tail3", "Tail4"
]

# Body parts that will be specifically tracked for distance calculations
TRACKED_BODY_PARTS = [
    "Head1", "Head2",
    "MFC1", "MFC2", "MFC3", "MFC4", "MFC5",
    "Tail1", "Tail2", "Tail3", "Tail4"
]

# Conversion rate from pixels (px) to millimeters (mm)
conversion_rate_mm_per_px = 2.0

# Jitter threshold: distances below this (px) are considered negligible
JITTER_THRESHOLD = 1

# Threshold for detecting "wrong detection" by sudden jumps
WRONG_DETECT_THRESHOLD_PX = 200

# Different thresholds for different tail parts
SWIMMING_JITTER_THRESHOLDS = {
    'Tail1': 3,
    'Tail2': 5,
    'Tail3': 9,
    'Tail4': 15,
}

# Number of parallel processes to use for batch processing
desired_cpu_percentage = 0.5
MAX_PROCESSES = max(1, math.floor(cpu_count() * desired_cpu_percentage))

# Body length filtering limits for visualization
UPPER_BODY_LENGTH_LIMIT = 670  # mm
LOWER_BODY_LENGTH_LIMIT = 270  # mm

# ----------------------------------------------------------------------------
# Curvature Classification Thresholds
# ----------------------------------------------------------------------------
EXTREME_BENDING_THRESHOLD = 0.01     # mm^-1
NORMAL_BENDING_THRESHOLD = 0.0033    # mm^-1
INVALID_CURVATURE_THRESHOLD = 0.02   # mm^-1

# ----------------------------------------------------------------------------
# Curvature and Swimming Count Configuration
# ----------------------------------------------------------------------------
curvature_part_names_1 = [
    "MFC1", "MFC2", "MFC3", "MFC4", "MFC5",
]

curvature_part_names_2 = [
    "Head2", "MFC1", "MFC2", "MFC3", "MFC4", "MFC5", "Tail1",
]

curvature_part_names_3 = [
    "Head1", "Head2",
    "MFC1", "MFC2", "MFC3", "MFC4", "MFC5",
    "Tail1", "Tail2", "Tail3", "Tail4"
]

# Reference line configuration
REFERENCE_START = 'Head1'          # must exist (no fallback)
REFERENCE_END_PRIMARY = 'Head2'    # preferred end point
REFERENCE_END_FALLBACK = 'MFC1'    # fallback if Head2 is missing

# Build global curvature_sets for top-level use
curvature_sets = []
if curvature_part_names_1:
    curvature_sets.append({'name': 'curvature_1',
                           'parts': curvature_part_names_1})
if curvature_part_names_2:
    curvature_sets.append({'name': 'curvature_2',
                           'parts': curvature_part_names_2})
if curvature_part_names_3:
    curvature_sets.append({'name': 'curvature_3',
                           'parts': curvature_part_names_3})

# Minimum number of points required for curvature fit (80% of points for each set)
CURVATURE_MIN_POINTS_DICT = {
    'curvature_1': 4,  # 80% of 5 points
    'curvature_2': 6,  # 80% of 7 points
    'curvature_3': 9,  # 80% of 11 points
}

# Default circular fitting method ('hyper', 'pratt', 'threept')
DEFAULT_CIRCULAR_FIT_METHOD = 'hyper'  # HyperLS as default

# Curvature length bounds (per set; None/None keeps original behavior)
curvature_length_bounds = [
    (None, None),
    (None, None),
    (None, None),
]

# Define body parts for swimming count (tail-beat) analysis
swimming_count_part_names = [
    "Tail1", "Tail2", "Tail3", "Tail4", "MFC5"
]

# Body parts sequences for length calculation
MFC_LENGTH_SEQUENCE = ["MFC1", "MFC2", "MFC3", "MFC4", "MFC5"]
BODY_LENGTH_SEQUENCE = ["Head1", "Head2", "MFC1", "MFC2", "MFC3", "MFC4", "MFC5", "Tail1", "Tail2", "Tail3", "Tail4"]

# ----------------------------------------------------------------------------
# GLOBAL timestamp (shared across all base paths for this batch run)
# ----------------------------------------------------------------------------
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

# ----------------------------------------------------------------------------
# Utility & helpers
# ----------------------------------------------------------------------------
def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def normalize_video_basename(basename: str) -> str:
    """Expand 'TYYMMDD_' → 'TYYYYMMDD_' exactly once at the start of a basename."""
    if re.match(r'^T20\d{6}_', basename):
        return basename
    return re.sub(r'^T(\d{6})_', r'T20\1_', basename)

def truncate_filename(filename: str) -> str:
    if '_mergedDLC_' in filename:
        return filename.split('_mergedDLC_')[0]
    elif 'DLC_' in filename:
        return filename.split('DLC_')[0]
    else:
        return filename

def distance_2d(p1, p2):
    if p1 is None or p2 is None:
        return np.nan
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def apply_value_thresholds_and_rounding(value, column_name, data_type='numeric'):
    if pd.isna(value):
        return np.nan
    name = column_name.lower()
    # radius first
    if 'radius_px' in name or 'radius_mm' in name:
        return int(round(value))
    # pure curvature (exclude radius/centers/class/shape counters)
    if (name.startswith('curvature_') and name.endswith('_mm') and
        not any(k in name for k in ['radius','center','class','u_shape','n_shape','percentage','count'])):
        if abs(value) > INVALID_CURVATURE_THRESHOLD:
            return np.nan
        return round(value, 4)
    if (name.startswith('curvature_') and name.endswith('_px') and
        not any(k in name for k in ['radius','center','class','u_shape','n_shape','percentage','count'])):
        return round(value, 4)
    # lengths/distances
    if 'body_length' in name and name.endswith('_mm'):
        if value < LOWER_BODY_LENGTH_LIMIT or value > UPPER_BODY_LENGTH_LIMIT:
            return np.nan
        return int(round(value))
    if 'mfc_length' in name and name.endswith('_mm'):
        return int(round(value))
    if 'distance' in name and name.endswith('_mm'):
        return int(round(value))
    if 'distance' in name and name.endswith('_px'):
        return int(round(value))
    # generic mm/px
    if name.endswith('_mm') and 'curvature' not in name and 'radius' not in name:
        return int(round(value))
    if name.endswith('_px') and 'curvature' not in name and 'radius' not in name:
        return int(round(value))
    # coords
    if name.endswith('_x') or name.endswith('_y') or 'center_x' in name or 'center_y' in name:
        return round(value, 3)
    # likelihood
    if 'likelihood' in name:
        return round(value, 3)
    # counts
    if any(keyword in name for keyword in ['count','swim','bending']):
        return int(round(value))
    # percentages/frequencies
    if 'percentage' in name:
        return round(value, 2)
    if 'frequency_hz' in name:
        return round(value, 4)
    # frame counts
    if name in ['total_frames','skipped_frames_missing_invalid','skipped_frames_wrong_detection']:
        return int(round(value))
    if isinstance(value, (int,float)):
        return int(value) if value == int(value) else round(value, 3)
    return value

def apply_dataframe_rounding_and_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    for column in df_copy.columns:
        if df_copy[column].dtype in ['float64','float32','int64','int32']:
            if 'frequency_hz' in column:
                df_copy[column] = df_copy[column].apply(lambda x: round(x,4) if pd.notna(x) else np.nan).astype('float64')
            else:
                df_copy[column] = df_copy[column].apply(lambda x: apply_value_thresholds_and_rounding(x, column))
    return df_copy

def extract_coordinates(data_row, mapping, parts_to_extract):
    coords = {}
    for part in parts_to_extract:
        if part in mapping:
            try:
                x = data_row[mapping[part][0]]
                y = data_row[mapping[part][1]]
                likelihood = data_row[mapping[part][2]]
                if not pd.isna(x) and not pd.isna(y) and likelihood >= LIKELIHOOD_THRESHOLD:
                    coords[part] = (x, y)
                else:
                    coords[part] = None
            except (KeyError, IndexError):
                coords[part] = None
        else:
            coords[part] = None
    return coords

def calculate_sequential_length(coords_dict, sequence):
    for part in sequence:
        if part not in coords_dict or coords_dict[part] is None:
            return None
    total_length = 0
    for i in range(len(sequence) - 1):
        dist = distance_2d(coords_dict[sequence[i]], coords_dict[sequence[i+1]])
        if dist is None or np.isnan(dist):
            return None
        total_length += dist
    return total_length

def parse_sign_value(sign_str):
    if not sign_str or pd.isna(sign_str):
        return 0
    sign_str = str(sign_str)
    try:
        return float(sign_str)
    except ValueError:
        parts = sign_str.split(':')
        if len(parts) == 2:
            return float(parts[1].strip())
        return 0

def classify_curvature(curvature_mm):
    if pd.isna(curvature_mm):
        return None
    if abs(curvature_mm) >= INVALID_CURVATURE_THRESHOLD:
        return 'Invalid'
    elif abs(curvature_mm) >= EXTREME_BENDING_THRESHOLD:
        return 'Extreme'
    elif abs(curvature_mm) >= NORMAL_BENDING_THRESHOLD:
        return 'Normal'
    else:
        return 'Minimal'

def get_video_name_from_csv(csv_filename: str) -> str:
    base = os.path.splitext(csv_filename)[0]
    if base.endswith('_filtered'):
        base = base[:-9]
    return normalize_video_basename(base) + '.mp4'

def determine_shape_type(center_x, center_y, head_coord, tailbase_coord):
    if pd.isna(center_x) or pd.isna(center_y) or not head_coord or not tailbase_coord:
        return 'Unknown'
    v_ht = (tailbase_coord[0] - head_coord[0], tailbase_coord[1] - head_coord[1])
    v_hc = (center_x - head_coord[0], center_y - head_coord[1])
    cross_product = v_ht[0] * v_hc[1] - v_ht[1] * v_hc[0]
    if cross_product > 0:
        return 'U'
    elif cross_product < 0:
        return 'N'
    else:
        return 'Unknown'

def check_all_parts_likelihood(data_row, mapping, parts_list):
    for part in parts_list:
        if part in mapping:
            try:
                likelihood = data_row[mapping[part][2]]
                if pd.isna(likelihood) or likelihood < LIKELIHOOD_THRESHOLD:
                    return False
            except (KeyError, IndexError):
                return False
        else:
            return False
    return True

# ----------------------------------------------------------------------------
# Circular Fitting Methods
# ----------------------------------------------------------------------------
def fit_circle_pratt(points):
    P = np.asarray(points, float)
    if P.shape[0] < 3:
        raise ValueError("Pratt method requires at least 3 points.")
    xc, yc, r, _ = _prattSVD(P)
    if not (np.isfinite(r) and r > 0):
        raise ValueError("Pratt method returned invalid radius.")
    return 1.0 / r, r, xc, yc

def fit_circle_hyper(points):
    P = np.asarray(points, float)
    if P.shape[0] < 3:
        raise ValueError("HyperLS method requires at least 3 points.")
    if _hyperSVD is not None:
        xc, yc, r, _ = _hyperSVD(P)
    elif _hyperLSQ is not None:
        xc, yc, r, _ = _hyperLSQ(P)
    else:
        raise RuntimeError("No HyperLS routine available.")
    if not (np.isfinite(r) and r > 0):
        raise ValueError("HyperLS method returned invalid radius.")
    return 1.0 / r, r, xc, yc

def _circumcircle_from_3pts(p1, p2, p3):
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3
    a = x2 - x1; b = y2 - y1
    c = x3 - x1; d = y3 - y1
    e = a * (x1 + x2) + b * (y1 + y2)
    f = c * (x1 + x3) + d * (y1 + y3)
    g = 2.0 * (a * (y3 - y2) - b * (x3 - x2))
    if abs(g) < 1e-12:
        return np.nan, np.nan, np.nan, np.nan
    cx = (d * e - b * f) / g
    cy = (a * f - c * e) / g
    R = math.hypot(x1 - cx, y1 - cy)
    if not (np.isfinite(R) and R > 0):
        return np.nan, np.nan, np.nan, np.nan
    return 1.0 / R, R, cx, cy

def fit_circle_three_point(points, weights=None):
    P = np.asarray(points, float)
    n = len(P)
    if n < 3:
        return np.nan, np.nan, np.nan, np.nan
    if weights is None or len(weights) != n:
        W = np.ones(n, float)
    else:
        W = np.asarray(weights, float)
        W = np.clip(W, 1e-6, None)
    base_triplets = [(0, n // 2, n - 1)]
    best_score = -1.0
    best_res = (np.nan, np.nan, np.nan, np.nan)
    if n <= 12:
        iters = [(i, j, k) for i in range(n-2) for j in range(i+1, n-1) for k in range(j+1, n)]
    else:
        mids = np.linspace(1, n-2, 8, dtype=int)
        candidates = [0] + mids.tolist() + [n-1]
        iters = []
        for i in range(len(candidates) - 2):
            for j in range(i + 1, len(candidates) - 1):
                for k in range(j + 1, len(candidates)):
                    iters.append((candidates[i], candidates[j], candidates[k]))
    for i, j, k in iters + base_triplets:
        p1, p2, p3 = P[i], P[j], P[k]
        area2 = abs(np.cross(p2 - p1, p3 - p1))
        if area2 < 1e-6:
            continue
        score = area2 * (W[i] + W[j] + W[k])
        if score <= best_score:
            continue
        kappa, R, cx, cy = _circumcircle_from_3pts(p1, p2, p3)
        if np.isfinite(R) and R > 0:
            best_score = score
            best_res = (kappa, R, cx, cy)
    return best_res

def fit_circle_and_curvature(points, method='hyper', weights=None):
    if method == 'pratt':
        return fit_circle_pratt(points)
    elif method == 'hyper':
        return fit_circle_hyper(points)
    elif method == 'threept':
        return fit_circle_three_point(points, weights)
    else:
        raise ValueError(f"Unknown circle fit method: {method}")

# ----------------------------------------------------------------------------
# Worker init (limit per-process threads, disable OpenCL)
# ----------------------------------------------------------------------------
def init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    try:
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass
    # Make workers ignore KeyboardInterrupt; main will handle it
    signal.signal(signal.SIGINT, signal.SIG_IGN)

# ----------------------------------------------------------------------------
# Data Saving and Summary Generation
# ----------------------------------------------------------------------------
def save_results(data, video_basename, result_csv_path, logger):
    csv_file = os.path.join(result_csv_path, f"{video_basename}_tracking_analysis.csv")
    try:
        processed_data = apply_dataframe_rounding_and_thresholds(data)
        with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
            processed_data.to_csv(f, index=False)
        logger.info(f"Saved processed data for {video_basename} to CSV with proper rounding and thresholds.")
    except Exception as e:
        logger.error(f"Error saving processed results for {video_basename}: {e}")

def prepare_summary_data(data,
                         video_basename,
                         total_frames,
                         conversion_rate,
                         tracked_body_parts,
                         total_skipped_frames_missing_invalid,
                         total_skipped_frames_wrong_detection,
                         valid_curvature_sets,
                         result_csv_path,
                         logger,
                         mapping):
    summary_data = {
        'video_name': [video_basename],
        'total_frames': [total_frames],
        'skipped_frames_missing_invalid': [total_skipped_frames_missing_invalid],
        'skipped_frames_wrong_detection': [total_skipped_frames_wrong_detection]
    }

    # Tracked parts distances
    for part in tracked_body_parts:
        if f'{part}_cum_dist_px' in data.columns:
            cum_dist_px_series = data[f'{part}_cum_dist_px'].dropna()
            cum_dist_mm_series = data[f'{part}_cum_dist_mm'].dropna()
            cum_dist_px = cum_dist_px_series.iloc[-1] if len(cum_dist_px_series) > 0 else 0
            cum_dist_mm = cum_dist_mm_series.iloc[-1] if len(cum_dist_mm_series) > 0 else 0
            summary_data[f'{part}_total_distance_px'] = [cum_dist_px]
            summary_data[f'{part}_total_distance_mm'] = [cum_dist_mm]
        else:
            summary_data[f'{part}_total_distance_px'] = [0]
            summary_data[f'{part}_total_distance_mm'] = [0]

    # Curvatures
    for curv_set in valid_curvature_sets:
        set_name = curv_set['name']
        parts_list = curv_set['parts']
        valid_rows_mask = data.apply(lambda row: check_all_parts_likelihood(row, mapping, parts_list), axis=1)
        valid_data = data[valid_rows_mask]
        if f'{set_name}_px' in data.columns and len(valid_data) > 0:
            curv_px_col = valid_data[f'{set_name}_px'].dropna()
            curv_mm_col = valid_data[f'{set_name}_mm'].dropna()
            curv_px_col = curv_px_col[abs(curv_px_col / conversion_rate) <= INVALID_CURVATURE_THRESHOLD]
            curv_mm_col = curv_mm_col[abs(curv_mm_col) <= INVALID_CURVATURE_THRESHOLD]
            radius_px_col = valid_data[f'{set_name}_radius_px'].dropna()
            radius_mm_col = valid_data[f'{set_name}_radius_mm'].dropna()

            summary_data[f'{set_name}_min_px']  = [curv_px_col.min()  if len(curv_px_col)  > 0 else np.nan]
            summary_data[f'{set_name}_max_px']  = [curv_px_col.max()  if len(curv_px_col)  > 0 else np.nan]
            summary_data[f'{set_name}_mean_px'] = [curv_px_col.mean() if len(curv_px_col) > 0 else np.nan]
            summary_data[f'{set_name}_std_px']  = [curv_px_col.std()  if len(curv_px_col)  > 0 else np.nan]

            summary_data[f'{set_name}_min_mm']  = [curv_mm_col.min()  if len(curv_mm_col)  > 0 else np.nan]
            summary_data[f'{set_name}_max_mm']  = [curv_mm_col.max()  if len(curv_mm_col)  > 0 else np.nan]
            summary_data[f'{set_name}_mean_mm'] = [curv_mm_col.mean() if len(curv_mm_col) > 0 else np.nan]
            summary_data[f'{set_name}_std_mm']  = [curv_mm_col.std()  if len(curv_mm_col)  > 0 else np.nan]

            summary_data[f'{set_name}_radius_min_px']  = [radius_px_col.min() if len(radius_px_col) > 0 else np.nan]
            summary_data[f'{set_name}_radius_max_px']  = [radius_px_col.max() if len(radius_px_col) > 0 else np.nan]
            summary_data[f'{set_name}_radius_mean_px'] = [radius_px_col.mean() if len(radius_px_col) > 0 else np.nan]

            summary_data[f'{set_name}_radius_min_mm']  = [radius_mm_col.min() if len(radius_mm_col) > 0 else np.nan]
            summary_data[f'{set_name}_radius_max_mm']  = [radius_mm_col.max() if len(radius_mm_col) > 0 else np.nan]
            summary_data[f'{set_name}_radius_mean_mm'] = [radius_mm_col.mean() if len(radius_mm_col) > 0 else np.nan]

            valid_class_data = valid_data[valid_data[f'{set_name}_class'] != 'Invalid']
            u_count = valid_class_data[f'{set_name}_u_shape'].sum() if f'{set_name}_u_shape' in valid_class_data.columns else 0
            n_count = valid_class_data[f'{set_name}_n_shape'].sum() if f'{set_name}_n_shape' in valid_class_data.columns else 0
            summary_data[f'{set_name}_u_shape_count'] = [u_count]
            summary_data[f'{set_name}_n_shape_count'] = [n_count]
            summary_data[f'{set_name}_u_shape_percentage'] = [u_count / len(valid_class_data) * 100 if len(valid_class_data) > 0 else 0]
            summary_data[f'{set_name}_n_shape_percentage'] = [n_count / len(valid_class_data) * 100 if len(valid_class_data) > 0 else 0]

            if f'{set_name}_class' in valid_data.columns:
                class_counts = valid_data[f'{set_name}_class'].value_counts()
                summary_data[f'{set_name}_extreme_count'] = [class_counts.get('Extreme', 0)]
                summary_data[f'{set_name}_normal_count']  = [class_counts.get('Normal', 0)]
                summary_data[f'{set_name}_minimal_count'] = [class_counts.get('Minimal', 0)]
                summary_data[f'{set_name}_invalid_count'] = [class_counts.get('Invalid', 0)]

                total_valid_classified = len(valid_data[(valid_data[f'{set_name}_class'].notna()) & (valid_data[f'{set_name}_class'] != 'Invalid')])
                if total_valid_classified > 0:
                    summary_data[f'{set_name}_extreme_percentage'] = [class_counts.get('Extreme', 0) / total_valid_classified * 100]
                    summary_data[f'{set_name}_normal_percentage']  = [class_counts.get('Normal', 0)  / total_valid_classified * 100]
                    summary_data[f'{set_name}_minimal_percentage'] = [class_counts.get('Minimal', 0) / total_valid_classified * 100]
                else:
                    summary_data[f'{set_name}_extreme_percentage'] = [0]
                    summary_data[f'{set_name}_normal_percentage']  = [0]
                    summary_data[f'{set_name}_minimal_percentage'] = [0]

                total_classified = len(valid_data[f'{set_name}_class'].dropna())
                summary_data[f'{set_name}_invalid_percentage'] = [class_counts.get('Invalid', 0) / total_classified * 100 if total_classified > 0 else 0]
        else:
            for stat in ['min_px','max_px','mean_px','std_px','min_mm','max_mm','mean_mm','std_mm',
                         'radius_min_px','radius_max_px','radius_mean_px','radius_min_mm','radius_max_mm','radius_mean_mm']:
                summary_data[f'{set_name}_{stat}'] = [np.nan]
            for count in ['u_shape_count','n_shape_count','extreme_count','normal_count','minimal_count','invalid_count']:
                summary_data[f'{set_name}_{count}'] = [0]
            for percentage in ['u_shape_percentage','n_shape_percentage','extreme_percentage','normal_percentage','minimal_percentage','invalid_percentage']:
                summary_data[f'{set_name}_{percentage}'] = [0]

    # Swimming counts/distances
    for part in swimming_count_part_names:
        if f'{part}_cumulative_swim_count' in data.columns:
            swim_count_series = data[f'{part}_cumulative_swim_count'].dropna()
            one_sided_series = data[f'{part}_cumulative_one_sided_bending_count'].dropna()
            total_swim_count = swim_count_series.iloc[-1] if len(swim_count_series) > 0 else 0
            total_one_sided = one_sided_series.iloc[-1] if len(one_sided_series) > 0 else 0

            rel_px_series = data[f'{part}_cumulative_relative_travel_distance_px'].dropna()
            rel_mm_series = data[f'{part}_cumulative_relative_travel_distance_mm'].dropna()
            trav_px_series = data[f'{part}_cumulative_travel_distance_px'].dropna()
            trav_mm_series = data[f'{part}_cumulative_travel_distance_mm'].dropna()

            relative_travel_px = rel_px_series.iloc[-1] if len(rel_px_series) > 0 else 0
            relative_travel_mm = rel_mm_series.iloc[-1] if len(rel_mm_series) > 0 else 0
            travel_px = trav_px_series.iloc[-1] if len(trav_px_series) > 0 else 0
            travel_mm = trav_mm_series.iloc[-1] if len(trav_mm_series) > 0 else 0

            summary_data[f'{part}_total_swim_count'] = [total_swim_count]
            summary_data[f'{part}_total_one_sided_bending'] = [total_one_sided]
            summary_data[f'{part}_relative_travel_distance_px'] = [relative_travel_px]
            summary_data[f'{part}_relative_travel_distance_mm'] = [relative_travel_mm]
            summary_data[f'{part}_travel_distance_px'] = [travel_px]
            summary_data[f'{part}_travel_distance_mm'] = [travel_mm]

            if 'MFC5_total_distance_mm' in summary_data and travel_mm > 0:
                head_distance = summary_data['MFC5_total_distance_mm'][0]
                efficiency = (head_distance / travel_mm) * 100 if travel_mm > 0 else 0
                summary_data[f'{part}_swimming_efficiency_%'] = [efficiency]
        else:
            summary_data[f'{part}_total_swim_count'] = [0]
            summary_data[f'{part}_total_one_sided_bending'] = [0]
            summary_data[f'{part}_relative_travel_distance_px'] = [0]
            summary_data[f'{part}_relative_travel_distance_mm'] = [0]
            summary_data[f'{part}_travel_distance_px'] = [0]
            summary_data[f'{part}_travel_distance_mm'] = [0]

    # Body lengths
    if 'mfc_length_mm' in data.columns:
        mfc = data['mfc_length_mm'].dropna()
        if len(mfc) > 0:
            summary_data['mfc_length_mean_mm'] = [mfc.mean()]
            summary_data['mfc_length_std_mm'] = [mfc.std()]
            summary_data['mfc_length_min_mm'] = [mfc.min()]
            summary_data['mfc_length_max_mm'] = [mfc.max()]
        else:
            summary_data['mfc_length_mean_mm'] = [np.nan]
            summary_data['mfc_length_std_mm'] = [np.nan]
            summary_data['mfc_length_min_mm'] = [np.nan]
            summary_data['mfc_length_max_mm'] = [np.nan]

    if 'body_length_mm' in data.columns:
        bl = data['body_length_mm'].dropna()
        bl = bl[(bl >= LOWER_BODY_LENGTH_LIMIT) & (bl <= UPPER_BODY_LENGTH_LIMIT)]
        if len(bl) > 0:
            summary_data['body_length_mean_mm'] = [bl.mean()]
            summary_data['body_length_std_mm'] = [bl.std()]
            summary_data['body_length_min_mm'] = [bl.min()]
            summary_data['body_length_max_mm'] = [bl.max()]
        else:
            summary_data['body_length_mean_mm'] = [np.nan]
            summary_data['body_length_std_mm'] = [np.nan]
            summary_data['body_length_min_mm'] = [np.nan]
            summary_data['body_length_max_mm'] = [np.nan]

    summary_df = pd.DataFrame(summary_data)
    summary_df = apply_dataframe_rounding_and_thresholds(summary_df)
    summary_file = os.path.join(result_csv_path, f"{video_basename}_summary.csv")
    try:
        with open(summary_file, 'w', newline='', encoding='utf-8-sig') as f:
            summary_df.to_csv(f, index=False)
        logger.info(f"Saved summary data for {video_basename} to CSV with proper rounding and thresholds.")
    except Exception as e:
        logger.error(f"Error saving summary for {video_basename}: {e}")

# ----------------------------------------------------------------------------
# Per-file processing (worker target)
# ----------------------------------------------------------------------------
def process_csv_file(args: Dict[str, Any]) -> Tuple[str, bool, str]:
    """
    Processes a single CSV file with enhanced swimming count logic and curvature classifications.
    Returns: (csv_file, success_bool, error_message_if_any)
    """
    csv_file = args['csv_file']
    source_path = args['source_path']
    result_csv_path = args['result_csv_path']
    result_log_path = args['result_log_path']
    body_part_names = args['body_part_names']
    conversion_rate = args['conversion_rate']
    progress_queue = args['progress_queue']
    tracked_body_parts = args['tracked_body_parts']
    wrong_detect_threshold_px = args['wrong_detect_threshold_px']
    show_frame_progress = args.get('show_frame_progress', False)

    curvature_sets_local = []
    if 'curvature_part_names_1' in args:
        curvature_sets_local.append({'name': 'curvature_1', 'parts': args['curvature_part_names_1']})
    if 'curvature_part_names_2' in args:
        curvature_sets_local.append({'name': 'curvature_2', 'parts': args['curvature_part_names_2']})
    if 'curvature_part_names_3' in args:
        curvature_sets_local.append({'name': 'curvature_3', 'parts': args['curvature_part_names_3']})

    curvature_min_points_dict = args.get('curvature_min_points_dict', CURVATURE_MIN_POINTS_DICT)
    curvature_length_bounds_local = args.get('curvature_length_bounds', [(None, None)] * len(curvature_sets_local))
    swimming_count_parts = args.get('swimming_count_part_names', [])
    circular_fit_method = args.get('circular_fit_method', DEFAULT_CIRCULAR_FIT_METHOD)

    # Logger per-file
    log_prefix = os.path.splitext(csv_file)[0]
    log_filename = os.path.join(result_log_path, f'processing_{log_prefix}.log')
    logger = setup_logger(log_filename)
    logger.info(f"Started processing file: {csv_file}")
    logger.info(f"Using circular fitting method: {circular_fit_method}")
    logger.info(f"Invalid curvature threshold: {INVALID_CURVATURE_THRESHOLD} mm^-1")
    logger.info(f"Body length limits: {LOWER_BODY_LENGTH_LIMIT}-{UPPER_BODY_LENGTH_LIMIT} mm")

    csv_path = os.path.join(source_path, csv_file)
    try:
        test_df = pd.read_csv(csv_path, nrows=5)
        logger.info(f"CSV validation - Shape: {test_df.shape}, Columns: {len(test_df.columns)}")
        data = pd.read_csv(csv_path, skiprows=2)
        logger.info(f"Successfully read CSV: {csv_file}, Shape after skiprows: {data.shape}")
        if len(data) == 0:
            msg = f"Empty CSV file: {csv_file}"
            logger.error(msg)
            if progress_queue:
                progress_queue.put({'csv_file': csv_file, 'progress_percent': 100.0}, block=False)
            return (csv_file, False, msg)
        expected_cols = len(body_part_names) * 3 + 1
        if len(data.columns) < expected_cols:
            msg = f"Insufficient columns in {csv_file}: found {len(data.columns)}, expected at least {expected_cols}"
            logger.error(msg)
            if progress_queue:
                progress_queue.put({'csv_file': csv_file, 'progress_percent': 100.0}, block=False)
            return (csv_file, False, msg)
    except Exception as e:
        msg = f"Error reading {csv_file}: {e}"
        logger.error(msg)
        if progress_queue:
            progress_queue.put({'csv_file': csv_file, 'progress_percent': 100.0}, block=False)
        return (csv_file, False, msg)

    # Mapping
    mapping = {}
    try:
        for i, name in enumerate(body_part_names):
            if (1 + i*3 + 2) < len(data.columns):
                mapping[name] = data.columns[1 + i*3 : 4 + i*3]
            else:
                logger.warning(f"Columns for body part '{name}' missing in {csv_file}.")
    except IndexError as e:
        msg = f"Error mapping columns for {csv_file}: {e}"
        logger.error(msg)
        if progress_queue:
            progress_queue.put({'csv_file': csv_file, 'progress_percent': 100.0}, block=False)
        return (csv_file, False, msg)

    video_basename = os.path.splitext(csv_file)[0]
    video_basename = normalize_video_basename(video_basename)
    _ = truncate_filename(video_basename)  # retained for parity; not otherwise used

    # Pre-allocate new columns
    new_columns = {}
    for part in tracked_body_parts:
        if part in mapping:
            new_columns[f'{part}_frame_dist_px'] = np.nan
            new_columns[f'{part}_frame_dist_mm'] = np.nan
            new_columns[f'{part}_cum_dist_px'] = 0.0
            new_columns[f'{part}_cum_dist_mm'] = 0.0

    valid_curvature_sets = []
    for idx, curv_conf in enumerate(curvature_sets_local):
        parts_list = curv_conf['parts']
        set_name = curv_conf['name']
        if all(part in mapping for part in parts_list):
            new_columns[f'{set_name}_px'] = np.nan
            new_columns[f'{set_name}_mm'] = np.nan
            new_columns[f'{set_name}_radius_px'] = np.nan
            new_columns[f'{set_name}_radius_mm'] = np.nan
            new_columns[f'{set_name}_center_x'] = np.nan
            new_columns[f'{set_name}_center_y'] = np.nan
            new_columns[f'{set_name}_u_shape'] = 0
            new_columns[f'{set_name}_n_shape'] = 0
            new_columns[f'{set_name}_class'] = None
            valid_curvature_sets.append({'name': set_name, 'parts': parts_list})
        else:
            logger.warning(f"Curvature set {set_name} has missing body part columns.")

    # Require Head1 and (Head2 or MFC1) to compute swimming metrics
    if not ( (REFERENCE_START in mapping) and ((REFERENCE_END_PRIMARY in mapping) or (REFERENCE_END_FALLBACK in mapping)) ):
        if swimming_count_parts:
            logger.warning(f"Reference line parts missing for swimming metrics. "
                        f"Need {REFERENCE_START} and ({REFERENCE_END_PRIMARY} or {REFERENCE_END_FALLBACK}). "
                        f"Swimming count calculation will be skipped.")
        swimming_count_parts = []

    valid_swimming_parts = []
    for part in list(swimming_count_parts):
        if part in mapping:
            valid_swimming_parts.append(part)
        else:
            logger.warning(f"Tail part '{part}' not found in data.")
    swimming_count_parts = valid_swimming_parts

    for part in swimming_count_parts:
        new_columns[f'{part}_sign'] = np.nan
        new_columns[f'{part}_swim_count'] = 0
        new_columns[f'{part}_cumulative_swim_count'] = 0
        new_columns[f'{part}_one_sided_bending_count'] = 0
        new_columns[f'{part}_cumulative_one_sided_bending_count'] = 0
        new_columns[f'{part}_relative_travel_distance_px'] = 0.0
        new_columns[f'{part}_relative_travel_distance_mm'] = 0.0
        new_columns[f'{part}_cumulative_relative_travel_distance_px'] = 0.0
        new_columns[f'{part}_cumulative_relative_travel_distance_mm'] = 0.0
        new_columns[f'{part}_travel_distance_px'] = 0.0
        new_columns[f'{part}_travel_distance_mm'] = 0.0
        new_columns[f'{part}_cumulative_travel_distance_px'] = 0.0
        new_columns[f'{part}_cumulative_travel_distance_mm'] = 0.0

    new_columns['mfc_length_px'] = np.nan
    new_columns['mfc_length_mm'] = np.nan
    new_columns['body_length_px'] = np.nan
    new_columns['body_length_mm'] = np.nan

    data = pd.concat([data, pd.DataFrame(new_columns, index=data.index)], axis=1)

    class SwimmingTracker:
        def __init__(self, part_name):
            self.part_name = part_name
            self.threshold = SWIMMING_JITTER_THRESHOLDS.get(part_name, 2)
            self.prev_sign_numeric = None
            self.cumulative_swim_count = 0
            self.cumulative_one_sided_count = 0

        def update(self, current_sign_numeric):
            per_frame_swim = 0
            per_frame_one_sided = 0
            if self.prev_sign_numeric is not None:
                if (self.prev_sign_numeric < 0 and current_sign_numeric > 0) or (self.prev_sign_numeric > 0 and current_sign_numeric < 0):
                    per_frame_swim = 1
                    self.cumulative_swim_count += 1
                elif (abs(current_sign_numeric - self.prev_sign_numeric) > self.threshold and self.prev_sign_numeric * current_sign_numeric > 0):
                    per_frame_one_sided = 1
                    self.cumulative_one_sided_count += 1
            self.prev_sign_numeric = current_sign_numeric
            return per_frame_swim, self.cumulative_swim_count, per_frame_one_sided, self.cumulative_one_sided_count

    swimming_trackers = {part: SwimmingTracker(part) for part in swimming_count_parts}

    prev_coords = {}
    total_frames = len(data)
    total_skipped_frames_missing_invalid = 0
    total_skipped_frames_wrong_detection = 0

    last_valid_cumulative = {}
    for part in tracked_body_parts:
        if part in mapping:
            last_valid_cumulative[f'{part}_cum_dist_px'] = 0.0
            last_valid_cumulative[f'{part}_cum_dist_mm'] = 0.0
    for part in swimming_count_parts:
        last_valid_cumulative[f'{part}_cumulative_swim_count'] = 0
        last_valid_cumulative[f'{part}_cumulative_one_sided_bending_count'] = 0
        last_valid_cumulative[f'{part}_cumulative_relative_travel_distance_px'] = 0.0
        last_valid_cumulative[f'{part}_cumulative_relative_travel_distance_mm'] = 0.0
        last_valid_cumulative[f'{part}_cumulative_travel_distance_px'] = 0.0
        last_valid_cumulative[f'{part}_cumulative_travel_distance_mm'] = 0.0

    prev_valid_coords = {}        # last ACCEPTED coords per part
    GRACE_FRAMES_AFTER_SKIP = 1   # forgive the next frame after a skip
    grace_remaining = 0
    K_BAD_PARTS = 2               # require >= K parts to exceed threshold to skip frame

    report_interval = max(1, total_frames // 100)  # ~1%

    iterator = tqdm(data.iterrows(), total=total_frames, desc=f"Processing {csv_file}", leave=False) if show_frame_progress else data.iterrows()

    frames_processed = 0
    for idx, row in iterator:
        frames_processed += 1
        if progress_queue and frames_processed % report_interval == 0:
            try:
                progress_queue.put({
                    'csv_file': csv_file,
                    'frames_processed': frames_processed,
                    'total_frames': total_frames,
                    'progress_percent': (frames_processed / total_frames) * 100.0
                }, block=False)
            except Exception:
                pass

        current_coords_tracked_parts = extract_coordinates(row, mapping, tracked_body_parts)
        valid_frame = True
        for part in tracked_body_parts:
            if part not in current_coords_tracked_parts or current_coords_tracked_parts[part] is None:
                valid_frame = False
                break

        if not valid_frame:
            total_skipped_frames_missing_invalid += 1
            for part in tracked_body_parts:
                if part in mapping:
                    data.at[idx, f'{part}_cum_dist_px'] = last_valid_cumulative[f'{part}_cum_dist_px']
                    data.at[idx, f'{part}_cum_dist_mm'] = last_valid_cumulative[f'{part}_cum_dist_mm']
            for part in swimming_count_parts:
                data.at[idx, f'{part}_cumulative_swim_count'] = last_valid_cumulative[f'{part}_cumulative_swim_count']
                data.at[idx, f'{part}_cumulative_one_sided_bending_count'] = last_valid_cumulative[f'{part}_cumulative_one_sided_bending_count']
                data.at[idx, f'{part}_cumulative_relative_travel_distance_px'] = last_valid_cumulative[f'{part}_cumulative_relative_travel_distance_px']
                data.at[idx, f'{part}_cumulative_relative_travel_distance_mm'] = last_valid_cumulative[f'{part}_cumulative_relative_travel_distance_mm']
                data.at[idx, f'{part}_cumulative_travel_distance_px'] = last_valid_cumulative[f'{part}_cumulative_travel_distance_px']
                data.at[idx, f'{part}_cumulative_travel_distance_mm'] = last_valid_cumulative[f'{part}_cumulative_travel_distance_mm']
            continue

        all_coords = extract_coordinates(row, mapping, body_part_names)

        mfc_length_px = calculate_sequential_length(all_coords, MFC_LENGTH_SEQUENCE)
        if mfc_length_px is not None:
            data.at[idx, 'mfc_length_px'] = mfc_length_px
            data.at[idx, 'mfc_length_mm'] = mfc_length_px * conversion_rate

        body_length_px = calculate_sequential_length(all_coords, BODY_LENGTH_SEQUENCE)
        if body_length_px is not None:
            body_length_mm = body_length_px * conversion_rate
            if LOWER_BODY_LENGTH_LIMIT <= body_length_mm <= UPPER_BODY_LENGTH_LIMIT:
                data.at[idx, 'body_length_px'] = body_length_px
                data.at[idx, 'body_length_mm'] = body_length_mm
            else:
                data.at[idx, 'body_length_px'] = np.nan
                data.at[idx, 'body_length_mm'] = np.nan

        # WRONG-DETECTION with re-ID recovery
        if not prev_valid_coords:
            prev_valid_coords = {p: current_coords_tracked_parts.get(p, None) for p in tracked_body_parts if p in mapping}
        else:
            body_len_px = data.at[idx, 'body_length_px'] if 'body_length_px' in data.columns else np.nan
            per_part_thresh = max(wrong_detect_threshold_px, 0.25 * body_len_px) if pd.notna(body_len_px) else wrong_detect_threshold_px

            bad_parts = []
            for part, coord in current_coords_tracked_parts.items():
                prev = prev_valid_coords.get(part, None)
                if prev is not None and coord is not None:
                    dist = distance_2d(coord, prev)
                    if dist > per_part_thresh:
                        bad_parts.append((part, dist))

            if grace_remaining > 0:
                grace_remaining -= 1
                bad_parts = []
                prev_valid_coords = {p: current_coords_tracked_parts.get(p, None) for p in tracked_body_parts if p in mapping}

            if len(bad_parts) >= K_BAD_PARTS:
                total_skipped_frames_wrong_detection += 1
                prev_valid_coords = {p: current_coords_tracked_parts.get(p, None) for p in tracked_body_parts if p in mapping}
                grace_remaining = GRACE_FRAMES_AFTER_SKIP
                for part in tracked_body_parts:
                    if part in mapping:
                        data.at[idx, f'{part}_cum_dist_px'] = last_valid_cumulative[f'{part}_cum_dist_px']
                        data.at[idx, f'{part}_cum_dist_mm'] = last_valid_cumulative[f'{part}_cum_dist_mm']
                for part in swimming_count_parts:
                    data.at[idx, f'{part}_cumulative_swim_count'] = last_valid_cumulative[f'{part}_cumulative_swim_count']
                    data.at[idx, f'{part}_cumulative_one_sided_bending_count'] = last_valid_cumulative[f'{part}_cumulative_one_sided_bending_count']
                    data.at[idx, f'{part}_cumulative_relative_travel_distance_px'] = last_valid_cumulative[f'{part}_cumulative_relative_travel_distance_px']
                    data.at[idx, f'{part}_cumulative_relative_travel_distance_mm'] = last_valid_cumulative[f'{part}_cumulative_relative_travel_distance_mm']
                    data.at[idx, f'{part}_cumulative_travel_distance_px'] = last_valid_cumulative[f'{part}_cumulative_travel_distance_px']
                    data.at[idx, f'{part}_cumulative_travel_distance_mm'] = last_valid_cumulative[f'{part}_cumulative_travel_distance_mm']
                continue
            else:
                for part, _dist in bad_parts:
                    data.at[idx, f'{part}_frame_dist_px'] = 0
                    data.at[idx, f'{part}_frame_dist_mm'] = 0
                for part in tracked_body_parts:
                    if current_coords_tracked_parts.get(part) is not None:
                        prev_valid_coords[part] = current_coords_tracked_parts[part]

        # distances
        for part in tracked_body_parts:
            if part in current_coords_tracked_parts:
                if part in prev_coords and prev_coords[part] is not None:
                    dist = distance_2d(current_coords_tracked_parts[part], prev_coords[part])
                    if dist < JITTER_THRESHOLD:
                        dist = 0
                    data.at[idx, f'{part}_frame_dist_px'] = dist
                    data.at[idx, f'{part}_frame_dist_mm'] = dist * conversion_rate
                    data.at[idx, f'{part}_cum_dist_px'] = last_valid_cumulative[f'{part}_cum_dist_px'] + dist
                    data.at[idx, f'{part}_cum_dist_mm'] = last_valid_cumulative[f'{part}_cum_dist_mm'] + (dist * conversion_rate)
                else:
                    data.at[idx, f'{part}_cum_dist_px'] = last_valid_cumulative[f'{part}_cum_dist_px']
                    data.at[idx, f'{part}_cum_dist_mm'] = last_valid_cumulative[f'{part}_cum_dist_mm']
                last_valid_cumulative[f'{part}_cum_dist_px'] = data.at[idx, f'{part}_cum_dist_px']
                last_valid_cumulative[f'{part}_cum_dist_mm'] = data.at[idx, f'{part}_cum_dist_mm']

        # curvature
        for curv_idx, curv_conf in enumerate(valid_curvature_sets):
            parts_list = curv_conf['parts']
            set_name = curv_conf['name']
            min_points_required = curvature_min_points_dict.get(set_name, 4)
            curv_coords = []
            for part in parts_list:
                if part in all_coords and all_coords[part] is not None:
                    curv_coords.append(all_coords[part])

            min_length, max_length = curvature_length_bounds_local[curv_idx]
            if len(curv_coords) >= min_points_required:
                seq_length_px = 0
                valid_length = True
                for i in range(len(curv_coords) - 1):
                    d = distance_2d(curv_coords[i], curv_coords[i+1])
                    if np.isnan(d):
                        valid_length = False
                        break
                    seq_length_px += d
                seq_length_mm = seq_length_px * conversion_rate
                if valid_length:
                    if min_length is not None and seq_length_mm < min_length:
                        valid_length = False
                    if max_length is not None and seq_length_mm > max_length:
                        valid_length = False

                if valid_length:
                    curvature_px, radius_px, center_x, center_y = fit_circle_and_curvature(curv_coords, method=circular_fit_method)
                    if not np.isnan(curvature_px):
                        curvature_mm = curvature_px / conversion_rate
                        if abs(curvature_mm) <= INVALID_CURVATURE_THRESHOLD:
                            data.at[idx, f'{set_name}_px'] = curvature_px
                            data.at[idx, f'{set_name}_mm'] = curvature_mm
                            data.at[idx, f'{set_name}_radius_px'] = int(round(radius_px))
                            data.at[idx, f'{set_name}_radius_mm'] = int(round(radius_px * conversion_rate))
                            data.at[idx, f'{set_name}_center_x'] = center_x
                            data.at[idx, f'{set_name}_center_y'] = center_y

                            classification = classify_curvature(curvature_mm)
                            apply_body_length_filter = (
                                UPPER_BODY_LENGTH_LIMIT is not None and UPPER_BODY_LENGTH_LIMIT > 0 and
                                LOWER_BODY_LENGTH_LIMIT is not None and LOWER_BODY_LENGTH_LIMIT > 0
                            )
                            if apply_body_length_filter:
                                current_body_length_mm = data.at[idx, 'body_length_mm'] if 'body_length_mm' in data.columns else np.nan
                                if pd.isna(current_body_length_mm):
                                    classification = 'Invalid'
                            data.at[idx, f'{set_name}_class'] = classification

                            if classification != 'Invalid' and 'Head1' in all_coords and parts_list[-1] in all_coords:
                                shape_type = determine_shape_type(center_x, center_y, all_coords['Head1'], all_coords[parts_list[-1]])
                                if shape_type == 'U':
                                    data.at[idx, f'{set_name}_u_shape'] = 1
                                elif shape_type == 'N':
                                    data.at[idx, f'{set_name}_n_shape'] = 1
                        else:
                            data.at[idx, f'{set_name}_px'] = np.nan
                            data.at[idx, f'{set_name}_mm'] = np.nan
                            data.at[idx, f'{set_name}_radius_px'] = np.nan
                            data.at[idx, f'{set_name}_radius_mm'] = np.nan
                            data.at[idx, f'{set_name}_center_x'] = np.nan
                            data.at[idx, f'{set_name}_center_y'] = np.nan
                            data.at[idx, f'{set_name}_class'] = 'Invalid'

        # swimming metrics
        if swimming_count_parts and all_coords:
            start_pt = all_coords.get(REFERENCE_START, None)
            end_pt = None
            if start_pt is not None:
                end_pt = all_coords.get(REFERENCE_END_PRIMARY, None)
                if end_pt is None:
                    end_pt = all_coords.get(REFERENCE_END_FALLBACK, None)

                if (start_pt is not None) and (end_pt is not None):
                    ref_vector = (end_pt[0] - start_pt[0], end_pt[1] - start_pt[1])
                    ref_length = math.sqrt(ref_vector[0]**2 + ref_vector[1]**2)
                    if ref_length > 0:
                        ref_unit = (ref_vector[0] / ref_length, ref_vector[1] / ref_length)
                        for part in swimming_count_parts:
                            if part in all_coords and all_coords[part] is not None:
                                tail_point = all_coords[part]
                                vec_to_tail = (tail_point[0] - start_pt[0], tail_point[1] - start_pt[1])

                                # signed lateral distance relative to Head1→(Head2 or MFC1)
                                signed_distance = ref_unit[0] * vec_to_tail[1] - ref_unit[1] * vec_to_tail[0]
                                data.at[idx, f'{part}_sign'] = signed_distance

                                tracker = swimming_trackers[part]
                                per_frame_swim, cumulative_swim, per_frame_one_sided, cumulative_one_sided = tracker.update(signed_distance)

                                data.at[idx, f'{part}_swim_count'] = per_frame_swim
                                data.at[idx, f'{part}_cumulative_swim_count'] = cumulative_swim
                                data.at[idx, f'{part}_one_sided_bending_count'] = per_frame_one_sided
                                data.at[idx, f'{part}_cumulative_one_sided_bending_count'] = cumulative_one_sided

                                last_valid_cumulative[f'{part}_cumulative_swim_count'] = cumulative_swim
                                last_valid_cumulative[f'{part}_cumulative_one_sided_bending_count'] = cumulative_one_sided

                                if idx > 0 and not pd.isna(data.at[idx-1, f'{part}_sign']):
                                    prev_sign_value = data.at[idx-1, f'{part}_sign']
                                    curr_sign_value = signed_distance
                                    relative_travel = abs(curr_sign_value - prev_sign_value)
                                    data.at[idx, f'{part}_relative_travel_distance_px'] = relative_travel
                                    data.at[idx, f'{part}_relative_travel_distance_mm'] = relative_travel * conversion_rate
                                    data.at[idx, f'{part}_cumulative_relative_travel_distance_px'] = (
                                        last_valid_cumulative[f'{part}_cumulative_relative_travel_distance_px'] + relative_travel
                                    )
                                    data.at[idx, f'{part}_cumulative_relative_travel_distance_mm'] = (
                                        last_valid_cumulative[f'{part}_cumulative_relative_travel_distance_mm'] + relative_travel * conversion_rate
                                    )

                                    if part in prev_coords and prev_coords[part] is not None:
                                        travel_dist = distance_2d(tail_point, prev_coords[part])
                                        if travel_dist >= JITTER_THRESHOLD:
                                            data.at[idx, f'{part}_travel_distance_px'] = travel_dist
                                            data.at[idx, f'{part}_travel_distance_mm'] = travel_dist * conversion_rate
                                            data.at[idx, f'{part}_cumulative_travel_distance_px'] = (
                                                last_valid_cumulative[f'{part}_cumulative_travel_distance_px'] + travel_dist
                                            )
                                            data.at[idx, f'{part}_cumulative_travel_distance_mm'] = (
                                                last_valid_cumulative[f'{part}_cumulative_travel_distance_mm'] + travel_dist * conversion_rate
                                            )
                                        else:
                                            data.at[idx, f'{part}_travel_distance_px'] = 0
                                            data.at[idx, f'{part}_travel_distance_mm'] = 0
                                            data.at[idx, f'{part}_cumulative_travel_distance_px'] = last_valid_cumulative[f'{part}_cumulative_travel_distance_px']
                                            data.at[idx, f'{part}_cumulative_travel_distance_mm'] = last_valid_cumulative[f'{part}_cumulative_travel_distance_mm']
                                    else:
                                        data.at[idx, f'{part}_cumulative_travel_distance_px'] = last_valid_cumulative[f'{part}_cumulative_travel_distance_px']
                                        data.at[idx, f'{part}_cumulative_travel_distance_mm'] = last_valid_cumulative[f'{part}_cumulative_travel_distance_mm']

                                    last_valid_cumulative[f'{part}_cumulative_relative_travel_distance_px'] = data.at[idx, f'{part}_cumulative_relative_travel_distance_px']
                                    last_valid_cumulative[f'{part}_cumulative_relative_travel_distance_mm'] = data.at[idx, f'{part}_cumulative_relative_travel_distance_mm']
                                    last_valid_cumulative[f'{part}_cumulative_travel_distance_px'] = data.at[idx, f'{part}_cumulative_travel_distance_px']
                                    last_valid_cumulative[f'{part}_cumulative_travel_distance_mm'] = data.at[idx, f'{part}_cumulative_travel_distance_mm']
                    else:
                        logger.warning(f"Frame {idx}: Zero-length reference line, skipping swimming count")

        prev_coords = {part: current_coords_tracked_parts.get(part, (None, None)) for part in tracked_body_parts if part in mapping}

    # Save
    save_results(data, video_basename, result_csv_path, logger)
    prepare_summary_data(data, video_basename, total_frames, conversion_rate,
                         [p for p in tracked_body_parts if p in mapping],
                         total_skipped_frames_missing_invalid,
                         total_skipped_frames_wrong_detection,
                         valid_curvature_sets,
                         result_csv_path, logger, mapping)

    logger.info(f"Finished processing file: {csv_file}")

    # Ensure monitor sees completion for this specific file
    if progress_queue:
        try:
            progress_queue.put({'csv_file': csv_file, 'progress_percent': 100.0}, block=False)
        except Exception:
            pass

    return (csv_file, True, "")

# ----------------------------------------------------------------------------
# Tail Beat Frequency Calculation for Total Summary
# ----------------------------------------------------------------------------
def calculate_tail_beat_frequencies(combined_df, result_csv_path, logger):
    logger.info("Calculating tail beat frequencies for Total_summary.csv")
    for part in swimming_count_part_names:
        combined_df[f'{part}_swim_frequency_hz'] = np.nan
        combined_df[f'{part}_one_sided_frequency_hz'] = np.nan
        combined_df[f'{part}_combined_frequency_hz'] = np.nan

    for idx, row in combined_df.iterrows():
        video_name = row['video_name']
        tracking_file = None
        for file in os.listdir(result_csv_path):
            if file.endswith('_tracking_analysis.csv') and video_name in file:
                tracking_file = file
                break
        if not tracking_file:
            logger.warning(f"No tracking analysis file found for {video_name}")
            continue
        try:
            tracking_path = os.path.join(result_csv_path, tracking_file)
            tracking_data = pd.read_csv(tracking_path, encoding='utf-8-sig')
            logger.info(f"Processing tail beat frequencies for {video_name}")
            for part in swimming_count_part_names:
                sign_col = f'{part}_sign'
                swim_count_col = f'{part}_total_swim_count'
                one_sided_col = f'{part}_total_one_sided_bending'
                if swim_count_col not in row.index or one_sided_col not in row.index:
                    logger.warning(f"Missing count columns for {part} in summary data")
                    continue
                total_swim_count = row[swim_count_col] if pd.notna(row[swim_count_col]) else 0
                total_one_sided_count = row[one_sided_col] if pd.notna(row[one_sided_col]) else 0
                if sign_col in tracking_data.columns:
                    effective_frames = tracking_data[sign_col].notna().sum()
                    if effective_frames > 0:
                        duration_seconds = effective_frames / DEFAULT_FPS
                        swim_frequency = (total_swim_count / 2) / duration_seconds if duration_seconds > 0 else 0
                        one_sided_frequency = (total_one_sided_count / 2) / duration_seconds if duration_seconds > 0 else 0
                        combined_frequency = ((total_swim_count + total_one_sided_count) / 2) / duration_seconds if duration_seconds > 0 else 0
                        combined_df.at[idx, f'{part}_swim_frequency_hz'] = swim_frequency
                        combined_df.at[idx, f'{part}_one_sided_frequency_hz'] = one_sided_frequency
                        combined_df.at[idx, f'{part}_combined_frequency_hz'] = combined_frequency
                    else:
                        logger.warning(f"No effective frames found for {part} in {video_name}")
                else:
                    logger.warning(f"Sign column {sign_col} not found in tracking data for {video_name}")
        except Exception as e:
            logger.error(f"Error calculating frequencies for {video_name}: {e}")
            continue

    logger.info("Tail beat frequency calculation completed")
    return combined_df

# ----------------------------------------------------------------------------
# MULTI-BASE CHANGE: factor the original "main body" into a function that
#                    processes a single base path (creates its own outputs).
# ----------------------------------------------------------------------------
def run_for_base_path(base_path: str, ctx, main_logger: logging.Logger):
    # Derived Paths (per base_path)
    source_path = base_path
    result_base_path = base_path + f'_step1_result({timestamp})'
    result_csv_path = os.path.join(result_base_path, '1_analysis_result_csv')
    result_log_path = os.path.join(result_base_path, '2_processing_logs')

    # Create necessary directories
    os.makedirs(result_csv_path, exist_ok=True)
    os.makedirs(result_log_path, exist_ok=True)

    main_logger.info("Starting batch processing of fish movement data (analysis only).")
    main_logger.info(f"[BASE] {base_path}")
    main_logger.info(f"Configuration: Conversion Rate={conversion_rate_mm_per_px} mm/px, "
                     f"Tracked Parts={TRACKED_BODY_PARTS}, "
                     f"Curvature Sets={[c['name'] for c in curvature_sets] if curvature_sets else None}, "
                     f"Swimming Count Parts={swimming_count_part_names}, "
                     f"Swimming Jitter Thresholds={SWIMMING_JITTER_THRESHOLDS}, Max Processes={MAX_PROCESSES}, "
                     f"Wrong Detect Threshold={WRONG_DETECT_THRESHOLD_PX}px, "
                     f"Body Length Limits: {LOWER_BODY_LENGTH_LIMIT}-{UPPER_BODY_LENGTH_LIMIT}mm, "
                     f"MFC Length Sequence={MFC_LENGTH_SEQUENCE}, Body Length Sequence={BODY_LENGTH_SEQUENCE}, "
                     f"Curvature Thresholds: Extreme={EXTREME_BENDING_THRESHOLD}, Normal={NORMAL_BENDING_THRESHOLD}, "
                     f"Invalid={INVALID_CURVATURE_THRESHOLD}, "
                     f"Circular Fit Method={DEFAULT_CIRCULAR_FIT_METHOD}, "
                     f"Min Points Dict={CURVATURE_MIN_POINTS_DICT}")

    if not os.path.isdir(source_path):
        main_logger.error(f"Source path does not exist: {source_path}")
        print(f"[SKIP] Source path does not exist: {source_path}")
        return

    # Find CSV files
    csv_files = [f for f in os.listdir(source_path) if f.endswith('.csv')]
    if not csv_files:
        main_logger.warning(f"No CSV files found in directory: {source_path}")
        print(f"[INFO] No CSV files found to process in: {source_path}")
        return
    else:
        main_logger.info(f"Found {len(csv_files)} CSV files to process.")

    tasks = []
    for csv_file in csv_files:
        tasks.append({
            'csv_file': csv_file,
            'source_path': source_path,
            'result_csv_path': result_csv_path,
            'result_log_path': result_log_path,
            'body_part_names': body_part_names,
            'conversion_rate': conversion_rate_mm_per_px,
            'tracked_body_parts': TRACKED_BODY_PARTS,
            'wrong_detect_threshold_px': WRONG_DETECT_THRESHOLD_PX,
            'curvature_part_names_1': curvature_part_names_1,
            'curvature_part_names_2': curvature_part_names_2,
            'curvature_part_names_3': curvature_part_names_3,
            'curvature_min_points_dict': CURVATURE_MIN_POINTS_DICT,
            'curvature_length_bounds': curvature_length_bounds,
            'swimming_count_part_names': swimming_count_part_names,
            'circular_fit_method': DEFAULT_CIRCULAR_FIT_METHOD,
            'show_frame_progress': False,  # off in workers for clean output
        })

    print(f"\n[BASE] {base_path}")
    print(f"Processing {len(tasks)} CSV files...")
    print(f"Using {MAX_PROCESSES} parallel processes.")
    print(f"Circular fitting method: {DEFAULT_CIRCULAR_FIT_METHOD}")
    print(f"Minimum points for curvature sets: {CURVATURE_MIN_POINTS_DICT}")
    print(f"Invalid curvature threshold: {INVALID_CURVATURE_THRESHOLD} mm^-1")
    print(f"Body length limits: {LOWER_BODY_LENGTH_LIMIT}-{UPPER_BODY_LENGTH_LIMIT} mm")

    def monitor_progress(progress_queue, total_files, stop_event: threading.Event, timeout=1):
        file_progress = {}
        completed_files = set()
        last_print_len = 0
        while not stop_event.is_set():
            try:
                progress_data = progress_queue.get(timeout=timeout)
                csv_file = progress_data.get('csv_file')
                progress_percent = float(progress_data.get('progress_percent', 0.0))
                file_progress[csv_file] = min(100.0, max(0.0, progress_percent))
                if progress_percent >= 100.0:
                    completed_files.add(csv_file)
                overall_progress = sum(file_progress.values()) / max(1, len(file_progress))
                line = f"\rOverall Progress: {overall_progress:5.1f}% | Completed: {len(completed_files)}/{total_files} files"
                print(line + " " * max(0, last_print_len - len(line)), end="", flush=True)
                last_print_len = len(line)
                if len(completed_files) >= total_files:
                    break
            except queue.Empty:
                continue
            except Exception:
                break
        print()  # newline at end

    successful_files: List[str] = []
    failed_files: List[str] = []

    try:
        with ctx.Manager() as manager:
            progress_queue = manager.Queue(maxsize=1000)
            stop_event = threading.Event()
            progress_thread = threading.Thread(target=monitor_progress, args=(progress_queue, len(tasks), stop_event), daemon=True)
            progress_thread.start()

            # Fill tasks with the shared queue
            for t in tasks:
                t['progress_queue'] = progress_queue

            # Use context Pool with initializer
            with ctx.Pool(processes=MAX_PROCESSES, initializer=init_worker) as pool:
                # launch all and collect results as they complete
                results_async = [pool.apply_async(process_csv_file, (t,)) for t in tasks]

                # Wait for results
                for i, res in enumerate(results_async):
                    try:
                        csv_file, ok, err = res.get()
                        if ok:
                            successful_files.append(csv_file)
                            print(f"✅ Completed: {csv_file}")
                        else:
                            failed_files.append(csv_file)
                            main_logger.error(f"Error processing {csv_file}: {err}")
                            print(f"❌ Error processing {csv_file}: {err}")
                    except KeyboardInterrupt:
                        print("\nInterrupted by user. Terminating workers...")
                        pool.terminate()
                        raise
                    except Exception as e:
                        this_file = tasks[i]['csv_file']
                        failed_files.append(this_file)
                        main_logger.error(f"Unhandled error processing {this_file}: {e}")
                        print(f"❌ Unhandled error processing {this_file}: {e}")

            # Tell monitor to stop when all done
            stop_event.set()
            progress_thread.join(timeout=3)

        # Summary
        print(f"\n{'=' * 80}")
        print(f"[BASE] {base_path} — Processing Summary:")
        print(f"  ✅ Successful: {len(successful_files)}/{len(tasks)} files")
        print(f"  ❌ Failed: {len(failed_files)}/{len(tasks)} files")
        if failed_files:
            print(f"  Failed files: {', '.join(failed_files)}")
        print("All files processed for this base.")

        # Combine summary files (per base)
        main_logger.info("Batch processing completed for this base. Compiling total summary.")
        all_summary_files = [f for f in os.listdir(result_csv_path) if f.endswith('_summary.csv')]

        if all_summary_files:
            combined_df = pd.DataFrame()
            for summary_file in all_summary_files:
                try:
                    df = pd.read_csv(os.path.join(result_csv_path, summary_file), encoding='utf-8-sig')
                    combined_df = pd.concat([combined_df, df], ignore_index=True)
                except Exception as e:
                    main_logger.error(f"Error reading {summary_file} for total summary: {e}")

            if not combined_df.empty:
                combined_df = calculate_tail_beat_frequencies(combined_df, result_csv_path, main_logger)
                combined_df = apply_dataframe_rounding_and_thresholds(combined_df)
                total_summary_path = os.path.join(result_csv_path, 'Total_summary.csv')
                try:
                    with open(total_summary_path, 'w', newline='', encoding='utf-8-sig') as f:
                        combined_df.to_csv(f, index=False)
                    main_logger.info(f"Total_summary.csv created at {total_summary_path} with proper rounding and thresholds.")
                    print(f"\n✅ [BASE] {base_path} complete! Total summary saved at: {total_summary_path}")
                except Exception as e:
                    main_logger.error(f"Error saving Total_summary.csv: {e}")
            else:
                main_logger.warning("No data to combine for Total_summary.csv.")
        else:
            main_logger.warning("No summary files found to combine into Total_summary.csv.")
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
        raise

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # Reduce parent-thread contention too
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    try:
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

    # Safer start method for OpenCV/BLAS heavy workloads
    ctx = get_context("spawn")

    # MULTI-BASE CHANGE: one top-level log file for the whole batch run
    # (Each file still gets its own per-file log under each base's log folder)
    top_level_log_dir = os.path.join(os.getcwd(), f"multi_base_logs_{timestamp}")
    os.makedirs(top_level_log_dir, exist_ok=True)
    main_log_filename = os.path.join(top_level_log_dir, f'main_processing_log_{timestamp}.log')
    main_logger = setup_logger(main_log_filename)

    # Normalize BASE_PATHS into a list
    if isinstance(BASE_PATHS, str):
        paths_to_run = [BASE_PATHS]
    else:
        paths_to_run = list(BASE_PATHS)

    # Run each base path sequentially (each uses its own parallel Pool internally)
    for base in paths_to_run:
        try:
            run_for_base_path(base, ctx, main_logger)
        except Exception as e:
            print(f"\n[BASE ERROR] {base}: {e}")
            main_logger.error(f"Top-level error while processing base '{base}': {e}")

    print("\n🎉 All requested base paths have been processed.")
