#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepLabCut (DLC) Inference Script

Runs pose-estimation inference on a directory of video files using a
pre-trained DeepLabCut model.  Optionally creates labeled (overlaid)
videos for visual verification.

Features
--------
- GPU selection, batch size, and CPU thread counts are controlled via
  global parameters at the top of the file.
- CPU-side decode / preprocessing bottlenecks are mitigated by pinning
  OMP, MKL, OpenBLAS, and TensorFlow thread counts.
- TensorFlow GPU memory growth is enabled to avoid large pre-allocation
  spikes on multi-GPU machines.
- Video discovery is non-recursive (top-level of the given directory only).

Prerequisites
-------------
- DeepLabCut (tested with DLC 2.x / 3.x)
- TensorFlow (GPU build recommended)
- OpenCV (optional, for thread tuning)

Usage
-----
1. Set ``CONFIG_PATH`` to the ``config.yaml`` of your trained DLC project.
2. Set ``VIDEO_DIRECTORY`` to the folder containing your input videos.
3. Adjust ``GPU_INDEX``, ``BATCHSIZE``, and ``CPU_THREADS`` as needed.
4. Run:  ``python dlc_inference.py``

Output
------
- Per-video CSV files with keypoint coordinates and likelihood scores,
  saved alongside the source videos by DLC convention.
- (Optional) Labeled video files with keypoint overlays.
"""

import os

# ============================================================================
# Global Parameters — adjust these before running
# ============================================================================
GPU_INDEX    = 0       # Physical GPU index (as shown by nvidia-smi)
BATCHSIZE    = 12      # Frames per batch; 16-32 is often a good starting point
CPU_THREADS  = 16      # Threads for decode / BLAS / TF intra-op work
SKIP_LABELED = False   # Set True to skip labeled-video creation after analysis

# ---------------------------------------------------------------------------
# Path to the DeepLabCut project config.yaml produced during training.
# This file defines the model architecture, snapshot to use, body-part
# definitions, and other project metadata.
# ---------------------------------------------------------------------------
CONFIG_PATH = "<DLC_PROJECT_DIR>/config.yaml"
# Example: "/path/to/dlc-project-YYYY-MM-DD/config.yaml"

# ---------------------------------------------------------------------------
# Directory containing the video files to be analyzed.
# Only top-level files with the extensions listed below will be discovered
# (subdirectories are NOT searched).
# ---------------------------------------------------------------------------
VIDEO_DIRECTORY = "<VIDEO_DIR>"
# Example: "/path/to/videos/session_01"

VIDEO_EXTENSIONS = ['mp4', 'avi', 'mov']  # case-insensitive search

# ============================================================================
# Environment / Thread Setup  (set BEFORE importing heavy libraries)
# ============================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)

os.environ["OMP_NUM_THREADS"]        = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"]        = str(CPU_THREADS)
os.environ["OPENBLAS_NUM_THREADS"]   = str(CPU_THREADS)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(CPU_THREADS)
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

# ============================================================================
# Main Program
# ============================================================================
import glob

try:
    import cv2
    cv2.setNumThreads(int(os.getenv("OMP_NUM_THREADS", str(CPU_THREADS))))
except Exception:
    pass

import tensorflow as tf
import deeplabcut

# --- TensorFlow GPU configuration -------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        try:
            tf.config.threading.set_intra_op_parallelism_threads(CPU_THREADS)
            tf.config.threading.set_inter_op_parallelism_threads(2)
        except Exception:
            pass
        logical = tf.config.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPU(s), {len(logical)} Logical GPU(s) detected.")
        print("TensorFlow is configured to use the GPU.")
    except RuntimeError as e:
        print(f"[Warn] Memory growth setup failed: {e}")
else:
    print("No GPUs detected. Using CPU.")

# --- Collect videos (non-recursive; case-insensitive extensions) -------------
videos = []
patterns = []
for ext in VIDEO_EXTENSIONS:
    patterns.append(os.path.join(VIDEO_DIRECTORY, f'*.{ext}'))
    patterns.append(os.path.join(VIDEO_DIRECTORY, f'*.{ext.upper()}'))

for pat in patterns:
    videos.extend(glob.glob(pat))

if not videos:
    print("No videos found. Please check VIDEO_DIRECTORY and VIDEO_EXTENSIONS.")
    raise SystemExit(0)

print(f"Found {len(videos)} video(s) to process.")
print(f"Using physical GPU {GPU_INDEX} (visible as GPU:0 inside this process), "
      f"batchsize={BATCHSIZE}, threads={CPU_THREADS}")

# --- DLC Inference -----------------------------------------------------------
deeplabcut.analyze_videos(
    CONFIG_PATH,
    videos,
    shuffle=1,
    save_as_csv=True,
    batchsize=int(BATCHSIZE),
)

print("Video analysis completed and CSV files saved.")

# --- Labeled Video (optional) ------------------------------------------------
if not SKIP_LABELED:
    deeplabcut.create_labeled_video(
        CONFIG_PATH,
        videos,
        shuffle=1,
    )
    print("Labeled videos have been created.")
else:
    print("Skipping labeled video creation (SKIP_LABELED=True).")
