#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility: Move Unlabeled Videos

Moves raw (unlabeled) .mp4 video files from one directory to another,
skipping any files that contain '_labeled' in their filename.  Useful
for separating DLC-labeled overlay videos from the original source files.

Usage
-----
Set ``PATH_SOURCE`` and ``PATH_DESTINATION`` below, then run:
    python move_unlabeled_videos.py
"""

import os
import shutil
from tqdm import tqdm

def move_unlabeled_videos(path_a, path_b):
    # Ensure destination directory exists
    os.makedirs(path_b, exist_ok=True)

    # List all files in Path A
    all_files = os.listdir(path_a)

    # Filter .mp4 files without '_labeled' in the filename
    video_files = [
        f for f in all_files
        if f.lower().endswith('.mp4') and '_labeled' not in f
    ]

    # Move files with a progress bar
    for file_name in tqdm(video_files, desc="Moving Unlabeled Videos", unit="file"):
        src = os.path.join(path_a, file_name)
        dst = os.path.join(path_b, file_name)
        try:
            shutil.move(src, dst)
        except Exception as e:
            print(f"Error moving {file_name}: {e}")

if __name__ == "__main__":
    # Source directory containing mixed labeled/unlabeled videos.
    PATH_SOURCE = "<SOURCE_VIDEO_DIR>"
    # Example: "/path/to/dlc_output/inference_results"

    # Destination directory for unlabeled (raw) videos only.
    PATH_DESTINATION = "<DESTINATION_VIDEO_DIR>"
    # Example: "/path/to/sorted_videos/raw_only"

    move_unlabeled_videos(PATH_SOURCE, PATH_DESTINATION)
