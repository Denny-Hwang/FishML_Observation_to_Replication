"""
Video Preprocessing Pipeline

End-to-end preprocessing for raw surveillance/monitoring video recordings:
  1. Organizes raw videos into date-based and hourly sub-folders.
  2. Applies elliptical cropping mask (and optional secondary circular mask).
  3. Optionally resizes frames to target resolution.
  4. Repairs corrupted videos via FFmpeg stream-copy.
  5. Merges per-hour clip segments into single 1-hour videos.

Designed for multi-camera fish monitoring setups where the camera records
short clips (e.g., 3-min segments) that need to be cropped, cleaned,
and concatenated into standardized hourly files before DLC inference.

Input
-----
A directory tree containing raw .mp4 video files organized by date folders.
Files are expected to contain timestamp info in their names
(e.g., ``RecM03_YYYYMMDD_HHMMSS_*.mp4``).

Output
------
  ``<HOME>/Video_source/Processed/``    — individually cropped/resized clips.
  ``<HOME>/Video_source/Merged_1hour/`` — merged 1-hour video files.
  ``<HOME>/Video_source/Failed/``       — videos exceeding corruption threshold.

Prerequisites
-------------
- OpenCV (cv2)
- FFmpeg (system binary or static build)
"""

import cv2
import numpy as np
from tqdm import tqdm
import os
import time
from datetime import timedelta, datetime
import multiprocessing
import logging
import signal
import sys
from contextlib import redirect_stderr, redirect_stdout, contextmanager
import io
import subprocess
import tempfile
from logging.handlers import RotatingFileHandler
import shutil

# ----------------------- Configuration Parameters -----------------------
APPLY_SECONDARY_MASK = False      # Apply secondary circular mask if True
ENABLE_RESIZE = True             # Enable or disable resizing (Task 1)

# Cropping and (optional) resizing parameters
AXIS_X = 1600                  # Ellipse axis length for cropping
AXIS_Y = 1600
OFFSET_X = 20
OFFSET_Y = 0
RESIZE_WIDTH = 860             # Used if ENABLE_RESIZE is True
RESIZE_HEIGHT = 860            # Used if ENABLE_RESIZE is True

# Secondary mask parameters
SEC_MASK_CENTER_X = 432
SEC_MASK_CENTER_Y = 424
SEC_MASK_RADIUS = 38

FPS_VALUE = 20

# Threshold for corrupted frames (10%)
CORRUPTED_FRAME_THRESHOLD = 0.1

# Multiprocessing CPU usage percentage
CPU_PERCENTAGE = 40

# ----------------------- Folder Paths -----------------------
# Root directory containing raw video date-folders.
# Expected structure:  HOME_PATH/<date_folder>/<RecM*.mp4 files>
HOME_PATH = "<RAW_VIDEO_ROOT_DIR>"
# Example: "/path/to/experiment_recordings/2024_session"

INPUT_BASE_FOLDER_PATH = HOME_PATH
# Revised folder structure for clarity:
OUTPUT_BASE_FOLDER_PATH = os.path.join(HOME_PATH, "Video_source")
PROCESSED_FOLDER_PATH = os.path.join(OUTPUT_BASE_FOLDER_PATH, "Processed")
MERGED_1HOUR_FOLDER_PATH = os.path.join(OUTPUT_BASE_FOLDER_PATH, "Merged_1hour")
FAILED_PROCESSING_DIR = os.path.join(OUTPUT_BASE_FOLDER_PATH, "Failed")

# -------------------------- Logging Setup -------------------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Rotating file handler: 5 files, each up to 10MB
file_handler = RotatingFileHandler("video_processing.log", maxBytes=10**7, backupCount=5)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console output
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ---------------------- Graceful Shutdown Setup -------------------------
shutdown_flag = multiprocessing.Event()

def signal_handler(signum, frame):
    """Handle signals to initiate graceful shutdown."""
    logging.warning(f"Received signal {signum}. Initiating graceful shutdown...")
    shutdown_flag.set()

signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination signals

# -------------------- Helper: Suppress FFmpeg Output --------------------
@contextmanager
def suppress_ffmpeg_output():
    """Suppress FFmpeg's verbose output by redirecting stderr and stdout."""
    with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
        yield

# -------------------- Helper: Multiprocessing Init ----------------------
def init_worker():
    """Initializer for each worker process in the pool."""
    import os, sys
    sys.stderr.flush()
    sys.stdout.flush()
    devnull = open(os.devnull, 'w')
    os.dup2(devnull.fileno(), sys.stderr.fileno())

# ------------------ Helper: Organize Raw Videos (Steps 0 & 1) ------------------
def create_rec_subfolders(date_folder):
    """
    Create 'RecM' and 'RecS' subfolders inside a given date folder.
    """
    recm_path = os.path.join(date_folder, "RecM")
    recs_path = os.path.join(date_folder, "RecS")
    os.makedirs(recm_path, exist_ok=True)
    os.makedirs(recs_path, exist_ok=True)
    return recm_path, recs_path

def get_hour_from_filename(filename):
    """
    Extract the hour (00-23) from a filename formatted as 'RecM03_YYYYMMDD_HHMMSS_...mp4'.
    """
    parts = filename.split("_")
    if len(parts) < 3:
        return None
    time_part = parts[2]  # Expected format 'HHMMSS'
    if len(time_part) < 6:
        return None
    try:
        hour = int(time_part[:2])
        return hour
    except ValueError:
        return None

def move_videos_to_rec_subfolders(date_folder):
    """
    Move videos with 'RecM' or 'RecS' in their names into corresponding subfolders.
    Then, further separate 'RecM' videos into hour-based subfolders.
    """
    recm_path, recs_path = create_rec_subfolders(date_folder)

    # Move videos into RecM or RecS subfolders
    for f in os.listdir(date_folder):
        if f.lower().endswith(".mp4"):
            full_path = os.path.join(date_folder, f)
            if "RecM" in f:
                dest_path = os.path.join(recm_path, f)
                if os.path.abspath(full_path) != os.path.abspath(dest_path):
                    shutil.move(full_path, dest_path)
            elif "RecS" in f:
                dest_path = os.path.join(recs_path, f)
                if os.path.abspath(full_path) != os.path.abspath(dest_path):
                    shutil.move(full_path, dest_path)

    # Separate RecM videos by hour
    for f in os.listdir(recm_path):
        if f.lower().endswith(".mp4"):
            hour = get_hour_from_filename(f)
            if hour is not None:
                parts = f.split("_")
                date_part = parts[1] if len(parts) >= 2 else "unknown"
                time_folder_suffix = hour + 1  # 1-based indexing
                subfolder_name = f"{date_part[2:]}_time_{time_folder_suffix:02d}"
                subfolder_path = os.path.join(recm_path, subfolder_name)
                os.makedirs(subfolder_path, exist_ok=True)
                src_path = os.path.join(recm_path, f)
                dest_path = os.path.join(subfolder_path, f)
                if os.path.abspath(src_path) != os.path.abspath(dest_path):
                    shutil.move(src_path, dest_path)

# ----------------------- Mask & Processing Functions ---------------------
def calculate_center(frame_width, frame_height, offset_x, offset_y):
    """Calculate the center of the ellipse based on frame dimensions and offsets."""
    center_x = frame_width // 2 + offset_x
    center_y = frame_height // 2 + offset_y
    return center_x, center_y

def create_primary_mask(frame_width, frame_height, center_x, center_y, axis_x, axis_y):
    """Create an elliptical primary mask."""
    mask = np.ones((frame_height, frame_width), dtype=np.uint8) * 255
    cv2.ellipse(mask, (center_x, center_y), (axis_x // 2, axis_y // 2), 0, 0, 360, 0, -1)
    return mask

def apply_secondary_circular_mask(image, center_x, center_y, radius):
    """Apply a circular secondary mask to the image."""
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, (255, 255, 255), -1)
    cv2.bitwise_or(image, mask, dst=image)
    return image

# --------------------- Attempt to Repair Video --------------------------
def repair_video(input_video_path):
    """
    Attempt to repair a corrupted video using FFmpeg stream copy.
    Returns the path to the repaired video if successful; otherwise, None.
    """
    try:
        # Path to FFmpeg executable.
        # If FFmpeg is on your system PATH, you can simply use "ffmpeg".
        ffmpeg_path = "<FFMPEG_PATH>"
        # Example: "/usr/bin/ffmpeg"  or  "ffmpeg"
        if not os.path.isfile(ffmpeg_path):
            logging.error(f"FFmpeg executable not found at {ffmpeg_path}.")
            return None
        if not os.access(ffmpeg_path, os.X_OK):
            logging.error(f"FFmpeg at {ffmpeg_path} is not executable.")
            return None

        # Temporary directory for FFmpeg repaired files.
        custom_tmp_dir = "<FFMPEG_TMP_DIR>"
        # Example: "/tmp/ffmpeg_repair"
        os.makedirs(custom_tmp_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=custom_tmp_dir) as temp_repaired:
            temp_repaired_path = temp_repaired.name

        command = [ffmpeg_path, '-y', '-i', input_video_path, '-c', 'copy', temp_repaired_path]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            logging.error(f"FFmpeg failed to repair video: {input_video_path}")
            logging.error(f"FFmpeg stderr: {result.stderr}")
            os.remove(temp_repaired_path)
            return None

        if os.path.getsize(temp_repaired_path) > 0:
            logging.info(f"Video repaired successfully: {input_video_path}")
            return temp_repaired_path
        else:
            logging.error(f"Repaired video is empty: {input_video_path}")
            os.remove(temp_repaired_path)
            return None
    except Exception as e:
        logging.error(f"Unexpected error during video repair for {input_video_path}: {e}")
        if 'temp_repaired_path' in locals() and os.path.exists(temp_repaired_path):
            os.remove(temp_repaired_path)
        return None

# --------------------- Process Single Video (Mask, Crop, [Resize]) ---------------------
def process_video(input_video_path, output_video_path, apply_secondary_mask, failed_processing_dir,
                  threshold=CORRUPTED_FRAME_THRESHOLD):
    """
    Process a single video: repair (if possible), apply masks, crop, optionally resize,
    and handle corrupted frames. If the corrupted frame ratio exceeds the threshold,
    the video is skipped and copied to the failed processing directory.
    """
    repaired_video_path = repair_video(input_video_path)
    if repaired_video_path:
        video_to_process = repaired_video_path
        delete_repaired = True
        logging.info(f"Video repaired: {input_video_path}")
    else:
        video_to_process = input_video_path
        delete_repaired = False
        logging.warning(f"Using original video for processing (may be corrupted): {input_video_path}")

    with suppress_ffmpeg_output():
        cap = cv2.VideoCapture(video_to_process)
    if not cap.isOpened():
        logging.error(f"Failed to open video for processing: {video_to_process}")
        if delete_repaired and os.path.exists(repaired_video_path):
            os.remove(repaired_video_path)
        return False

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = FPS_VALUE
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    center_x, center_y = calculate_center(frame_width, frame_height, OFFSET_X, OFFSET_Y)
    primary_mask = create_primary_mask(frame_width, frame_height, center_x, center_y, AXIS_X, AXIS_Y)
    primary_mask_bgr = cv2.cvtColor(primary_mask, cv2.COLOR_GRAY2BGR)

    # Calculate cropping rectangle
    crop_left = max(center_x - AXIS_X // 2, 0)
    crop_top = max(center_y - AXIS_Y // 2, 0)
    crop_right = min(center_x + AXIS_X // 2, frame_width)
    crop_bottom = min(center_y + AXIS_Y // 2, frame_height)
    cropped_width = crop_right - crop_left
    cropped_height = crop_bottom - crop_top

    # Determine output dimensions based on the ENABLE_RESIZE flag
    if ENABLE_RESIZE:
        output_width, output_height = RESIZE_WIDTH, RESIZE_HEIGHT
    else:
        output_width, output_height = cropped_width, cropped_height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))
    if not out.isOpened():
        logging.error(f"Failed to open VideoWriter for: {output_video_path}")
        cap.release()
        if delete_repaired and os.path.exists(repaired_video_path):
            os.remove(repaired_video_path)
        return False

    placeholder_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
    last_good_frame = placeholder_frame.copy()
    corrupted_frame_count = 0

    for frame_idx in tqdm(range(total_frames), desc=f"Processing {os.path.basename(input_video_path)}", leave=False):
        if shutdown_flag.is_set():
            logging.info("Shutdown flag detected. Stopping video processing.")
            break

        ret, frame = cap.read()
        if not ret:
            corrupted_frame_count += 1
            frame_to_process = last_good_frame.copy()
            logging.error(f"Frame {frame_idx} read error in {input_video_path}; using last good frame or placeholder.")
        else:
            frame_to_process = frame
            last_good_frame = frame.copy()

        current_frame_number = frame_idx + 1
        corrupted_ratio = corrupted_frame_count / current_frame_number
        if corrupted_ratio > threshold:
            logging.warning(f"Corrupted frame ratio {corrupted_ratio:.2%} exceeded threshold in {input_video_path}. Skipping video.")
            cap.release()
            out.release()
            if delete_repaired and os.path.exists(repaired_video_path):
                os.remove(repaired_video_path)
            if os.path.exists(output_video_path):
                os.remove(output_video_path)
            os.makedirs(failed_processing_dir, exist_ok=True)
            destination_path = os.path.join(failed_processing_dir, os.path.basename(input_video_path))
            try:
                shutil.copy2(input_video_path, destination_path)
                logging.info(f"Copied failed video to: {destination_path}")
            except Exception as e:
                logging.error(f"Failed to copy video to failed_processing_dir: {e}")
            return False

        try:
            cv2.bitwise_or(frame_to_process, primary_mask_bgr, dst=frame_to_process)
            cropped_frame = frame_to_process[crop_top:crop_bottom, crop_left:crop_right]
            if cropped_frame.size == 0:
                logging.error(f"Cropped frame has invalid dimensions at frame {frame_idx} in {input_video_path}. Using placeholder.")
                final_frame = placeholder_frame.copy()
            else:
                if ENABLE_RESIZE:
                    processed_frame = cv2.resize(cropped_frame, (output_width, output_height), interpolation=cv2.INTER_LANCZOS4)
                else:
                    processed_frame = cropped_frame
                if apply_secondary_mask:
                    final_frame = apply_secondary_circular_mask(processed_frame, SEC_MASK_CENTER_X, SEC_MASK_CENTER_Y, SEC_MASK_RADIUS)
                else:
                    final_frame = processed_frame
        except Exception as e:
            logging.error(f"Error processing frame {frame_idx} in {input_video_path}: {e}")
            final_frame = placeholder_frame.copy()

        out.write(final_frame)

    cap.release()
    out.release()
    if delete_repaired and os.path.exists(repaired_video_path):
        os.remove(repaired_video_path)

    if shutdown_flag.is_set():
        logging.info(f"Processing of video {input_video_path} was interrupted.")
        return False
    else:
        if current_frame_number > 0:
            final_corrupted_ratio = corrupted_frame_count / current_frame_number
            if final_corrupted_ratio > threshold:
                logging.warning(f"Final corrupted frame ratio {final_corrupted_ratio:.2%} exceeded threshold in {input_video_path}.")
                if os.path.exists(output_video_path):
                    os.remove(output_video_path)
                os.makedirs(failed_processing_dir, exist_ok=True)
                destination_path = os.path.join(failed_processing_dir, os.path.basename(input_video_path))
                try:
                    shutil.copy2(input_video_path, destination_path)
                    logging.info(f"Copied failed video to: {destination_path}")
                except Exception as e:
                    logging.error(f"Failed to copy video to failed_processing_dir: {e}")
                return False

        logging.info(f"Processed and saved video: {output_video_path}")
        return True

# --------------------- Merge Videos Helper ---------------------
def merge_videos(video_paths, output_merged_video_path, fps=FPS_VALUE, target_dims=None):
    """
    Merge multiple videos into a single video.
    If target_dims is None, determine the dimensions from the first video.
    """
    if not video_paths:
        logging.warning(f"No videos to merge for {output_merged_video_path}")
        return False

    if target_dims is None:
        with suppress_ffmpeg_output():
            cap = cv2.VideoCapture(video_paths[0])
        if not cap.isOpened():
            logging.error(f"Failed to open first video for merging: {video_paths[0]}")
            return False
        ret, frame = cap.read()
        if not ret:
            logging.error(f"Failed to read frame from first video for merging: {video_paths[0]}")
            cap.release()
            return False
        target_dims = (frame.shape[1], frame.shape[0])
        cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_merged_video_path, fourcc, fps, target_dims)
    if not out.isOpened():
        logging.error(f"Failed to open VideoWriter for merged video: {output_merged_video_path}")
        return False

    for video_path in tqdm(video_paths, desc=f"Merging into {os.path.basename(output_merged_video_path)}", leave=False):
        with suppress_ffmpeg_output():
            cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video for merging: {video_path}")
            cap.release()
            continue
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame.shape[1] != target_dims[0] or frame.shape[0] != target_dims[1]:
                frame = cv2.resize(frame, target_dims, interpolation=cv2.INTER_LANCZOS4)
            out.write(frame)
        cap.release()

    out.release()
    logging.info(f"Merged video saved as: {output_merged_video_path}")
    return True

# --------------------- Process Hour Subfolder ---------------------
def process_hour_subfolder(hour_subfolder_path):
    """
    Process all '.mp4' videos in a given hour subfolder.
    Returns a list of processed video paths (to be merged later).
    """
    processed_video_paths = []
    videos = [f for f in os.listdir(hour_subfolder_path) if f.endswith('.mp4')]
    if not videos:
        return []

    # Create a mirror output subfolder under PROCESSED_FOLDER_PATH
    relative_path = os.path.relpath(hour_subfolder_path, INPUT_BASE_FOLDER_PATH)
    output_hour_subfolder = os.path.join(PROCESSED_FOLDER_PATH, relative_path)
    os.makedirs(output_hour_subfolder, exist_ok=True)

    for video_name in videos:
        input_video_path = os.path.join(hour_subfolder_path, video_name)
        output_video_name = (f"({RESIZE_WIDTH}x{RESIZE_HEIGHT})_{video_name}"
                             if ENABLE_RESIZE else f"(cropped)_{video_name}")
        output_video_path = os.path.join(output_hour_subfolder, output_video_name)

        success = process_video(
            input_video_path=input_video_path,
            output_video_path=output_video_path,
            apply_secondary_mask=APPLY_SECONDARY_MASK,
            failed_processing_dir=FAILED_PROCESSING_DIR
        )
        if success:
            processed_video_paths.append(output_video_path)

    return processed_video_paths

# --------------------- Main Workflow ---------------------
def get_cpu_count_percentage(percentage=40):
    """Determine the number of processes based on the CPU count and given percentage."""
    cpu_count = multiprocessing.cpu_count()
    return max(1, cpu_count * percentage // 100)

def main():
    # 0 & 1: Organize videos into 'RecM' and 'RecS' subfolders and further into hour-based subfolders
    date_folders = [os.path.join(INPUT_BASE_FOLDER_PATH, d)
                    for d in os.listdir(INPUT_BASE_FOLDER_PATH)
                    if os.path.isdir(os.path.join(INPUT_BASE_FOLDER_PATH, d))]
    if not date_folders:
        logging.error(f"No date folders found in: {INPUT_BASE_FOLDER_PATH}")
        return

    for date_folder in date_folders:
        move_videos_to_rec_subfolders(date_folder)

    # 2: Collect all hour-based RecM subfolders
    recm_hour_subfolders = []
    for date_folder in date_folders:
        recm_folder = os.path.join(date_folder, "RecM")
        if not os.path.isdir(recm_folder):
            continue
        for subf in os.listdir(recm_folder):
            potential_hour_path = os.path.join(recm_folder, subf)
            if os.path.isdir(potential_hour_path) and "_time_" in subf:
                recm_hour_subfolders.append(potential_hour_path)

    if not recm_hour_subfolders:
        logging.warning("No hour-based RecM subfolders found to process.")
        return

    # 3: Process each hour subfolder in parallel
    num_processes = get_cpu_count_percentage(CPU_PERCENTAGE)
    logging.info(f"Starting parallel processing of hour subfolders with {num_processes} processes.")
    start_time = time.time()
    subfolder_to_processed_videos = {}

    with multiprocessing.Pool(processes=num_processes, initializer=init_worker) as pool:
        try:
            results = list(tqdm(pool.imap(process_hour_subfolder, recm_hour_subfolders),
                                total=len(recm_hour_subfolders),
                                desc="Processing hour subfolders"))
            for subfolder_path, processed_paths in zip(recm_hour_subfolders, results):
                subfolder_to_processed_videos[subfolder_path] = processed_paths
        except KeyboardInterrupt:
            logging.warning("KeyboardInterrupt detected. Terminating pool.")
            pool.terminate()
            pool.join()
            sys.exit(1)
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            pool.terminate()
            pool.join()
            sys.exit(1)

    # 4: Merge processed videos for each hour subfolder into a single 1-hour clip
    os.makedirs(MERGED_1HOUR_FOLDER_PATH, exist_ok=True)
    merge_target_dims = (RESIZE_WIDTH, RESIZE_HEIGHT) if ENABLE_RESIZE else None

    for hour_subfolder, processed_paths in subfolder_to_processed_videos.items():
        if not processed_paths:
            continue
        subfolder_name = os.path.basename(hour_subfolder)
        merged_video_filename = f"{subfolder_name}_merged.mp4"
        merged_video_path = os.path.join(MERGED_1HOUR_FOLDER_PATH, merged_video_filename)
        merge_videos(processed_paths, merged_video_path, fps=FPS_VALUE, target_dims=merge_target_dims)

    processing_time = time.time() - start_time
    formatted_time = str(timedelta(seconds=int(processing_time)))
    if shutdown_flag.is_set():
        logging.info("Processing was interrupted by a signal.")
    else:
        logging.info("All videos processed and merged (for 1-hour groups).")
    logging.info(f"Total processing time: {formatted_time}")

# --------------------- Entry Point ---------------------
if __name__ == '__main__':
    os.makedirs(PROCESSED_FOLDER_PATH, exist_ok=True)
    os.makedirs(MERGED_1HOUR_FOLDER_PATH, exist_ok=True)
    os.makedirs(FAILED_PROCESSING_DIR, exist_ok=True)
    
    main()
