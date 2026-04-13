# FishML: Observation to Replication

This repository accompanies the paper:

> **From observation to replication: machine-learning-driven quantification and replication of fine-scale fish kinematics and behavior**
>
> Hwang, S., Li, H., Janak, J.M., Jung, H., Lu, Z., Deng, Z.D. (2026). From observation to replication: machine-learning-driven quantification and replication of fine-scale fish kinematics and behavior. Ecological Informatics. https://doi.org/10.1016/j.ecoinf.2026.103760

A DeepLabCut-based pipeline for fish swimming kinematics analysis. This repository provides end-to-end tools for video preprocessing, pose estimation model training, batch inference, and quantitative movement analysis including body curvature, tail-beat frequency (TBF), swimming speed, and diel activity patterns.

## Overview

FishML extracts continuous swimming kinematics from long-term behavioral video recordings using [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) for markerless pose estimation. The pipeline processes raw video footage through a series of modular steps to produce per-frame movement metrics and temporally aggregated statistics suitable for downstream analysis or publication.

This work addresses two key challenges: (1) determining how long fish must be monitored to obtain reliable behavioral metrics, and (2) physically reproducing natural swimming motion for controlled experimentation. Using juvenile white sturgeon as a case study, we derived metric-specific monitoring thresholds (8–17 days for reliability > 0.8) and developed a hardware-in-the-loop simulator that reconstructs swimming kinematics with high fidelity (r ≥ 0.98).

<p align="center">
  <img src="figures/Figure2.png" width="1200">
</p>

The analysis pipeline computes the following kinematic metrics from pose-estimated keypoints:

- **Moving distance and speed** (px and mm) for each tracked body part
- **Body curvature** via circular fitting (Pratt, HyperLS, or three-point methods)
- **Tail-beat frequency (TBF)** from zero-crossing detection on tail keypoints
- **Body and segment lengths** (head-to-tail and sub-segment)
- **Bending classification** (minimal, normal, extreme) based on curvature thresholds
- **Diel (day/night) activity patterns** with temporal windowed statistics

## Repository Structure

```
FishML_Observation_to_Replication/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── code/
│   ├── preprocessing/
│   │   └── video_preprocessing.py      # Video organization, masking, cropping, repair, merging
│   ├── training/
│   │   ├── dlc_training.py             # DLC model training, evaluation, and snapshot selection
│   │   └── dlc_export_model.py         # TensorRT model export for optimized inference
│   ├── inference/
│   │   └── dlc_inference.py            # Batch video analysis with GPU/thread optimization
│   ├── analysis/
│   │   ├── step1_pose_tracking_analysis.py  # Pose data → per-frame movement metrics
│   │   ├── step2_windowed_statistics.py     # Temporal window aggregation and frequency analysis
│   │   ├── step3_diel_pattern_analysis.py   # Day/night activity pattern analysis
│   │   └── step4_paper_figures.py           # Publication-ready figure generation
│   └── utils/
│       └── move_unlabeled_videos.py    # File organization utility
├── data/examples/raw_images/           # Example labeled images for training
├── models/                             # DLC model configuration and training data
│   ├── config.yaml
│   ├── pose_cfg.yaml
│   └── training/
│       ├── CollectedData_SH.csv
│       ├── CollectedData_SH.h5
│       ├── CombinedEvaluation-results.csv
│       └── learning_stats.csv
└── videos/examples/                    # Example video files
```

## Installation

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- FFmpeg (for video preprocessing)

### Setup

```bash
git clone https://github.com/Denny-Hwang/FishML_Observation_to_Replication.git
cd FishML_Observation_to_Replication
pip install -r requirements.txt
```

Key dependencies include DeepLabCut, TensorFlow, OpenCV, NumPy, pandas, SciPy, and matplotlib. See `requirements.txt` for the full list.

## Usage

Each script uses global parameters defined at the top of the file. Before running, update the placeholder paths (marked with `<PLACEHOLDER>`) to match your local environment.

### 1. Video Preprocessing

```bash
python code/preprocessing/video_preprocessing.py
```

Organizes raw video files, applies region-of-interest masking and cropping, repairs corrupted frames, and merges short clips into analysis-ready segments.

**Placeholders to configure:** `<RAW_VIDEO_ROOT_DIR>`, `<FFMPEG_PATH>`, `<FFMPEG_TMP_DIR>`

### 2. Model Training

```bash
python code/training/dlc_training.py
```

Creates a training dataset from labeled frames, trains the DeepLabCut network, evaluates snapshots at regular intervals, and selects the best-performing model based on test error.

**Placeholders to configure:** `<DLC_PROJECT_DIR>`, `<VIDEO_DIR>`

### 3. Batch Inference

```bash
python code/inference/dlc_inference.py
```

Runs pose estimation on all videos in a directory with configurable GPU selection, batch size, and CPU thread pinning. Optionally generates labeled overlay videos.

**Placeholders to configure:** `<DLC_PROJECT_DIR>`, `<VIDEO_DIR>`, `<OUTPUT_DIR>`

### 4. Post-Processing Analysis

Run the analysis scripts sequentially:

```bash
# Step 1: Extract per-frame kinematics from DLC output (.h5/.csv)
python code/analysis/step1_pose_tracking_analysis.py

# Step 2: Compute windowed statistics (e.g., 10-second windows)
python code/analysis/step2_windowed_statistics.py

# Step 3: Analyze diel (day/night) activity patterns
python code/analysis/step3_diel_pattern_analysis.py

# Step 4: Generate publication figures
python code/analysis/step4_paper_figures.py
```

**Placeholders to configure:**
- Steps 1–3: `<INPUT_H5_OR_DIR>`, `<OUTPUT_CSV_OR_DIR>`
- Step 4: `<INPUT_CSV_OR_DIR>`, `<OUTPUT_FIGURE_DIR>`

### Optional Utilities

```bash
# Export trained model to TensorRT format
python code/training/dlc_export_model.py

# Organize unlabeled video files
python code/utils/move_unlabeled_videos.py
```

## Configuration

All scripts use global variables for configuration rather than command-line arguments. Key parameters include:

| Parameter | Script | Description |
|-----------|--------|-------------|
| `LIKELIHOOD_THRESHOLD` | step1 | Minimum confidence for keypoint detection (default: 0.6) |
| `DEFAULT_FPS` | step1 | Video frame rate (default: 20) |
| `JITTER_THRESHOLD` | step1 | Minimum movement threshold in pixels (default: 1) |
| `WINDOW_SECONDS` | step2 | Temporal window size in seconds (default: 10.0) |
| `BL_MM` | step2 | Body length in mm for BL/s speed conversion |
| `GPU_INDEX` | inference | GPU device index for CUDA |
| `BATCHSIZE` | inference | Batch size for DLC inference |
| `MAX_PROCESSES` | step1 | Number of parallel workers for batch processing |

## Citation

If you use this code in your research, please cite:

> Hwang, S., Li, H., Janak, J.M., Jung, H., Lu, Z., & Deng, Z.D. (2026). From observation to replication: machine-learning-driven quantification and replication of fine-scale fish kinematics and behavior. *Ecological Informatics*. *(submitted)*

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Author

**Sungjoo Hwang**
Pacific Northwest National Laboratory
sungjoo.hwang@pnnl.gov
GitHub: [Denny-Hwang](https://github.com/Denny-Hwang)

## Acknowledgments

This research was funded by the U.S. Department of Energy (DOE) Water Power Technologies Office. The study was conducted by Pacific Northwest National Laboratory, which is operated by Battelle for DOE under Contract DE-AC05-76RL01830.
