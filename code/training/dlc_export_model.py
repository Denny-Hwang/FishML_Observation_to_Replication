#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DLC Model Export (TensorRT)

Exports a trained DeepLabCut model to TensorRT format for optimized
GPU inference.  The exported model is saved under an ``exported-models``
folder inside the DLC project directory.

Prerequisites
-------------
- DeepLabCut with TensorRT support
- TensorFlow + TensorRT libraries
- A fully trained DLC project (completed ``train_network``)

Usage
-----
1. Set ``CONFIG_PATH`` to your DLC project's ``config.yaml``.
2. Adjust ``GPU_INDEX`` if needed.
3. Run:  ``python dlc_export_model.py``
"""

import deeplabcut
import os

# ============================================================================
# Configuration
# ============================================================================
# Path to the DLC project's config.yaml file.
CONFIG_PATH = "<DLC_PROJECT_DIR>/config.yaml"
# Example: "/path/to/dlc-project-YYYY-MM-DD/config.yaml"

# GPU to use for the export process.
GPU_INDEX = 0

# DLC shuffle index (must match the shuffle used during training).
SHUFFLE = 1

# ============================================================================
# Export
# ============================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)

print("Starting TensorRT model export...")

deeplabcut.export_model(
    CONFIG_PATH,
    shuffle=SHUFFLE,
    tensorrt=True,
)

print("\nTensorRT model exported successfully!")
print("Look for a new 'exported-models' folder inside your DLC project directory.")
