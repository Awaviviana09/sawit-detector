#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   @File Name:     config.py
   @Author:        Zahwa
   @Date:          2025/04/30
   @Description: Configuration file for palm oil maturity detection
-------------------------------------------------
"""
from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Source
SOURCES_LIST = ["Image", "Video", "Webcam"]

# DL model config
DETECTION_MODEL_DIR = ROOT / 'weights' / 'palm_oil_maturity'
MaturityModel = DETECTION_MODEL_DIR / "yolov8n.pt"
MaturityModel = str(MaturityModel)
# MaturityModel = "yolov8n.pt"


DETECTION_MODEL_LIST = [
    "yolov8n.pt"
]
