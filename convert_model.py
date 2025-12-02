#!/usr/bin/env python
"""
Helper script for converting YOLOv8 models to ONNX format.

This is a convenience wrapper around the model_converter module.
"""

import sys
from src.model_converter import main

if __name__ == "__main__":
    sys.exit(main())
