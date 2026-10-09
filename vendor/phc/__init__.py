"""Headless subset of Perpetual Humanoid Control (PHC) used by robo-saber/track.py.

Adapted from https://github.com/ZhengyiLuo/PHC (BSD 3-Clause Clear, see LICENSE).
"""

import os

ASSET_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(ASSET_DIR)), "data", "phc")
