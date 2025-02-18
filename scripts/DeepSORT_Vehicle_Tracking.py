import sys
import os
from ultralytics import YOLO

# Importing DeepSORT
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from deep_sort.tracker import Tracker
# from deep_sort import DeepSort

video_path = os.path.join('.', 'base_video.mp4')

