import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import random
import numpy as np
import time

# Library Constants
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
DrawingUtil = mp.solutions.drawing_utils



class Game():
    def __init__(self, mode):
        
        self.mode = mode
        self.targets = []
        self.score = 0

        options = PoseLandmarkerOptions(
        base_options = BaseOptions(model_asset_path="data/pose_landmarker.task", 
                                   num_poses="1")
        running_mode = VisionRunningMode.VIDEO)
        self.video = cv2.VideoCapture(1)

    def __main__():
        print()