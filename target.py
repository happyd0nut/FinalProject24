import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import random
import numpy as np
import time


class Target():
    def __init__(self, color, screen_width=600, screen_height=400):
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.x = 0
        self.y = 0
        self.color = color
        self.respawn()

    def respawn(self):
        self.x = random.randint(50, self.screen_width)
        self.y = random.randint(50, self.screen_height)

    def draw(self, image):
        cv2.circle(image, (self.x, self.y), 25, self.color, 5)