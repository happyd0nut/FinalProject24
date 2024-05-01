import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import random
import numpy as np
import time


class Target():
    def __init__(self, color, quadrant, screen_width=1200, screen_height=800):
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.quadrant = quadrant
        self.x = 0
        self.y = 0
        self.color = color
        self.respawn()

    def respawn(self):
        if self.quadrant == 2:
            self.x = random.randint(50, self.screen_width/2)
            self.y = random.randint(50, self.screen_height/2)
        elif self.quadrant == 1:
            self.x = random.randint(self.screen_width/2, self.screen_width)
            self.y = random.randint(50, self.screen_height/2)
        elif self.quadrant == 4:
            self.x = random.randint(self.screen_width/2, self.screen_width)
            self.y = random.randint(self.screen_height/2, self.screen_height)
        elif self.quadrant == 3:
            self.x = random.randint(50, self.screen_width/2)
            self.y = random.randint(self.screen_height/2, self.screen_height)
        else:
            self.x = random.randint(50, self.screen_width)
            self.y = random.randint(50, self.screen_height)
            
        

    def draw(self, image):
        cv2.circle(image, center=(self.x, self.y), radius=50, color=self.color, thickness=5)