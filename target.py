import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import random
import numpy as np
import time

GREEN = (0,250,0)

class Target():

    def __init__(self, x=0, y=0, color=GREEN, quadrant=0, screen_width=1200, screen_height=700):
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.x = x
        self.y = y
        self.color = color
        self.quadrant = quadrant
        
        self.respawn()


    def respawn(self):
        if self.quadrant == 2:
            self.x = random.randint(300, self.screen_width/2)
            self.y = random.randint(50, self.screen_height/2)
        elif self.quadrant == 1:
            self.x = random.randint(self.screen_width/2, self.screen_width/2 + 300)
            self.y = random.randint(50, self.screen_height/2)
        elif self.quadrant == 4:
            self.x = random.randint(self.screen_width/2, self.screen_width-300)
            self.y = random.randint(self.screen_height-70, self.screen_height-30)
        elif self.quadrant == 3:
            self.x = random.randint(350, self.screen_width/2)
            self.y = random.randint(self.screen_height-70, self.screen_height-30)
        else:
            self.x = random.randint(50, self.screen_width)
            self.y = random.randint(50, self.screen_height)
            

    def draw(self, image):
        cv2.circle(image, center=(self.x, self.y), radius=40, color=self.color, thickness=5)