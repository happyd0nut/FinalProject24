import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import vision
import cv2
import random
import time
from target import Target

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
        
        # Create PoseLandmarker detector
        base_options = BaseOptions(model_asset_path="data/pose_landmarker_full.task")
        options = PoseLandmarkerOptions(base_options = base_options, num_poses = 2,
                                        output_segmentation_masks = True)
        self.detector = PoseLandmarker.create_from_options(options)

        # Start video
        self.video = cv2.VideoCapture(1)


    def draw_landmarks(self, image, detection_result):
        
        # Get list of poses detected
        pose_landmarks_list = detection_result.pose_landmarks

        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks])
            DrawingUtil.draw_landmarks(image,
                                        pose_landmarks_proto,
                                        solutions.pose.POSE_CONNECTIONS,
                                        solutions.drawing_styles.get_default_pose_landmarks_style())


    def label_structures(self, image, detection_result):
        pass


    def run(self):

        while self.video.isOpened():
            
            # Get frame of video feed
            frame = self.video.read()[1]

            # Turn to RGB and flip image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)

            # Use PoseLandmarker to detect poses
            to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = self.detector.detect(to_detect)

            # Draw landmarks on poses
            self.draw_landmarks(image, results)

            # Display image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow("Pose Tracking", image)

            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

        self.video.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    g = Game(0)
    g.run()