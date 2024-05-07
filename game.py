import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import vision
import cv2
from pygame import mixer
import time
from target import Target

# Library Constants
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkPoints = mp.solutions.pose.PoseLandmark
PoseLandmarkConnections = mp.tasks.vision.PoseLandmarksConnections
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
DrawingUtil = mp.solutions.drawing_utils

RED = (250,0,0)
GREEN = (0,250,0)
BLUE = (0,0,250)

class Game():
    def __init__(self, mode):
        
        self.mode = mode
        self.score = 0

        self.rh_target = Target(color=RED, quadrant=1)
        self.lh_target = Target(color=BLUE, quadrant=2)
        self.rf_target = Target(color=RED, quadrant=4)
        self.lf_target = Target(color=BLUE, quadrant=3)

        self.targets = [self.rh_target, self.lh_target, self.rf_target, self.lf_target]
        
        # Create PoseLandmarker detector
        base_options = BaseOptions(model_asset_path="data/pose_landmarker_full.task")
        options = PoseLandmarkerOptions(base_options = base_options, num_poses = 2,
                                        output_segmentation_masks = True)
        self.detector = PoseLandmarker.create_from_options(options)

        # Start video
        self.video = cv2.VideoCapture(1)

        # Create Sound object
        mixer.init()
        mixer.music.load("data/music/just_dance_audio.mp3")
        mixer.music.set_volume(0.7)
        mixer.music.play()


    def draw_landmarks(self, image, detection_result):
        """
        Function adapted from MediaPipe's provided example code
        for the PoseLandmarker library in Google CoLab.
        """
        
        # PoseLandmarker results gets a list of poses detected
        pose_landmarks_list = detection_result.pose_landmarks

        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks])
            DrawingUtil.draw_landmarks(image,
                                        pose_landmarks_proto,
                                        solutions.pose.POSE_CONNECTIONS,
                                        solutions.drawing_styles.get_default_pose_landmarks_style())


    def check_target_match(self, image, detection_result):
        
        # Get image info and list of pose landmarks detected from PoseLandmarker detector
        imageHeight, imageWidth = image.shape[:2]
        pose_landmarks_list = detection_result.pose_landmarks
        pose_segmentation_list = detection_result.segmentation_masks

        # Loop through each pose landmark
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]
            pose_seg_image = pose_segmentation_list[idx]
            
            # Extract the right hand from pose landmark (R/L values flipped because of mirrored image)
            right_hand = pose_landmarks[PoseLandmarkPoints.LEFT_INDEX.value]
            left_hand = pose_landmarks[PoseLandmarkPoints.RIGHT_INDEX.value]
            right_foot = pose_landmarks[PoseLandmarkPoints.LEFT_ANKLE.value]
            left_foot = pose_landmarks[PoseLandmarkPoints.RIGHT_ANKLE.value]

            # Get coordinates from desired points
            pixelCoord_r_hand = DrawingUtil._normalized_to_pixel_coordinates(right_hand.x,
                                                                      right_hand.y,
                                                                      imageWidth,
                                                                      imageHeight)
            pixelCoord_l_hand = DrawingUtil._normalized_to_pixel_coordinates(left_hand.x,
                                                                      left_hand.y,
                                                                      imageWidth,
                                                                      imageHeight)
            pixelCoord_r_foot = DrawingUtil._normalized_to_pixel_coordinates(right_foot.x,
                                                                      right_foot.y,
                                                                      imageWidth,
                                                                      imageHeight)
            pixelCoord_l_foot = DrawingUtil._normalized_to_pixel_coordinates(left_foot.x,
                                                                      left_foot.y,
                                                                      imageWidth,
                                                                      imageHeight)
            
            # Draw circle around desired points and check intercept
            if pixelCoord_r_hand and pixelCoord_l_hand:
                cv2.circle(image, (pixelCoord_r_hand[0], pixelCoord_r_hand[1]), 50, RED, 5)
                cv2.circle(image, (pixelCoord_l_hand[0], pixelCoord_l_hand[1]), 50, BLUE, 5)
                cv2.circle(image, (pixelCoord_r_foot[0], pixelCoord_r_foot[1]), 50, RED, 5)
                cv2.circle(image, (pixelCoord_l_foot[0], pixelCoord_l_hand[1]), 50, BLUE, 5)
                
                r_hand_int = self.check_target_intercept(pixelCoord_r_hand[0], pixelCoord_r_hand[1], self.rh_target)
                l_hand_int = self.check_target_intercept(pixelCoord_l_hand[0], pixelCoord_l_hand[1], self.lh_target)
                r_foot_int = self.check_target_intercept(pixelCoord_r_foot[0], pixelCoord_r_foot[1], self.rf_target)
                l_foot_int = self.check_target_intercept(pixelCoord_l_foot[0], pixelCoord_l_foot[1], self.lf_target)

                if r_hand_int and l_hand_int and r_foot_int and l_foot_int:
                    self.score += 1
                    for target in self.targets:
                        target.respawn()


    def check_target_intercept(self, point_x, point_y, target):

        target_x = target.x
        target_y = target.y

        # Respawn target if point is hit
        if (point_x < target_x + 10 and point_x > target_x - 10) and (point_y < target_y + 10 and point_y > target_y - 10):
            return True
         
         

    def run(self):
        while self.video.isOpened():
            
            # Get frame of video feed
            frame = self.video.read()[1]

            # Turn to RGB and flip image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)

            # Draw all targets
            for target in self.targets:
                 target.draw(image)

            # Use PoseLandmarker to detect poses
            to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = self.detector.detect(to_detect)

            # Draw landmarks on poses
            self.draw_landmarks(image, results)
            self.check_target_match(image, results)

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