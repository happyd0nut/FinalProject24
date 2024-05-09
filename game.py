import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import vision
import cv2
from pygame import mixer
import os
import ast

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
ORANGE = (255, 165, 0)
GREEN = (0,250,0)
BLUE = (0,0,250)
PURPLE = (165, 0, 255)
WHITE = (255,255,255)

class Game():

    def __init__(self, mode):

        """
        Initializes all four targets, PoseLandmarker detector, the mixer, and VideoCapture
        for the game Just Dance.
        Args:
            mode: input "manual" or "random", determines whether the targets are generated
            randomly or in accordance with specific poses from images
        """
        
        # Initializing instance variables
        self.mode = mode
        self.score = 0

        self.rh_target = Target(color=RED, quadrant=1, respwan_type=self.mode)
        self.lh_target = Target(color=BLUE, quadrant=2, respwan_type=self.mode)
        self.rf_target = Target(color=ORANGE, quadrant=4, respwan_type=self.mode)
        self.lf_target = Target(color=PURPLE, quadrant=3, respwan_type=self.mode)
        self.targets = [self.rh_target, self.lh_target, self.rf_target, self.lf_target]
        
        # Create PoseLandmarker detector
        base_options = BaseOptions(model_asset_path="data/model/pose_landmarker_full.task")
        options = PoseLandmarkerOptions(base_options = base_options, num_poses = 1,
                                        output_segmentation_masks = True)
        self.detector = PoseLandmarker.create_from_options(options)

        # Start video
        self.video = cv2.VideoCapture(1)

        # Initialize mixer
        mixer.init()
        mixer.music.load("data/music/song1.mp3")
        mixer.music.set_volume(0.7)
        mixer.music.play(-1)


    def draw_landmarks(self, image, detection_result):

        """
        Function adapted from MediaPipe's provided example code for the PoseLandmarker 
        library in Google CoLab.
        Args:
            image: the image frame in which to draw the landmarks on
            detection_result: the outputed data of the PoseLandmarker detector after
            processing the image frame
        """
        
        # PoseLandmarker results gets a list of poses detected
        pose_landmarks_list = detection_result.pose_landmarks
        pose_segmentation_list = detection_result.segmentation_masks

        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]
            pose_seg_mask = pose_segmentation_list[idx].numpy_view()
            pose_visual_mask = np.repeat(pose_seg_mask[:, :, np.newaxis], 3, axis=2) * 255

            # Draw the pose landmarks
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks])
            DrawingUtil.draw_landmarks(image,
                                        pose_landmarks_proto,
                                        solutions.pose.POSE_CONNECTIONS,
                                        solutions.drawing_styles.get_default_pose_landmarks_style())
            
    def image_pose_to_csv(self):

        """
        Processes all images in the "data/images" folder and detects PoseLandmarker 
        data from each image. Saves its normalized coordinates into a dictionary then
        DataFrame then CSV file into the "data/csv" folder.
        Args:
            None
        """
        
        save_filepath = "data/csv/image_pose_data.csv"

        # Create dictionary holding coordinates for each landmark point
        pose_images_coordinates = {"RHX" : [], 
                                   "RHY" : [],
                                   "LHX" : [],
                                   "LHY" : [],
                                   "RFX" : [], 
                                   "RFY" : [],
                                   "LFX" : [],
                                   "LFY" : [],
                                   "image_size" : []}

        directory_str = "data/images"
        directory = os.fsencode(directory_str)
        
        # Loop through each image in folder "images"
        for file in os.listdir(directory):
            
            image_filename = os.fsdecode(file)
            image = cv2.imread("data/images/" + image_filename)
            formatted_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = self.detector.detect(formatted_image)
            self.draw_landmarks(image, results)

            pose_landmarks_list = results.pose_landmarks
            pose_landmarks = pose_landmarks_list[0]
            
            right_hand = pose_landmarks[PoseLandmarkPoints.LEFT_INDEX.value]
            left_hand = pose_landmarks[PoseLandmarkPoints.RIGHT_INDEX.value]
            right_foot = pose_landmarks[PoseLandmarkPoints.LEFT_ANKLE.value]
            left_foot = pose_landmarks[PoseLandmarkPoints.RIGHT_ANKLE.value]

            # Save coordinates in dictionary
            pose_images_coordinates["RHX"].append(right_hand.x)
            pose_images_coordinates["RHY"].append(right_hand.y)
            pose_images_coordinates["LHX"].append(left_hand.x)
            pose_images_coordinates["LHY"].append(left_hand.y)
            pose_images_coordinates["RFX"].append(right_foot.x)
            pose_images_coordinates["RFY"].append(right_foot.y)
            pose_images_coordinates["LFX"].append(left_foot.x)
            pose_images_coordinates["LFY"].append(left_foot.y)
            pose_images_coordinates["image_size"].append(image.shape[:2])
            
        # Save dictionary to CSV file
        df = pd.DataFrame.from_dict(pose_images_coordinates)
        df.to_csv(save_filepath)
    

    def check_target_match(self, image, detection_result):
        
        # Get image info and list of pose landmarks detected from PoseLandmarker detector
        imageHeight, imageWidth = image.shape[:2]
        pose_landmarks_list = detection_result.pose_landmarks

        # Loop through each pose landmark
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]
            
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
            if pixelCoord_r_hand and pixelCoord_l_hand and pixelCoord_r_foot and pixelCoord_l_foot:
                cv2.circle(image, (pixelCoord_r_hand[0], pixelCoord_r_hand[1]), 40, RED, 5)
                cv2.circle(image, (pixelCoord_l_hand[0], pixelCoord_l_hand[1]), 40, BLUE, 5)
                cv2.circle(image, (pixelCoord_r_foot[0], pixelCoord_r_foot[1]), 40, ORANGE, 5)
                cv2.circle(image, (pixelCoord_l_foot[0], pixelCoord_l_foot[1]), 40, PURPLE, 5)
                
                r_hand_int = self.check_target_intercept(pixelCoord_r_hand[0], pixelCoord_r_hand[1], self.rh_target)
                l_hand_int = self.check_target_intercept(pixelCoord_l_hand[0], pixelCoord_l_hand[1], self.lh_target)
                r_foot_int = self.check_target_intercept(pixelCoord_r_foot[0], pixelCoord_r_foot[1], self.rf_target)
                l_foot_int = self.check_target_intercept(pixelCoord_l_foot[0], pixelCoord_l_foot[1], self.lf_target)
                
                # Respawn target if all targets are hit
                if r_hand_int and l_hand_int and r_foot_int and l_foot_int:
                    self.score += 1
                    
                    if self.mode == "manual":
                        df = pd.read_csv("data/csv/image_pose_data.csv")
                        df["image_size"] = df["image_size"].apply(ast.literal_eval)
                        cols_list = df.columns
                        row_idx = self.score - 1

                        # Rescale the dimensions the referenced pose image to the window height
                        img_dims = df.iloc[row_idx][-1]
                        scale = 700/int(img_dims[0])
                        scaled_img_dims = (img_dims[0]*scale, img_dims[1]*scale)
                        
                        # Respawn all targets at cooresponding coordinates
                        for col_idx in range(1, len(cols_list)-1, 2):
                            x_norm_coord = df.iloc[row_idx, col_idx]
                            y_norm_coord = df.iloc[row_idx, col_idx+1]
                            x_coord = DrawingUtil._normalized_to_pixel_coordinates(x_norm_coord, 0, scaled_img_dims[1], scaled_img_dims[0])[0]
                            y_coord = DrawingUtil._normalized_to_pixel_coordinates(0, y_norm_coord, scaled_img_dims[1], scaled_img_dims[0])[1]
                            
                            curr_target = self.targets[int((col_idx-1)/2)]
                            curr_target.respawn(x_coord,y_coord)
                    
                    if self.mode == "random":
                        for target in self.targets:
                            target.respawn_random()


    def check_target_intercept(self, point_x, point_y, target):

        target_x = target.x
        target_y = target.y

        # Respawn target if point is hit
        if (point_x < target_x + 30 and point_x > target_x - 30) and (point_y < target_y + 30 and point_y > target_y - 30):
            return True
         

    def run(self):

        # Testing functionality
        self.image_pose_to_csv()

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

            # Display score
            cv2.rectangle(image, (1080,50), (1250,120), color=WHITE, thickness=-1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image,"Score: " + str(self.score),(1090,105), font, fontScale=1, color=GREEN, thickness=2)
            
            # Display image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow("Pose Tracking", image)

            if cv2.waitKey(50) & 0xFF == ord('q'):
                break

        self.video.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    g = Game(mode="manual")
    g.run()