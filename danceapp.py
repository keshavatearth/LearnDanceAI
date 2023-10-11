import numpy as np
import cv2
import mediapipe as mp
from matplotlib import pyplot as plt
import time

# Load the MediaPipe Pose model
mp_pose = mp.solutions.pose

# Create a video capture object
cap = cv2.VideoCapture("/content/video.mp4")

# Initialize the stick figure generator
stick_figure_generator = mp_pose.Pose(static_image_mode=False)

# Define the connections between landmarks to create a stick figure
connections = [(11, 13), (13, 15), (12, 14), (14, 16),
               (11, 23), (12, 24), (23, 25), (24, 26),
               (25, 27), (26, 28), (23, 24), (11, 0),
               (12, 0),  (11, 12), (27, 31), (28, 32)]

# Loop over the frames of the video
start_time = time.time()  
frame_count = 0
while True:

    # Capture the next frame
    ret, frame = cap.read()

    # Check if the frame is empty
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run the MediaPipe Pose model on the frame
    results = stick_figure_generator.process(frame_rgb)

    # Check if any pose landmarks were detected
    if results.pose_landmarks:
        # Get the stick figure landmarks
        stick_figure_landmarks = results.pose_landmarks.landmark
        
        # Draw the stick figure on the frame
        h, w, _ = frame.shape
        for connection in connections:
            start = stick_figure_landmarks[connection[0]]
            end = stick_figure_landmarks[connection[1]]
            color = (0, 0, 255)  # Red for all other connections
            thickness = 4
            if connection in [(11, 13), (13, 15), (12, 14), (14, 16)]:  # Check if the connection is a part of the arm
                color = (255, 0, 0)  # Blue for arm connections
                thickness = 8
            cv2.line(frame, (int(start.x * w), int(start.y * h)), 
                (int(end.x * w), int(end.y * h)), color, thickness)
        
        # Draw big green dots at joint points
        joint_landmarks = set([item for sublist in connections for item in sublist])  # Get unique landmarks from connections
        for landmark_index in joint_landmarks:
            landmark = stick_figure_landmarks[landmark_index]
            cv2.circle(frame, (int(landmark.x * w), int(landmark.y * h)), 5, (0, 255, 0), -1)  # -1 fills the circle



        # Display the frame using Matplotlib
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.show()

    # Increment the frame counter
    frame_count += 1

end_time = time.time()  # Record the end time
elapsed_time = end_time - start_time  # Calculate the elapsed time
processing_rate = frame_count / elapsed_time  # Calculate the processing rate in frames per second

print(f'Processed {frame_count} frames in {elapsed_time:.2f} seconds ({processing_rate:.2f} fps)')

# Release the video capture object
cap.release()
