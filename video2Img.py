import cv2
import os

# Specify the input video file
video_path = "C:\\Users\\sofin\\Documents\\_Current Classes\\ECE722\\videoData\\video_20210208_212419.mp4"
# "C:\Users\sofin\Documents\_Current Classes\ECE722\videoData\video_20210208_212419.mp4"

# Specify the output folder for frames
output_folder = "frames"
os.makedirs(output_folder, exist_ok=True)

# Load the video
video = cv2.VideoCapture(video_path)

frame_count = 0
success, frame = video.read()

# Loop through the video and save each frame
while success:
    frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_filename, frame)
    print(f"Saved {frame_filename}")
    frame_count += 1
    success, frame = video.read()

video.release()
print(f"Extracted {frame_count} frames to {output_folder}.")
