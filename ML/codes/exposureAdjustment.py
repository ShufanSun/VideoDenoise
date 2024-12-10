import cv2
import numpy as np

# Input and output video paths
input_video = "./out.mp4"
output_video = "./adjusted_128.mp4"
target_average = 128

# Open the input video
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("Error: Unable to open input video.")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video

# Initialize the video writer
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:  # Break the loop if no frames are left
        break

    # Convert frame to grayscale (optional if you want per-channel adjustment)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Compute current average intensity
    current_average = np.mean(gray)

    # Compute scaling factor
    scaling_factor = target_average / current_average if current_average != 0 else 1

    # Scale the frame and clip pixel values
    adjusted_frame = np.clip(frame * scaling_factor, 0, 255).astype(np.uint8)

    # Write adjusted frame to the output video
    out.write(adjusted_frame)

# Release resources
cap.release()
out.release()
print(f"Processed video saved as {output_video}")
