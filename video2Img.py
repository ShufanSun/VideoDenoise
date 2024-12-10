import cv2
import os

class VideoFrameExtractor:
    def __init__(self, video_path, output_folder):
        """
        Initialize the VideoFrameExtractor.

        Parameters:
        - video_path: str, path to the input video file.
        - output_folder: str, path to the folder where extracted frames will be saved.
        """
        self.video_path = video_path
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def extract_frames(self):
        """
        Extract frames from the video and save them as image files.
        """
        video = cv2.VideoCapture(self.video_path)
        frame_count = 0
        success, frame = video.read()

        while success:
            frame_filename = os.path.join(self.output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved {frame_filename}")
            frame_count += 1
            success, frame = video.read()

        video.release()
        print(f"Extracted {frame_count} frames to {self.output_folder}.")
        
# Example usage:
if __name__ == "__main__":
    # /mnt/c/Users/sofin/Documents/_Current Classes/ECE722/videoData
    video_path = "/mnt/c/Users/sofin/Documents/_Current Classes/ECE722/videoData/C0004.MP4"
    output_folder = "/mnt/c/Users/sofin/Documents/_Current Classes/ECE722/C0004"

    extractor = VideoFrameExtractor(video_path, output_folder)
    extractor.extract_frames()

