import os
import torch
import cv2
import numpy as np
import argparse
import options.options as option
import utils.util as util
from models import create_model

def denoise_video(video_path, output_path, model, opt):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare the VideoWriter for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f"Processing video: {video_path}")
    print(f"Total frames: {frame_count}, Resolution: {frame_width}x{frame_height}, FPS: {fps}")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Preprocess the frame
        img = frame.astype(np.float32) / 255.0  # Normalize to [0, 1]
        img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float().unsqueeze(0)  # HWC to CHW with batch dim

        # Denoise the frame using the model
        model.feed_test_data(img_tensor)
        model.test()
        denoised_tensor = model.fake_H.detach().float().cpu()
        denoised_img = util.tensor2img(denoised_tensor)  # Convert to uint8 format

        # Write the denoised frame to the output video
        denoised_frame = cv2.cvtColor(denoised_img, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
        out.write(denoised_frame)

        frame_idx += 1
        print(f"Processed frame {frame_idx}/{frame_count}", end="\r")

    # Release resources
    cap.release()
    out.release()
    print(f"Denoised video saved to: {output_path}")

def main():
    # Parse options and initialize the model
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to options YAML file.')
    parser.add_argument('-video', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('-output_video', type=str, required=True, help='Path to save the denoised video.')
    args = parser.parse_args()

    # Parse configuration and create the model
    opt = option.parse(args.opt, is_train=False)
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)

    # Denoise the video
    denoise_video(args.video, args.output_video, model, opt)

if __name__ == "__main__":
    main()
