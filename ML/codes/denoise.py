import os
import torch
import cv2
import numpy as np
import argparse
import options.options as option
import utils.util as util
from models import create_model
from data.util import read_img

def denoise_image(noisy_image_path, output_path, model, opt):
    # Load and preprocess the noisy image
    img = read_img(env=None, path=noisy_image_path)  # env=None for direct file loading
    img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float().unsqueeze(0)  # To tensor and add batch dim

    # Feed the noisy image to the model
    model.feed_test_data(img_tensor)
    model.test()

    # Get denoised image from the model
    denoised_tensor = model.fake_H.detach().float().cpu()
    denoised_img = util.tensor2img(denoised_tensor)  # Convert to uint8 format

    # Save the denoised image
    cv2.imwrite(output_path, cv2.cvtColor(denoised_img, cv2.COLOR_RGB2BGR))
    print(f"Denoised image saved to: {output_path}")

def main():
    # Parse options and initialize the model
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to options YAML file.')
    parser.add_argument('-noisy_image', type=str, required=True, help='Path to the noisy input image.')
    parser.add_argument('-output_image', type=str, required=True, help='Path to save the denoised image.')
    args = parser.parse_args()

    # Parse configuration and create the model
    opt = option.parse(args.opt, is_train=False)
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)

    # Denoise the input image
    denoise_image(args.noisy_image, args.output_image, model, opt)

if __name__ == "__main__":
    main()
