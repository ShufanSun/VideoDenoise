import cv2
import numpy as np
import os
import pyheif
from PIL import Image
from whiteBalance import ImageProcessor
from demosaic import DemosaicProcessor
from denoise import ImageDenoiser
from sharpen import ImageSharpener
from GammaCorrection import GammaCorrection
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.io import imread, imshow
from skimage import img_as_ubyte

# Set the folder paths
input_folder = '/mnt/c/Users/sofin/Documents/_Current Classes/ECE722/C0004/original'  # Replace with your input folder path
output_folder = '/mnt/c/Users/sofin/Documents/_Current Classes/ECE722/C0004/final'  # Replace with your output folder path

# Get all image files from the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.heif'))]

for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    print(f"Processing {image_path}")
    
    # Initialize the processor with the image
    processor = ImageProcessor(image_path)
    
    # Define the output paths
    save_path = os.path.join(output_folder, f"whiteBalanced/{image_file}")
    save_path2 = os.path.join(output_folder, f"whiteBalanced/histogram_{image_file}")
    # save_path3 = os.path.join(output_folder, f"demosaic/demosaiced_{image_file}")

    # Step 1: White Balance
    processor.process_and_display(percentile_value=99.9, save_path=save_path, save_path2=save_path2)
    
    # Step 2: Demosaicing
    # processor = DemosaicProcessor(save_path)
    # processor.load_image()
    # r, g, b = processor.bilinear()
    # processor.save_image(np.dstack((r, g, b)), save_path3)

    # Step 3: Sharpening
    processor = ImageSharpener(save_path)
    edges, sharpened_img = processor.sharpen()

    processor.img.save(os.path.join(output_folder, f"sharpen/original_{image_file}"))
    edges.save(os.path.join(output_folder, f"sharpen/edges_{image_file}"))
    sharpened_img.save(os.path.join(output_folder, f"sharpen/sharpened_{image_file}"))

    # Create a combined image for comparison
    comparison = Image.new("RGB", (processor.img.width * 3, processor.img.height))
    comparison.paste(processor.img, (0, 0))
    comparison.paste(edges, (processor.img.width, 0))
    comparison.paste(sharpened_img, (processor.img.width * 2, 0))
    
    # Save and show the sharpened (third frame) from comparison
    sharpened_frame = comparison.crop((
        processor.img.width * 2,  # Left
        0,                       # Top
        processor.img.width * 3,  # Right
        processor.img.height      # Bottom
    ))
    sharpened_frame.save(os.path.join(output_folder, f"sharpen/sharpened_frame_{image_file}"))

    # Save the combined comparison image
    comparison.save(os.path.join(output_folder, f"sharpen/comparison_{image_file}"))

    # Step 4: Denoising
    processor = ImageDenoiser(os.path.join(output_folder, f"sharpen/sharpened_frame_{image_file}"))
    processor.load_image()
    region_size = 4  # Adjust this value for stronger/weaker denoising
    denoised_image = processor.denoise_rgb(region_size)
    output_path = os.path.join(output_folder, f"denoise/denoised_{image_file}")
    processor.save_image(denoised_image, output_path)

    # Step 5: Gamma Correction
    input_image_path = os.path.join(output_folder, f"denoise/denoised_{image_file}")
    processor = GammaCorrection(input_image_path)

    try:
        processor.load_image()
        gamma_value = 1.13
        processor.apply_gamma_correction(gamma_value)

        gamma_image_path = os.path.join(output_folder, f"gamma/gamma_image_{image_file}")
        gamma_corrected_image_path = os.path.join(output_folder, f"gamma/gamma_correction_{image_file}")
        processor.save_images(gamma_image_path, gamma_corrected_image_path)

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
