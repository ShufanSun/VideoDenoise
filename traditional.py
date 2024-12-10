import cv2
import numpy as np
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

# image = cv2.imread(output_path)
image_path= 'results/frame_0290.jpg'
# "C:\Users\sofin\Documents\_Current Classes\ECE722\frames7\frame_0290.jpg"
processor = ImageProcessor(image_path)
save_path = 'results/whiteBalancing/whitebalanced_image.jpg'  # Specify where to save the processed image
save_path2 = 'results/whiteBalancing/histogram.jpg'  # Path for saving the histogram plot
save_path3='results/demosaic/demosaiced.jpg'

# Step 1: White Balance
processor.process_and_display(percentile_value=99.9, save_path=save_path, save_path2=save_path2)

# Step 2: Demosaicing
processor = DemosaicProcessor(save_path)
# Load the raw image
processor.load_image()
# Perform bilinear demosaicing
r, g, b = processor.bilinear()

# Save and display the image
processor.save_image(np.dstack((r,g,b)), save_path3)
# processor.display_image(r, g, b)

# Step 3: Sharpening
processor = ImageSharpener(save_path)
# output_path = "results/sharpen/sharpened_output.jpg"  # Define the output path
edges, sharpened_img = processor.sharpen()

processor.img.save("results/sharpen/original.jpg")
edges.save("results/sharpen/edges.jpg")
sharpened_img.save("results/sharpen/sharpened.jpg")

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
sharpened_frame.save("results/sharpen/sharpened_frame.jpg")

# Save the combined comparison image
# comparison.show()
comparison.save("results/sharpen/comparison.jpg")

# Step 4: Denoising
processor = ImageDenoiser('results/sharpen/sharpened_frame.jpg')

# Load the image
processor.load_image()

# Apply denoising with a region size of 5
region_size = 4  # Adjust this value for stronger/weaker denoising
denoised_image = processor.denoise_rgb(region_size)

# Save the denoised image
output_path = "results/denoise/denoised_frame.jpg"
processor.save_image(denoised_image, output_path)


# Step 5: Gamma Correction
# Input image path
input_image_path = "results/denoise/denoised_frame.jpg"

# Initialize the class and process the image
processor = GammaCorrection(input_image_path)

try:
    # Load the image
    processor.load_image()

    # Apply gamma correction with gamma = 1.13
    gamma_value = 1.13
    processor.apply_gamma_correction(gamma_value)

    # Save the results
    gamma_image_path = 'results/gamma/gamma_image.png'
    gamma_corrected_image_path = 'results/gamma/gamma_correction.png'
    processor.save_images(gamma_image_path, gamma_corrected_image_path)

except Exception as e:
    print(f"Error: {e}")