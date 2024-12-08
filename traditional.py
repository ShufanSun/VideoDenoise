import cv2
import numpy as np
import pyheif
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.io import imread, imshow
from skimage import img_as_ubyte

def white_balance(image):
    """Perform simple white balancing."""
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(result)
    l = cv2.equalizeHist(l)
    result = cv2.merge((l, a, b))
    dinner_max = (result*1.0 / result.max(axis=(0,1)))
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

def convert_heic_to_png(input_path, output_path):
    # Open the HEIC file
    heif_file = pyheif.read(input_path)
    
    # Convert HEIC to a Pillow Image
    image = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )
    
    # Save the image as PNG
    image.save(output_path, format="PNG")
    print(f"Image converted and saved to {output_path}")
    return output_path

def denoise_and_sharpen(image):
    """Denoise and apply sharpening."""
    # Denoise using Non-Local Means
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    # Sharpen using kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    return sharpened

def gamma_correction(image, gamma=2.2):
    """Apply gamma correction."""
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

# Load the image
image_path = 'IMG_0456.HEIC'
output_path=convert_heic_to_png(image_path, "example.png")

image = cv2.imread(output_path)

# Step 1: White Balance
balanced_image = white_balance(image)

# Step 2: Denoising and Sharpening
enhanced_image = denoise_and_sharpen(balanced_image)

# Step 3: Gamma Correction
gamma_corrected_image = gamma_correction(enhanced_image)

# Save and display the result
cv2.imshow('Processed Image', gamma_corrected_image)
cv2.imwrite('processed_image.png', gamma_corrected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
