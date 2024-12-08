from PIL import Image
from filters import *

def median(data):
    """Applies a median filter on the given data."""
    data = sorted(data)
    index = len(data) // 2  # Integer division to get the middle index
    return data[index]

def denoise_rgb(img):
    """Applies the median filter to each RGB channel of the image."""
    # Split the image into its R, G, B channels
    r, g, b = img.split()

    # Apply the filter (median) to each channel
    r = apply_median_filter(r)
    g = apply_median_filter(g)
    b = apply_median_filter(b)

    # Merge the channels back into an RGB image
    return Image.merge("RGB", (r, g, b))

def apply_median_filter(channel_img):
    """Applies a median filter to a single image channel."""
    width, height = channel_img.size
    imgdup = channel_img.copy()
    pixels = imgdup.load()

    for x in range(width):
        for y in range(height):
            # Extract the 3x3 region and apply the median filter
            region = region3x3(channel_img, x, y)
            pixels[x, y] = median(region)

    return imgdup

def region3x3(img, x, y):
    """Get the 3x3 region of pixels surrounding (x, y)."""
    me = getpixel(img, x, y)
    N = getpixel(img, x, y - 1)
    S = getpixel(img, x, y + 1)
    E = getpixel(img, x + 1, y)
    W = getpixel(img, x - 1, y)
    NW = getpixel(img, x - 1, y - 1)
    NE = getpixel(img, x + 1, y - 1)
    SE = getpixel(img, x + 1, y + 1)
    SW = getpixel(img, x - 1, y + 1)

    return [me, N, E, S, W, NW, NE, SE, SW]

def getpixel(img, x, y):
    """Safely gets a pixel value, handling out-of-bounds coordinates."""
    width, height = img.size
    pixels = img.load()

    if x < 0: x = 0
    if x >= width: x = width - 1
    if y < 0: y = 0
    if y >= height: y = height - 1

    return pixels[x, y]

# Main script
input_path = 'whitebalanced_image.png'  # Change this to your image path

# Open and process the image
img = Image.open(input_path)
img = img.convert("RGB")  # Ensure the image is in RGB mode
img.show()

# Apply denoising to the RGB channels
denoised_image = denoise_rgb(img)
denoised_image.show()

# Save the denoised image
output_path = f"twice-denoised.png"
denoised_image.save(output_path)
print(f"Saved denoised image to {output_path}")
