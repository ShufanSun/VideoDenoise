import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d

class DemosaicProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.raw_image = None

    def load_image(self):
        """Load a grayscale (raw Bayer) image from the given path."""
        self.raw_image = np.array(Image.open(self.image_path).convert('L'))

    # Color filtering: `rggb`
    def bayer(self, im):
        r = np.zeros(im.shape[:2])
        g = np.zeros(im.shape[:2])
        b = np.zeros(im.shape[:2])
        r[0::2, 0::2] += im[0::2, 0::2]
        g[0::2, 1::2] += im[0::2, 1::2]
        g[1::2, 0::2] += im[1::2, 0::2]
        b[1::2, 1::2] += im[1::2, 1::2]
        return r, g, b

    # Demosaicing
    def bilinear(self):
        r, g, b = self.bayer(self.raw_image)  # Use self.bayer here

        # Green interpolation
        k_g = 1/4 * np.array([[0,1,0],[1,0,1],[0,1,0]])
        convg = convolve2d(g, k_g, 'same')
        g = g + convg

        # Red interpolation
        k_r_1 = 1/4 * np.array([[1,0,1],[0,0,0],[1,0,1]])
        convr1 = convolve2d(r, k_r_1, 'same')
        convr2 = convolve2d(r + convr1, k_g, 'same')
        r = r + convr1 + convr2

        # Blue interpolation
        k_b_1 = 1/4 * np.array([[1,0,1],[0,0,0],[1,0,1]])
        convb1 = convolve2d(b, k_b_1, 'same')
        convb2 = convolve2d(b + convb1, k_g, 'same')
        b = b + convb1 + convb2

        # Normalize the values to the range 0-255 (for 8-bit images)
        r = np.clip(r * 255.0 / np.max(r), 0, 255)
        g = np.clip(g * 255.0 / np.max(g), 0, 255)
        b = np.clip(b * 255.0 / np.max(b), 0, 255)

        return r.astype(np.uint8), g.astype(np.uint8), b.astype(np.uint8)

    def save_image(self, demosaiced_image, output_path):
        """Save the demosaiced image to the specified path."""
        # Normalize to [0, 255] and convert to uint8
        demosaiced_image = np.clip(demosaiced_image, 0, 255)  # Clip values to be within valid range
        demosaiced_image = demosaiced_image.astype(np.uint8)  # Convert to uint8
        Image.fromarray(demosaiced_image).save(output_path)

    def display_image(self, r, g, b):
        """Display the demosaiced image using matplotlib."""
        # Stack the r, g, b channels into a 3D array
        demosaiced_image = np.dstack((r, g, b)).astype(np.uint8)
        # print(demosaiced_image)
        # Display the image
        plt.imshow(demosaiced_image.astype(np.uint8))  # Ensure uint8 for correct display
        plt.axis('off')  # Turn off axis
        plt.show()
  
# Example usage
if __name__ == "__main__":
    # Initialize the processor with the path to the raw Bayer image
    processor = DemosaicProcessor("whitebalanced_image.jpg")
    
    # Load the raw image
    processor.load_image()

    # Perform bilinear demosaicing
    r, g, b = processor.bilinear()

    # Save and display the image
    processor.save_image(np.dstack((r,g,b)), "demosaiced_image.jpg")
    processor.display_image(r, g, b)
