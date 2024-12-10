import cv2
import numpy as np
import os

class GammaCorrection:
    def __init__(self, input_path):
        """
        Initializes the GammaCorrection class with the input image path.
        """
        self.input_path = input_path
        self.image = None
        self.gamma_corrected_image = None
        self.gamma_image = None

    def load_image(self):
        """
        Loads the input image in RGB mode.
        """
        if os.path.exists(self.input_path):
            self.image = cv2.imread(self.input_path)  # Load as BGR by default
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)  # Convert to RGB
            print(f"Image loaded from: {self.input_path}")
        else:
            raise FileNotFoundError(f"File not found: {self.input_path}")

    def apply_gamma_correction(self, gamma):
        """
        Applies gamma correction and inverse gamma correction to the image.

        Parameters:
        gamma (float): The gamma value for correction.
        """
        if self.image is None:
            raise ValueError("Image not loaded. Please load the image first.")

        # Pre-compute gamma lookup tables for efficiency
        gamma_table = np.array([255 * ((i / 255) ** gamma) for i in range(256)], dtype=np.uint8)
        inv_gamma_table = np.array([255 * ((i / 255) ** (1 / gamma)) for i in range(256)], dtype=np.uint8)

        # Apply lookup tables to each channel
        self.gamma_image = cv2.LUT(self.image, gamma_table)
        self.gamma_corrected_image = cv2.LUT(self.image, inv_gamma_table)

        print("Gamma correction applied.")

    def save_images(self, gamma_image_path, gamma_corrected_path):
        """
        Saves the gamma-corrected images to specified file paths.

        Parameters:
        gamma_image_path (str): File path to save the gamma-corrected image.
        gamma_corrected_path (str): File path to save the inverse gamma-corrected image.
        """
        if self.gamma_image is not None and self.gamma_corrected_image is not None:
            # Convert RGB to BGR for saving with OpenCV
            cv2.imwrite(gamma_image_path, cv2.cvtColor(self.gamma_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(gamma_corrected_path, cv2.cvtColor(self.gamma_corrected_image, cv2.COLOR_RGB2BGR))
            print(f"Images saved: {gamma_image_path}, {gamma_corrected_path}")
        else:
            raise ValueError("Gamma-corrected images not available. Apply gamma correction first.")

# Example Usage
if __name__ == "__main__":
    # Input image path
    input_image_path = "results/denoise/denoised_frame.jpg"

    # Initialize the class and process the image
    processor = GammaCorrection(input_image_path)

    try:
        # Load the image
        processor.load_image()

        # Apply gamma correction with gamma = 1.13
        gamma_value = 1/1.13
        processor.apply_gamma_correction(gamma_value)

        # Save the results
        gamma_image_path = 'results/gamma/gamma_image.png'
        gamma_corrected_image_path = 'results/gamma/gamma_correction.png'
        processor.save_images(gamma_image_path, gamma_corrected_image_path)

    except Exception as e:
        print(f"Error: {e}")

