import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import img_as_ubyte

class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = self._load_image()

    def _load_image(self):
        """Load an image from a TIFF file and convert it to a NumPy array."""
        # Open the TIFF file using PIL
        image = Image.open(self.image_path)
        
        # Convert the image to a NumPy array
        image = np.array(image)

        # Convert to BGR for OpenCV compatibility
        if image.shape[-1] == 4:  # Handle images with an alpha channel
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image

    def percentile_whitebalance(self, image, percentile_value=95):
        """Perform white balancing using the specified percentile value."""
        # Normalize using the calculated percentile values
        whitebalanced = img_as_ubyte(
            (image * 1.0 / np.percentile(image, percentile_value, axis=(0, 1)))
            .clip(0, 1)
        )

        # Prepare data for histogram
        histograms = []
        for channel in range(3):
            channel_values = image[:, :, channel]
            value = np.percentile(channel_values, percentile_value)
            histograms.append((np.bincount(channel_values.flatten(), minlength=256) * 1.0 / channel_values.size, value))

        return whitebalanced, histograms

    def process_and_display(self, percentile_value=95, save_path=None, save_path2=None):
        """Apply white balancing, display the images, and optionally save the processed image."""
        # Process the image
        whitebalanced, histograms = self.percentile_whitebalance(self.image, percentile_value)

        # Save the processed image if a save path is provided
        if save_path:
            cv2.imwrite(save_path, whitebalanced)
            print(f"Processed image saved to: {save_path}")

        # Create the plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Display the original image
        axes[0, 0].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # Display the processed image
        axes[0, 1].imshow(cv2.cvtColor(whitebalanced, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Whitebalanced Image')
        axes[0, 1].axis('off')

        # Display the histogram
        colors = ['r', 'g', 'b']
        for channel, color in enumerate(colors):
            values, percentile = histograms[channel]
            axes[1, 0].step(np.arange(256), values, c=color)
            axes[1, 0].axvline(percentile, ls='--', c=color, label=f'{color.upper()} max = {percentile:.2f}')
        axes[1, 0].set_xlim(0, 255)
        axes[1, 0].set_xlabel('Pixel Value')
        axes[1, 0].set_ylabel('Fraction of Pixels')
        axes[1, 0].set_title('Histogram of RGB Channels')
        axes[1, 0].legend()

        # Leave the bottom-right empty
        axes[1, 1].axis('off')

        plt.tight_layout()

        # Save the matplotlib figure if a save path is provided
        if save_path2:
            fig.savefig(save_path2)
            print(f"Figure saved to: {save_path2}")

        # plt.show()


# Example usage
if __name__ == "__main__":
    image_path = '00156_00_0.1s.tif'  # Your TIFF file path
    save_path = 'whitebalanced_image.tif'  # Specify where to save the processed image
    save_path2 = 'histogram.png'
    processor = ImageProcessor(image_path)
    processor.process_and_display(percentile_value=98, save_path=save_path, save_path2=save_path2)
