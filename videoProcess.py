from traditionalEncap import TraditionalProcessor
import os

class BatchImageProcessor:
    def __init__(self, input_folder, output_folder):
        """
        Initialize the batch processor for images in a folder.

        Parameters:
        - input_folder: str, path to the folder containing input images.
        - output_folder: str, path to the base folder where processed images will be saved (for each stage).
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def process_all_images(self):
        """
        Process all images in the input folder, stage by stage.
        """
        image_files = [f for f in os.listdir(self.input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        for image_file in image_files:
            input_path = os.path.join(self.input_folder, image_file)
            print(f"Processing {input_path}...")
            
            # Process image through all stages (whiteBalance, sharpen, denoise, gamma)
            processor = TraditionalProcessor(input_path, self.output_folder)
            processor.process_image()

if __name__ == "__main__":
    input_folder = "/mnt/c/Users/sofin/Documents/_Current Classes/ECE722/C0004/original"
    output_folder = "/mnt/c/Users/sofin/Documents/_Current Classes/ECE722/C0004/final"

    batch_processor = BatchImageProcessor(input_folder, output_folder)
    batch_processor.process_all_images()
