import os
from whiteBalance import ImageProcessor
from denoise import ImageDenoiser
from sharpen import ImageSharpener
from GammaCorrection import GammaCorrection
from PIL import Image


class TraditionalProcessor:
    def __init__(self, input_path, output_base_folder):
        """
        Initialize the processor for a single image, and prepare output folders for each stage.

        Parameters:
        - input_path: str, path to the input image file.
        - output_base_folder: str, path to the base output folder where processed images will be saved.
        """
        self.input_path = input_path
        self.output_base_folder = output_base_folder
        self.base_name = os.path.splitext(os.path.basename(self.input_path))[0]  # Extract unique name from input image
        
        # Create subfolders for each stage
        self.whitebalanced_folder = os.path.join(self.output_base_folder, 'whiteBalanced')
        self.sharpen_folder = os.path.join(self.output_base_folder, 'sharpen')
        self.denoise_folder = os.path.join(self.output_base_folder, 'denoise')
        self.gamma_folder = os.path.join(self.output_base_folder, 'gamma')
        
        # Ensure each folder exists
        os.makedirs(self.whitebalanced_folder, exist_ok=True)
        os.makedirs(self.sharpen_folder, exist_ok=True)
        os.makedirs(self.denoise_folder, exist_ok=True)
        os.makedirs(self.gamma_folder, exist_ok=True)

    def process_image(self):
        try:
            # Step 1: White Balance
            white_balance_path = os.path.join(self.whitebalanced_folder, f"{self.base_name}_whitebalanced.jpg")
            histogram_path = os.path.join(self.whitebalanced_folder, f"{self.base_name}_histogram.jpg")
            processor = ImageProcessor(self.input_path)
            processor.process_and_display(percentile_value=99.9, save_path=white_balance_path, save_path2=histogram_path)

            # Step 2: Sharpening (Use the white balanced image for this step)
            sharpen_path = os.path.join(self.sharpen_folder, f"{self.base_name}_sharpened.jpg")
            processor = ImageSharpener(white_balance_path)
            edges, sharpened_img = processor.sharpen()
            # sharpened_img.save(sharpen_path)
            
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
            sharpened_frame.save(sharpen_path)
            

            # Step 3: Denoising (Use the sharpened image for this step)
            denoise_path = os.path.join(self.denoise_folder, f"{self.base_name}_denoised.jpg")
            processor = ImageDenoiser(sharpen_path)
            processor.load_image()
            denoised_image = processor.denoise_rgb(region_size=4)
            processor.save_image(denoised_image, denoise_path)

            # Step 4: Gamma Correction (Use the denoised image for this step)
            final_output_path = os.path.join(self.gamma_folder, f"{self.base_name}_final.jpg")
            final_output_path2 = os.path.join(self.gamma_folder, f"{self.base_name}_final2.jpg")
            
            processor = GammaCorrection(denoise_path)
            processor.load_image()
            processor.apply_gamma_correction(gamma_value=1.13)
            processor.save_images(final_output_path, final_output_path2)

            print(f"Processing completed for {self.input_path}. Final image saved to {final_output_path}.")

        except Exception as e:
            print(f"Error processing {self.input_path}: {e}")
