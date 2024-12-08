import os
import re
import rawpy
import numpy as np
from PIL import Image
import math
from sklearn.model_selection import train_test_split


def parse_filename(filepath):
    """
    Parses the filename to extract the sequence number and exposure time.
    """
    filename = os.path.basename(filepath)
    match = re.match(r"(\d+)_\d+_(\d+\.?\d*)s", filename)
    if not match:
        raise ValueError(f"Filename {filename} does not match expected pattern.")
    sequence, exposure_time = match.groups()
    return int(sequence), float(exposure_time)


def calculate_brightness(raw_file):
    """
    Calculates the mean brightness of an image processed with rawpy.
    """
    with rawpy.imread(raw_file) as raw:
        image = raw.postprocess(output_bps=8)
        brightness = np.mean(image)
    return brightness


def process_and_save_image(raw_file, output_folder, subfolder, filename):
    """
    Processes a RAW file using rawpy and saves it in the required directory structure.
    """
    output_path = os.path.join(output_folder, subfolder, filename)

    with rawpy.imread(raw_file) as raw:
        processed_image = raw.postprocess(output_bps=8)
    processed_image_pil = Image.fromarray(processed_image)
    processed_image_pil.save(output_path, format="PNG")
    print(f"Saved processed image to {output_path}")

    return output_path


def generate_downsampled_image(gt_image_path, output_folder, filename):
    """
    Generates a low-quality downsampled version of the ground truth image.
    """
    output_path = os.path.join(output_folder, "LQ", filename)

    image = Image.open(gt_image_path)
    downsampled_image = image.resize((image.width // 2, image.height // 2))
    downsampled_image.save(output_path, format="PNG")
    print(f"Saved downsampled image to {output_path}")

    return output_path


def adjust_exposure_with_pixel_ev(base_file, target_file, output_folder, filename):
    """
    Adjusts the exposure of one RAW image to match another using pixel brightness-based EV calculations.
    """
    base_brightness = calculate_brightness(base_file)
    target_brightness = calculate_brightness(target_file)

    ev_diff = math.log2(target_brightness / base_brightness)
    adjustment_factor = 2 ** ev_diff

    output_path = os.path.join(output_folder, "Noisy", filename)
    with rawpy.imread(base_file) as raw:
        adjusted_image = raw.postprocess(
            bright=adjustment_factor,
            output_bps=8
        )
    adjusted_image_pil = Image.fromarray(adjusted_image)
    adjusted_image_pil.save(output_path, format="PNG")
    print(f"Saved exposure-adjusted image to {output_path}")

    return output_path


def automate_exposure_compensation(under_folder, gt_folder, output_folder):
    """
    Automates the entire pipeline of splitting, exposure compensation, and organizing the dataset.
    """
    # Read GT images and group by sequence number
    gt_files = {parse_filename(f)[0]: os.path.join(gt_folder, f)
                for f in os.listdir(gt_folder) if f.endswith('.ARW')}
    sequence_numbers = list(gt_files.keys())

    # Train/Validation/Test split using sklearn
    train_val_sequences, test_sequences = train_test_split(sequence_numbers, test_size=0.15, random_state=42)
    train_sequences, val_sequences = train_test_split(train_val_sequences, test_size=0.15 / 0.85, random_state=42)

    splits = {
        "train": train_sequences,
        "val": val_sequences,
        "test": test_sequences
    }

    control_number = 1  # Start naming files from 0001.png
    for split, sequences in splits.items():
        split_folder = os.path.join(output_folder, split)
        gt_output = os.path.join(split_folder, "GT")
        noisy_output = os.path.join(split_folder, "Noisy")
        lq_output = os.path.join(split_folder, "LQ")
        os.makedirs(gt_output, exist_ok=True)
        os.makedirs(noisy_output, exist_ok=True)
        os.makedirs(lq_output, exist_ok=True)

        for sequence in sequences:
            gt_path = gt_files[sequence]

            # Find all underexposed images for this sequence
            under_files = [f for f in os.listdir(under_folder)
                           if parse_filename(f)[0] == sequence and f.endswith('.ARW')]

            for under_file in under_files:
                under_path = os.path.join(under_folder, under_file)

                # Generate control number and filenames
                control_name = f"{control_number:04d}.png"

                # Process GT and Noisy images
                gt_image_path = process_and_save_image(gt_path, split_folder, "GT", control_name)
                adjust_exposure_with_pixel_ev(under_path, gt_path, split_folder, control_name)

                # Generate LQ image from GT
                generate_downsampled_image(gt_image_path, split_folder, control_name)

                # Increment control number
                control_number += 1

    print(f"Processing complete. Results saved to {output_folder}")


under_folder = "./ML/Sony/Sony/short/"  # File to adjust
gt_folder = "./ML/Sony/Sony/long/"  # File to match
output_folder = "./ML/output_ml/"  # Replace with actual output folder

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)
automate_exposure_compensation(under_folder, gt_folder, output_folder)
