import os
import re
import rawpy
import numpy as np
from PIL import Image
import math

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

def process_and_save_image(raw_file, output_folder, suffix):
    """
    Processes a RAW file using rawpy and saves it as an 8-bit JPEG in the output folder.
    """
    base_name = os.path.splitext(os.path.basename(raw_file))[0]
    output_path = os.path.join(output_folder, f"{base_name}_{suffix}.tif")

    with rawpy.imread(raw_file) as raw:
        processed_image = raw.postprocess(output_bps=8)
    processed_image_pil = Image.fromarray(processed_image)
    processed_image_pil.save(output_path)
    print(f"Saved processed image to {output_path}")

    return output_path

def adjust_exposure_with_pixel_ev(base_file, target_file, output_folder):
    """
    Adjusts the exposure of one RAW image to match another using pixel brightness-based EV calculations.
    """
    base_brightness = calculate_brightness(base_file)
    target_brightness = calculate_brightness(target_file)

    ev_diff = math.log2(target_brightness / base_brightness)
    adjustment_factor = 2 ** ev_diff

    base_name = os.path.splitext(os.path.basename(base_file))[0]
    adjusted_output_path = os.path.join(output_folder, f"{base_name}.tif")
    with rawpy.imread(base_file) as raw:
        adjusted_image = raw.postprocess(
            bright=adjustment_factor,
            output_bps=8
        )
    adjusted_image_pil = Image.fromarray(adjusted_image)
    adjusted_image_pil.save(adjusted_output_path)
    print(f"Saved exposure-adjusted image to {adjusted_output_path}")

    return adjusted_output_path

def automate_exposure_compensation(under_folder, gt_folder, output_folder):
    """
    Automates exposure compensation for all files in the underexposed folder
    relative to the ground truth folder.
    """
    short_output = os.path.join(output_folder, "short")
    long_output = os.path.join(output_folder, "long")
    os.makedirs(short_output, exist_ok=True)
    os.makedirs(long_output, exist_ok=True)

    # Collect all ground truth images by sequence number
    gt_files = {parse_filename(f)[0]: os.path.join(gt_folder, f)
                for f in os.listdir(gt_folder) if f.endswith('.ARW')}

    # Process all underexposed images
    for under_file in os.listdir(under_folder):
        if not under_file.endswith('.ARW'):
            continue
        under_path = os.path.join(under_folder, under_file)
        sequence, _ = parse_filename(under_file)

        if sequence not in gt_files:
            print(f"No matching ground truth for {under_file}")
            continue

        gt_path = gt_files[sequence]

        # Adjust exposure and save processed image
        adjust_exposure_with_pixel_ev(under_path, gt_path, short_output)

        # Save ground truth image to long/ (if not already saved)
        gt_output_path = os.path.join(long_output, os.path.basename(gt_path))
        if not os.path.exists(gt_output_path):
            process_and_save_image(gt_path, long_output, "gt")

    print(f"Processing complete. Results saved to {output_folder}")


# File paths
under_folder = "./Sony/Sony/short/"  # File to adjust
gt_folder = "./Sony/Sony/long/"  # File to match
output_folder = "./output/"  # Replace with actual output folder

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)
automate_exposure_compensation(under_folder, gt_folder, output_folder)
