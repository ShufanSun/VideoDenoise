import rawpy
import numpy as np
from PIL import Image
import re
import os

def parse_filename(filepath):
    """
    Parses the filename to extract pair number, index, and exposure time.
    Handles exposure times as integers or floating-point numbers.
    """
    filename = os.path.basename(filepath)
    match = re.match(r"(\d+)_(\d+)_(\d+\.?\d*)s", filename)
    if not match:
        raise ValueError(f"Filename {filename} does not match expected pattern.")
    pair, index, exposure_time = match.groups()
    return int(pair), int(index), float(exposure_time)

def adjust_exposure_with_rawpy(base_file, target_file, output_file):
    """
    Adjusts the exposure of one RAW image to match another using rawpy's exposure adjustment.
    """
    # Parse filenames to extract exposure times
    _, _, base_exposure = parse_filename(base_file)
    _, _, target_exposure = parse_filename(target_file)

    # Calculate exposure adjustment factor
    adjustment_factor = target_exposure / base_exposure

    # Read RAW image and adjust exposure
    with rawpy.imread(base_file) as raw:
        adjusted_image = raw.postprocess(
            bright=adjustment_factor/60,  # Adjust brightness based on the exposure factor
            output_bps=8  # Output as 8-bit image
        )

    # Convert to a PIL image
    adjusted_image_pil = Image.fromarray(adjusted_image)

    # Save adjusted image
    adjusted_image_pil.save(output_file)

    print(f"Exposure adjusted image saved to {output_file}")

# File paths
base_file_path = "./Sony/Sony/short/00145_00_0.1s.ARW"  # File to adjust
target_file_path = "./Sony/Sony/long/00145_00_30s.ARW"  # File to match
output_file_path = "adjusted_image.tiff"  # Output file path

# Adjust exposure
adjust_exposure_with_rawpy(base_file_path, target_file_path, output_file_path)
