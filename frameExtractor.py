import os
import shutil

# Set the source folder path where the images are stored
source_folder = '/mnt/c/Users/sofin/Documents/_Current Classes/ECE722/C0004/final/gamma'

# Set the destination folder paths
folder1 = '/mnt/c/Users/sofin/Documents/_Current Classes/ECE722/C0004/final/gamma/gamma_correction_frame'
folder2 = '/mnt/c/Users/sofin/Documents/_Current Classes/ECE722/C0004/final/gamma/gamma_image_frame'

# Create destination folders if they don't exist
os.makedirs(folder1, exist_ok=True)
os.makedirs(folder2, exist_ok=True)

# Loop through all files in the source folder
for filename in os.listdir(source_folder):
    # Get the full path of the file
    file_path = os.path.join(source_folder, filename)
    
    # Check if the file is an image (it ends with .jpg)
    if filename.endswith('.jpg'):
        if filename.startswith('gamma_correction_frame_'):
            # Move the image to folder1
            shutil.move(file_path, os.path.join(folder1, filename))
        elif filename.startswith('gamma_image_frame_'):
            # Move the image to folder2
            shutil.move(file_path, os.path.join(folder2, filename))

print("Images have been sorted into respective folders.")
