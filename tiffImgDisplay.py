from PIL import Image
import matplotlib.pyplot as plt

# Open the TIFF image

# Load the two TIFF images
img1 = Image.open('00156_00_30s_gt.tif')  # Replace with your image path
img2 = Image.open('twice-denoised.tif')  # Replace with your image path
img3 = Image.open('00156_00_0.1s.tif')  # Replace with your image path

# Create a figure with three subplots, one for each image
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

# Display the first image
axes[0].imshow(img1)
axes[0].axis('off')  # Hide axes for better view
axes[0].set_title('First Image')  # Set title for the first image

# Display the second image
axes[1].imshow(img2)
axes[1].axis('off')  # Hide axes for better view
axes[1].set_title('Second Image')  # Set title for the second image

# Display the third image
axes[2].imshow(img3)
axes[2].axis('off')  # Hide axes for better view
axes[2].set_title('Third Image')  # Set title for the third image

# Adjust layout and show
plt.tight_layout()
plt.show()
