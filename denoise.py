from PIL import Image

class ImageDenoiser:
    def __init__(self, img_path):
        """Initialize the denoiser with an image path."""
        self.img_path = img_path
        self.img = None  # Image will be loaded later
    
    def load_image(self):
        """Load the image from the provided path."""
        self.img = Image.open(self.img_path)
        self.img = self.img.convert("RGB")  # Ensure the image is in RGB format
    
    def median(self, data):
        """Applies a median filter on the given data."""
        data = sorted(data)
        index = len(data) // 2  # Integer division to get the middle index
        return data[index]

    def extract_region(self, img, x, y, offset):
        """Extracts a square region around (x, y) with the given offset."""
        width, height = img.size
        pixels = img.load()
        region = []

        for dx in range(-offset, offset + 1):
            for dy in range(-offset, offset + 1):
                # Handle out-of-bounds coordinates
                nx = min(max(x + dx, 0), width - 1)
                ny = min(max(y + dy, 0), height - 1)
                region.append(pixels[nx, ny])

        return region

    def apply_median_filter(self, channel_img, region_size=3):
        """Applies a median filter to a single image channel with a given region size."""
        width, height = channel_img.size
        imgdup = channel_img.copy()
        pixels = imgdup.load()

        # Calculate the offset based on region size
        offset = region_size // 2

        for x in range(width):
            for y in range(height):
                # Extract the region and apply the median filter
                region = self.extract_region(channel_img, x, y, offset)
                pixels[x, y] = self.median(region)

        return imgdup

    def denoise_rgb(self, region_size=5):
        """Applies the median filter to each RGB channel with a specified region size."""
        # Split the image into its R, G, B channels
        r, g, b = self.img.split()

        # Apply the filter with the updated region size
        r = self.apply_median_filter(r, region_size)
        g = self.apply_median_filter(g, region_size)
        b = self.apply_median_filter(b, region_size)

        # Merge the channels back into an RGB image
        return Image.merge("RGB", (r, g, b))

    def save(self, output_path, format="JPEG"):
        """Save the processed image to the specified path."""
        self.img.save(output_path, format=format)
        print(f"Processed image saved to: {output_path}")

    def show(self):
        """Display the current image."""
        self.img.show()

    def save_image(self, image, output_path):
        """Save the given image to the specified path."""
        image.save(output_path)
        print(f"Processed image saved to: {output_path}")
