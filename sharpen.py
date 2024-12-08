from PIL import Image
import sys

class ImageSharpener:
    def __init__(self, img_path):
        """Initialize the sharpener with an image path."""
        self.img = Image.open(img_path)
        self.img = self.img.convert("RGB")  # Ensure the image is in RGB format
    
    def laplace(self, region):
        """Apply Laplace operator to a given 3x3 pixel region (for each RGB channel)."""
        # Split the region into separate R, G, B components for each pixel
        r_values = [coord[0] for coord in region]  # Extract R values
        g_values = [coord[1] for coord in region]  # Extract G values
        b_values = [coord[2] for coord in region]  # Extract B values
        
        # Apply Laplace operator to each channel
        r_result = sum(r_values[1:5]) - 4 * r_values[0]
        g_result = sum(g_values[1:5]) - 4 * g_values[0]
        b_result = sum(b_values[1:5]) - 4 * b_values[0]
        
        # Return a tuple of the results for each channel
        return (r_result, g_result, b_result)
    
    def minus(self, A, B):
        """Subtract image B from image A pixel by pixel."""
        width, height = A.size
        imgdup = A.copy()
        pixels = imgdup.load()

        A = A.copy().load()
        B = B.copy().load()

        for x in range(width):
            for y in range(height):
                r = tuple(min(255, max(0, A[x, y][i] - B[x, y][i])) for i in range(3))
                pixels[x, y] = r

        return imgdup

    def sharpen(self):
        """Sharpen the image using the Laplace operator and subtract the edge."""
        edges = self.apply_filter(self.img, self.laplace)
        sharpened_img = self.minus(self.img, edges)
        return sharpened_img

    def apply_filter(self, img, filter_func):
        """Apply a given filter function to the image."""
        width, height = img.size
        imgdup = img.copy()
        pixels = imgdup.load()

        for x in range(width):
            for y in range(height):
                # Get the 3x3 region around the pixel
                region = self.region3x3(img, x, y)
                result = filter_func(region)
                # Apply the result to the pixel
                pixels[x, y] = result
        
        return imgdup

    def region3x3(self, img, x, y):
        """Get the 3x3 pixel region around (x, y)."""
        me = self.getpixel(img, x, y)
        N = self.getpixel(img, x, y - 1)
        S = self.getpixel(img, x, y + 1)
        E = self.getpixel(img, x + 1, y)
        W = self.getpixel(img, x - 1, y)
        NW = self.getpixel(img, x - 1, y - 1)
        NE = self.getpixel(img, x + 1, y - 1)
        SE = self.getpixel(img, x + 1, y + 1)
        SW = self.getpixel(img, x - 1, y + 1)

        return [me, N, E, S, W, NW, NE, SE, SW]

    def getpixel(self, img, x, y):
        """Safely get the pixel value, handling out-of-bounds coordinates."""
        width, height = img.size
        pixels = img.load()

        if x < 0: x = 0
        if x >= width: x = width - 1
        if y < 0: y = 0
        if y >= height: y = height - 1

        return pixels[x, y]

    def save(self, output_path):
        """Save the sharpened image to the specified path."""
        self.img.save(output_path)
        print(f"Saved sharpened image to {output_path}")


# Main script (example)
if __name__ == "__main__":
    input_path = "twice-denoised.png"  # Replace with your image path
    output_path = "sharpened_output.png"  # Define the output path

    sharpener = ImageSharpener(input_path)
    
    # Apply sharpening
    sharpened_img = sharpener.sharpen()
    
    # Show the sharpened image
    sharpened_img.show()

    # Save the sharpened image
    sharpener.save(output_path)