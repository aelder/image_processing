from PIL import Image
import os

def average_image_color(image):
    width, height = image.size
    pixels = image.load()
    r_total, g_total, b_total = 0, 0, 0

    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            r_total += r
            g_total += g
            b_total += b

    num_pixels = width * height
    r_avg = r_total // num_pixels
    g_avg = g_total // num_pixels
    b_avg = b_total // num_pixels

    return r_avg, g_avg, b_avg

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(".tif"):
            file_path = os.path.join(input_folder, file_name)
            image = Image.open(file_path)

            avg_color = average_image_color(image)
            new_image = Image.new('RGB', (image.width, 1), avg_color)

            output_file_path = os.path.join(output_folder, file_name)
            new_image.save(output_file_path)

input_folder = "\input"
output_folder = "\output"

process_images(input_folder, output_folder)