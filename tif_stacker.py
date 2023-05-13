from PIL import Image
import os
from tqdm import tqdm

def stack_tiff_images(input_folder, output_file):
    tiff_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".tif")]

    images = []
    total_height = 0
    max_width = 0

    for file_name in tqdm(tiff_files, desc="Loading images"):
        file_path = os.path.join(input_folder, file_name)
        with Image.open(file_path) as image:
            images.append(image.copy())

            width, height = image.size
            total_height += height
            max_width = max(max_width, width)

    stacked_image = Image.new("RGB", (max_width, total_height))
    current_height = 0

    for image in tqdm(images, desc="Stacking images"):
        stacked_image.paste(image, (0, current_height))
        current_height += image.size[1]
        image.close()

    stacked_image.save(output_file)

input_folder = "\input"
output_file = "\stacked_output.tif"

stack_tiff_images(input_folder, output_file)