import time

import cv2

from .image_utils import get_images_subdirectory
from .descriptors.descriptor import Descriptor


def train(train_directory: str, descriptor_used: Descriptor, output_file: str):
    """
    Uses selected descriptor on a set of images and saves the results to a file.

    Args:
        train_directory (str): Path to the directory containing training images.
        descriptor_used (str): Descriptor to use.
        output_file (str): Path to the output file.
    """
    start_time = time.time()
    train_image_list = get_images_subdirectory(train_directory)
    n_images = len(train_image_list)

    with open(output_file, 'w') as f:
        for idx, image_file in enumerate(train_image_list):
            print(f"{100 * idx / n_images:.1f}/100% completed\n"
                  f"{time.time() - start_time:.1f}s elapsed\n", end='')
            img = cv2.imread(image_file)
            descriptor = descriptor_used.get_descriptor(img)
            f.write(f"{image_file} {descriptor}\n")

            # Clear the line for progression
            print('\033[1A', end='\x1b[2K')
            print('\033[1A', end='\x1b[2K')

    end_time = time.time() - start_time
    print(f"\nUsed '{descriptor_used.name}' descriptors on files from the folder '{train_directory}'. "
          f"Results are saved in '{output_file}'.")
    print(f"{n_images} images processed in {end_time:.1f} s, {end_time / n_images:.3f} per file")
