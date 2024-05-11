import time

import cv2

from .descriptors.descriptor import Descriptor
from src.image_utils import get_images_subdirectory


def evaluate(test_directory: str, descriptor_used: Descriptor, descriptor_database: str):
    """
    Evaluates the performance of a descriptor on a test dataset.

    Args:
        test_directory (str): Path to the directory containing test images.
        descriptor_used (Descriptor): Descriptor to use.
        descriptor_database (str): Path to the descriptor database file.
    """
    y_true = []
    y_pred = []

    test_image_list = get_images_subdirectory(test_directory)
    n_images = len(test_image_list)

    start_time = time.time()

    for idx, image_file in enumerate(test_image_list):
        print(f"{100 * idx / n_images:.1f}/100% completed\n"
              f"{time.time() - start_time:.1f}s elapsed\n", end='')
        true_label = image_file.split("/")[-2]
        query_img = cv2.imread(image_file)

        y_true.append(true_label)

        distances_list = descriptor_used.get_prediction(query_img, descriptor_database)
        pred_labels = [distances[0].split("/")[-2] for distances in distances_list]

        y_pred.append(pred_labels)

        # Clear the line for progression
        print('\033[1A', end='\x1b[2K')
        print('\033[1A', end='\x1b[2K')

    end_time = time.time() - start_time

    print(f"{n_images} images processed in {end_time:.1f} s, {end_time / n_images:.3f} per file")
    print(f"The accuracy of the {descriptor_used.name} on the {test_directory} folder "
          f"is {calculate_mean_average_precision(y_pred, y_true) * 100:.2f}%")


def calculate_mean_average_precision(batch_pred: list, batch_true: list):
    """
    Calculates the mean average precision (mAP) for a set of predictions.

    Args:
        batch_pred (list): Predicted labels for each query image.
        batch_true (list): True labels for each query image.

    Returns:
        float: Mean average precision.
    """
    total_map = 0
    num_batches = len(batch_true)
    for batch_idx in range(num_batches):
        y_true = batch_true[batch_idx]
        y_pred = batch_pred[batch_idx]
        num_correct = 0
        total_precision = 0
        for i in range(5):
            if y_pred[i] == y_true:
                num_correct += 1
                precision = num_correct / (i + 1)
                total_precision += precision
        if num_correct == 0:
            avg_precision = 0
        else:
            avg_precision = total_precision / num_correct

        total_map += avg_precision
    return total_map / num_batches
