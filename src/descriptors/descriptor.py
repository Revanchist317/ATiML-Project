import numpy as np

from src.image_utils import iterate_file_lines


class Descriptor:
    """
    Base class for image descriptors.

    Args:
        name: Name of the descriptor

    Methods:
        __init__: Initializes the Descriptor class.
        get_descriptor: Method to compute the descriptor for an image.
        get_distance: Method to compute the distance between two descriptors.
    """
    def __init__(self, name):
        super().__init__()
        self.name = name

    def get_descriptor(self, img) -> str:
        """
        Compute the descriptor for an image.

        Args:
            img (numpy.ndarray): The input image.

        Returns:
            str: The computed descriptor for the input image.
        """
        pass

    def get_distance(self, d_1, d_2) -> float:
        """
        Compute the distance between two descriptors.

        Args:
            d_1 (numpy.ndarray): Descriptor array of the first image.
            d_2 (numpy.ndarray): Descriptor array of the second image.

        Returns:
            float: The distance between the two descriptors.
        """
        pass

    def get_prediction(self, query_img, descriptor_database: str):
        """
        Retrieves predictions for a query image based on descriptors in a database.

        Args:
            query_img (numpy.ndarray): Query image array.
            descriptor_database (str): Path to the descriptor database file.

        Returns:
            list: Predicted labels for the query image.
        """
        distances_list = []
        descriptor_query_img = self.get_descriptor(query_img)

        for file_name, descriptor in iterate_file_lines(descriptor_database):
            d_1 = np.array(descriptor_query_img.split(), dtype=int)
            d_2 = np.array(descriptor.split(), dtype=int)
            distance = self.get_distance(d_1, d_2)
            distances_list.append((file_name, distance))

        # Keep the 5 smaller distances
        distances_list.sort(key=lambda x: x[1])
        distances_list = distances_list[:5]

        return distances_list
