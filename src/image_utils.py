import os


def is_image(file_name: str):
    """
    Checks if a file is an image based on its extension.

    Args:
        file_name (str): Name of the file.

    Returns:
        bool: True if the file is an image, False otherwise.
    """
    return file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))


def get_images_subdirectory(directory_path: str):
    """
    Retrieves a list of image files within a directory and its subdirectories.

    Args:
        directory_path (str): Path to the directory.

    Returns:
        list: List of paths to image files.
    """
    return [os.path.join(root, file) for root, _, files in os.walk(directory_path)
            for file in files if is_image(file)]


def iterate_file_lines(database_file: str):
    """
    Iterates over lines in a database file containing descriptors.

    Args:
        database_file (str): Path to the database file.

    Yields:
        tuple: File name and descriptor values.
    """
    try:
        with open(database_file, 'r') as f:
            for descriptor in f:
                descriptor_line = descriptor.strip()
                descriptor_file_name, descriptor_values = descriptor_line.split(" ", 1)
                yield descriptor_file_name, descriptor_values
    except FileNotFoundError:
        print(f"File '{database_file}' is unknown.")
