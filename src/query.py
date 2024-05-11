import cv2
from matplotlib import pyplot as plt

from .descriptors.descriptor import Descriptor


def query(img_name: str, descriptor_used: Descriptor, descriptor_database: str):
    """
    Displays matching images for a query image.

    Args:
        img_name (str): Path to the query image.
        descriptor_used (Descriptor): Descriptor to use.
        descriptor_database (str): Path to the descriptor database file.
    """
    query_img = cv2.imread(img_name)

    distances_list = descriptor_used.get_prediction(query_img, descriptor_database)
    pred_labels = [distances[0].split("/")[-2] for distances in distances_list]

    fig = plt.figure(figsize=(15, 7))
    add_image_to_plot(fig, 1, query_img, "Query image")

    for i in range(5):
        match_image = cv2.imread(distances_list[i][0])

        add_image_to_plot(fig, i + 2, match_image, f"Match {i + 1}", pred_labels[i])

    plt.show()
    plt.close(fig)


def add_image_to_plot(fig, n: int, img, title: str, label: str = None):
    """
    Adds an image to a plot.

    Args:
        fig (matplotlib.figure.Figure): Figure object.
        n (int): Position index for the subplot.
        img (numpy.ndarray): Image array.
        title (str): Title for the subplot.
        label (str, optional): Label for the image. Defaults to None.
    """
    ax = fig.add_subplot(2, 3, n)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(title)
    if label:
        plt.text(0.5, -0.03, label,
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax.transAxes, color='black', fontsize=8)
