import cv2
import numpy as np
from .descriptor import Descriptor


class DominantColorDescriptor(Descriptor):
    def __init__(self, Td, alpha):
        super().__init__("DCD")
        self.Td = Td
        self.alpha = alpha
        self.d_max = Td * alpha

    def get_descriptor(self, img):
        # Prepare the image for K-means function
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        pixels = img_rgb.reshape((-1, 3))
        float_pixels = np.float32(pixels)

        # Apply K-means on the image
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 8
        _, labels, centroids = cv2.kmeans(float_pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centroids = np.uint8(centroids)
        pixels_label = labels.flatten()
        counts = np.bincount(pixels_label)

        # Generate the descriptor by sorting the centroids per counts
        descriptor = f"{len(counts)}"

        for idx in np.argsort(counts)[::-1][:K]:
            color = centroids[idx].tolist()
            percentage = int(counts[idx] / float(len(labels)) * 100)
            descriptor += f" {percentage} {color[0]} {color[1]} {color[2]}"

        return descriptor

    def get_distance(self, d_1, d_2):
        D = 0

        # Extract the number of colors and features from each descriptor
        N1, F1 = d_1[0], d_1[1:]
        N2, F2 = d_2[0], d_2[1:]

        # Calculate the squared sum of percentages for both descriptors
        for i in range(N1):
            D += F1[4 * i] ** 2
        for j in range(N2):
            D += F2[4 * j] ** 2

        # Compute color similarity between the descriptors
        colors_similarity = 0
        for i in range(N1):
            for j in range(N2):
                color_distance = np.linalg.norm(d_1[1 + 4 * i:4 + 4 * i] - d_2[1 + 4 * j:4 + 4 * j])
                a = 0
                if color_distance <= self.Td:
                    a = 1 - color_distance/self.d_max
                colors_similarity += a * F1[4 * i] * F2[4 * j]

        D -= colors_similarity

        return D
