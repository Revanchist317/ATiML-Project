import math

import cv2
import numpy as np
from .descriptor import Descriptor


def convert_hmmd(bgr_img):
    """
   Converts an RGB image to HMMD color space.

   Parameters:
       bgr_img (numpy.ndarray): The input BGR image.

   Returns:
       numpy.ndarray: The HMMD image.
   """
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    hue_channel = hsv_img[:, :, 0]

    max_rgb = np.amax(bgr_img, axis=2)
    min_rgb = np.amin(bgr_img, axis=2)

    hmmd_img = np.zeros_like(bgr_img, dtype=np.uint8)

    hmmd_img[:, :, 0] = hue_channel
    hmmd_img[:, :, 1] = max_rgb - min_rgb
    hmmd_img[:, :, 2] = ((max_rgb + min_rgb) / 2).astype(np.uint8)

    return hmmd_img


class ColorStructureDescriptor(Descriptor):
    def __init__(self, n_quantization):
        super().__init__("CSD")
        self.n_quantization = n_quantization

        subspaces_quantizations_list = {256: np.array([(1, 32), (4, 8), (16, 4), (16, 4), (16, 4)]),
                                        128: np.array([(1, 16), (4, 4), (8, 4), (8, 4), (8, 4)]),
                                        64: np.array([(1, 8), (4, 4), (4, 4), (8, 2), (8, 1)]),
                                        32: np.array([(1, 8), (4, 4), (4, 1), (4, 1)])}

        # Compute the number of values for each subspace for the correct number of quantization
        self.subspaces_quantizations = subspaces_quantizations_list[self.n_quantization]

        subspace_values = [0]
        gap_sum = 0
        for quantization in self.subspaces_quantizations[:-1]:
            gap_sum += quantization[0] * quantization[1]
            subspace_values.append(gap_sum)

        self.subspace_values = np.array(subspace_values)

    def get_descriptor(self, img):
        hmmd_img = convert_hmmd(img)

        # Get the spatial extent based on the image size
        w, h = img.shape[:2]
        p = max(0, round(0.5 * math.log2(w * h) - 8))
        K = 2 ** p

        quantized_img = self.get_quantizations(hmmd_img)
        descriptor_hist = np.zeros(self.n_quantization, dtype=np.uint8)

        # Get the number of structuring element for each color where it appears at least once
        for y in range(0, w - 8, 8*K):
            for x in range(0, h - 8, 8*K):
                struct_quantized = quantized_img[y:y + 8*K:K, x:x + 8*K:K]
                counts = np.unique(struct_quantized)
                descriptor_hist[counts] += 1

        descriptor = ' '.join(map(str, descriptor_hist))

        return descriptor

    def get_distance(self, d_1, d_2):
        D = np.sum(np.abs(d_1 - d_2))

        return D

    def get_quantizations(self, hmmd_img):
        """
        Computes quantized values based on HMMD image channels.

        Parameters:
            hmmd_img (numpy.ndarray): The HMMD image.

        Returns:
            numpy.ndarray: The color quantization values.
        """
        hue_channel = hmmd_img[:, :, 0]
        diff_channel = hmmd_img[:, :, 1]
        sum_channel = hmmd_img[:, :, 2]

        # Get the subspace for each color of the image
        if self.n_quantization == 32:
            subspace = np.digitize(diff_channel, [6, 60, 110])
        else:
            subspace = np.digitize(diff_channel, [6, 20, 60, 110])

        # Get the hue position based on the number of hue quantization in the corresponding subspace
        hue_gaps = (hue_channel / (256 / self.subspaces_quantizations[subspace, 0])).astype(int)

        # Compute the diff (subspace), hue and sum contribution to the color quantization
        subspace_repartition = self.subspace_values[subspace]
        hue_repartition = hue_gaps * self.subspaces_quantizations[subspace, 1]
        sum_repartition = (sum_channel / (256 / self.subspaces_quantizations[subspace, 1])).astype(int)

        color_quantization = subspace_repartition + hue_repartition + sum_repartition

        return color_quantization.astype(np.uint8)
