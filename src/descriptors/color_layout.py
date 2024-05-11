import math

import cv2
import numpy as np
from .descriptor import Descriptor


def zigzag_scan(matrix, num_values: int):
    """
    Performs a zigzag scan on a 2D matrix.

    Args:
        matrix (np.ndarray): The 2D matrix to be scanned.
        num_values (int): The number of values to be scanned.

    Returns:
        list: The scanned values in zigzag order.
    """
    result = []
    row, col = 0, 0
    direction = 1  # 1 for upward traversal, -1 for downward traversal

    while len(result) < num_values:
        result.append(matrix[row][col])

        if direction == 1:  # Upward traversal
            if col == 7:  # If at last column, move downward
                row += 1
                direction = -1
            elif row == 0:  # If at first row, move right
                col += 1
                direction = -1
            else:  # Move diagonally upward
                row -= 1
                col += 1
        else:  # Downward traversal
            if row == 7:  # If at last row, move right
                col += 1
                direction = 1
            elif col == 0:  # If at first column, move downward
                row += 1
                direction = 1
            else:  # Move diagonally downward
                row += 1
                col -= 1

    return result


class ColorLayoutDescriptor(Descriptor):
    def __init__(self, y_coeff_number: int, c_coeff_number: int, w_y, w_cr, w_cb):
        super().__init__("CLD")

        self.y_coeff_number = y_coeff_number
        self.c_coeff_number = c_coeff_number

        self.w_y = w_y
        self.w_cr = w_cr
        self.w_cb = w_cb

    def get_descriptor(self, img):
        # Divide the image in 64 blocks (8x8)
        num_blocks_x = 8
        num_blocks_y = 8

        block_height = img.shape[0] / num_blocks_y
        block_width = img.shape[1] / num_blocks_x

        new_image = np.zeros((8, 8, 3), dtype=np.uint8)

        # Calculate the average color for each block
        for y in range(num_blocks_y):
            for x in range(num_blocks_x):
                start_x = int(x * block_width)
                start_y = int(y * block_height)
                end_x = int((x + 1) * block_width)
                end_y = int((y + 1) * block_height)

                block = img[start_y:end_y, start_x:end_x]

                avg = np.mean(block, axis=(0, 1))

                new_image[y:y+1, x:x+1] = avg

        # Get DCT coefficient for each Y, Cr, Cb channels
        image_ycrcb = cv2.cvtColor(new_image, cv2.COLOR_BGR2YCrCb)

        y, cr, cb = cv2.split(np.float32(image_ycrcb))

        dct_y = cv2.dct(y).astype(int)
        dct_cr = cv2.dct(cr).astype(int)
        dct_cb = cv2.dct(cb).astype(int)

        # Zigzag scanning to get coefficients
        coeff_list = zigzag_scan(dct_y, self.y_coeff_number) + \
                    zigzag_scan(dct_cr, self.c_coeff_number) + \
                    zigzag_scan(dct_cb, self.c_coeff_number)

        str_coeff = [str(x) for x in coeff_list]
        descriptor = " ".join(str_coeff)

        return descriptor

    def get_distance(self, d_1, d_2):
        s_y = self.w_y * sum((d_1[i] - d_2[i]) ** 2 for i in range(self.y_coeff_number))
        s_cr = self.w_cr * sum((d_1[i] - d_2[i]) ** 2 for i in range(self.y_coeff_number,
                                                                     self.y_coeff_number + self.c_coeff_number))
        s_cb = self.w_cb * sum((d_1[i] - d_2[i]) ** 2 for i in range(self.y_coeff_number + self.c_coeff_number,
                                                                     len(d_1)))

        D = math.sqrt(s_y) + math.sqrt(s_cr) + math.sqrt(s_cb)

        return D
