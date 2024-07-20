import cv2
import numpy as np


def apply_dct(blocks):
    dct_blocks = []
    for block in blocks:
        dct_block = np.zeros_like(block)
        for channel in range(3):
            dct_block[:, :, channel] = cv2.dct(block[:, :, channel])
        dct_blocks.append(dct_block)
    return dct_blocks

def apply_idct(blocks):
    idct_blocks = []
    for block in blocks:
        idct_block = np.zeros_like(block)
        for channel in range(3):
            idct_block[:, :, channel] = cv2.idct(block[:, :, channel])
        idct_blocks.append(idct_block)
    return idct_blocks