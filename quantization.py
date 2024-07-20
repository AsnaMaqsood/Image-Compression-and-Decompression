import numpy as np
from tables import CHROMINANCE_QUANT_TABLE


def quantize_block(block, quant_table):
    return np.round(block / quant_table)

def dequantize_block(block, quant_table):
    return block * quant_table

def quantize_blocks(blocks, quant_table):
    quantized_blocks = []
    for block in blocks:
        quant_block = np.zeros_like(block)
        for channel in range(3):
            if channel == 0:  # Y channel
                quant_block[:, :, channel] = quantize_block(block[:, :, channel], quant_table)
            else:  # Cb and Cr channels
                quant_block[:, :, channel] = quantize_block(block[:, :, channel], CHROMINANCE_QUANT_TABLE)
        quantized_blocks.append(quant_block)
    return quantized_blocks

def dequantize_blocks(blocks, quant_table):
    dequantized_blocks = []
    for block in blocks:
        dequant_block = np.zeros_like(block)
        for channel in range(3):
            if channel == 0:  # Y channel
                dequant_block[:, :, channel] = dequantize_block(block[:, :, channel], quant_table)
            else:  # Cb and Cr channels
                dequant_block[:, :, channel] = dequantize_block(block[:, :, channel], CHROMINANCE_QUANT_TABLE)
        dequantized_blocks.append(dequant_block)
    return dequantized_blocks