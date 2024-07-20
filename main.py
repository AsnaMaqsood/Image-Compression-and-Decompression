import cv2
from color_space_inversion import ycbcr_to_bgr,bgr_to_ycbcr
from blocks_splitting import blocks_splitting,blocks_merging
from dct import *
from quantization import *
from tables import *
from huffman import *
from collections import defaultdict

def compress_image(image_path, huffman_codes_path, compressed_path):
    image = cv2.imread(image_path)
    
    ycbcr_image = bgr_to_ycbcr(image)
    
    blocks, height, width = blocks_splitting(ycbcr_image)
    with open('image_dimensions.txt', 'w') as dim_file:
        dim_file.write(f"{height},{width}")

    dct_blocks = apply_dct(blocks)
    
    quantized_blocks = quantize_blocks(dct_blocks, LUMINANCE_QUANT_TABLE)
    
    flat_coeffs = [int(coeff) for block in quantized_blocks for channel in range(3) for row in block[:, :, channel] for coeff in row]
    
    frequencies = defaultdict(int)
    for coeff in flat_coeffs:
        frequencies[coeff] += 1
    
    huffman_codes = generate_huffman_codes(frequencies)
    
    encoded_data = huffman_encode(flat_coeffs, huffman_codes)
    
    save_huffman_codes(huffman_codes, huffman_codes_path)
    
    with open(compressed_path, 'wb') as f:
        f.write(encoded_data.encode())
    
    return encoded_data

def save_huffman_codes(huffman_codes, filename):
    with open(filename, 'w') as file:
        for symbol, code in huffman_codes.items():
            file.write(f"{symbol}:{code}\n")

def decompress_image(huffman_codes_path, compressed_path, output_image_path):
    huffman_codes = load_huffman_codes(huffman_codes_path)
    
    with open(compressed_path, 'rb') as f:
        encoded_data = f.read().decode()
    
    decoded_data = huffman_decode(encoded_data, huffman_codes)
    
    with open('image_dimensions.txt', 'r') as dim_file:
        height, width = map(int, dim_file.read().split(','))
    
    num_blocks = (height // 8) * (width // 8)
    blocks = []
    index = 0
    for _ in range(num_blocks):
        block = np.zeros((8, 8, 3), dtype=np.float32)
        for channel in range(3):
            for i in range(8):
                for j in range(8):
                    block[i, j, channel] = decoded_data[index]
                    index += 1
        blocks.append(block)
    
    dequantized_blocks = dequantize_blocks(blocks, LUMINANCE_QUANT_TABLE)
    
    idct_blocks = apply_idct(dequantized_blocks)
    
    reconstructed_image = blocks_merging(idct_blocks, height, width)
    
    bgr_image = ycbcr_to_bgr(reconstructed_image)
    
    cv2.imwrite(output_image_path, bgr_image)

def load_huffman_codes(filename):
    huffman_codes = {}
    with open(filename, 'r') as file:
        for line in file:
            symbol, code = line.strip().split(':')
            huffman_codes[int(symbol)] = code
    return huffman_codes

def init(input_image_path):
    input_image_path = 'images.jpg'
    
    huffman_codes_path = 'huffman_codes_'+input_image_path[:-4]+'.txt'
    compressed_data_path = 'compressed_data_'+input_image_path[:-4]+'.bin'
    output_image_path = 'output_image_'+input_image_path[:-4]+'.jpg'
    
    compress_image(input_image_path, huffman_codes_path, compressed_data_path)
    print("Image compression completed.")
    
    decompress_image(huffman_codes_path, compressed_data_path, output_image_path)
    print("Image decompression completed.")


init('image4.jpg')