import numpy as np

def blocks_splitting(image_data):
    height, width, _ = image_data.shape
    
    image_data = image_data.astype(np.float32)
    blocks = []
    
    height = height - (height % 8)
    width = width - (width % 8)
    
    
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            block = image_data[y:y+8, x:x+8, :]
            blocks.append(block)
    
    return blocks, height, width

def blocks_merging(blocks, height, width):
    reconstructed_image = np.zeros((height, width, 3), dtype=np.float32)
    
    index = 0
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            reconstructed_image[y:y+8, x:x+8, :] = blocks[index]
            index += 1
    
    return reconstructed_image.astype(np.uint8)
