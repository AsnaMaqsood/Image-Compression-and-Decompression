import numpy as np

def bgr_to_ycbcr(bgr_image):
    bgr_image = bgr_image.astype(np.float32)

    ycbcr_image = np.zeros_like(bgr_image)
    ycbcr_image[..., 0] = 0.299 * bgr_image[..., 2] + 0.587 * bgr_image[..., 1] + 0.114 * bgr_image[..., 0]  # Y channel
    ycbcr_image[..., 1] = 128 - 0.168736 * bgr_image[..., 2] - 0.331264 * bgr_image[..., 1] + 0.5 * bgr_image[..., 0]  # Cb channel
    ycbcr_image[..., 2] = 128 + 0.5 * bgr_image[..., 2] - 0.418688 * bgr_image[..., 1] - 0.081312 * bgr_image[..., 0]  # Cr channel

    ycbcr_image = np.clip(ycbcr_image, 0, 255)

    ycbcr_image = ycbcr_image.astype(np.uint8)

    return ycbcr_image


def ycbcr_to_bgr(ycbcr_image):
    ycbcr_image = ycbcr_image.astype(np.float32)
    
    bgr_image = np.zeros_like(ycbcr_image,dtype=float)
    bgr_image[:,:,2] = ycbcr_image[:,:,0] + 1.402 * (ycbcr_image[:,:,2] - 128) # Blue channel
    bgr_image[:,:,1] = ycbcr_image[:,:,0] - 0.344136 * (ycbcr_image[:,:,1] - 128) - 0.714136 * (ycbcr_image[:,:,2] - 128) # Green channel
    bgr_image[:,:,0] = ycbcr_image[:,:,0] + 1.772 * (ycbcr_image[:,:,1] - 128) # Red channel
    bgr_image = np.clip(bgr_image, 0, 255)

    bgr_image = bgr_image.astype(np.uint8)
    return bgr_image