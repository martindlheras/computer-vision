from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

KXSIZE = 10
KYSIZE = 10
DETECTION_THRESHOLD = 200

def kernel(image, k_x, k_y, k_c, k_xsize=KXSIZE, k_ysize=KYSIZE):
    
    convolution = np.sum(image[k_x:k_x+k_xsize, k_y:k_y+k_ysize, 0])
    convolution /= k_xsize * k_ysize

    return convolution

if __name__ == "__main__":
    pic = Image.open('images/square.png')
    image = np.array(pic)

    xsize, ysize, nchannels = image.shape

    distance_matrix = np.zeros((xsize, ysize, nchannels))
    for x_pixel in range(xsize - KXSIZE):
        for y_pixel in range(ysize - KYSIZE):
            dd = kernel(image, x_pixel, y_pixel, 0)
            print(f'c: {0}, x: {x_pixel}, y: {y_pixel} -- {dd}')
            for channel in range(nchannels):
                distance_matrix[x_pixel + int(KXSIZE/2)][y_pixel + int(KYSIZE/2)][channel] = dd

    x_centers, y_centers = np.where(distance_matrix[:, :, 0] > DETECTION_THRESHOLD)

    for x_pixel in range(xsize):
        for y_pixel in range(ysize):
            if x_pixel in x_centers or y_pixel in y_centers:
                distance_matrix[x_pixel][y_pixel] = [255, 0, 0]

                image[x_pixel][y_pixel] = [255, 0, 0]

    pic = Image.fromarray(distance_matrix.astype(np.uint8), 'RGB')
    pic.save('images/distance_matrix.png')

    pic = Image.fromarray(image.astype(np.uint8), 'RGB')
    pic.save('images/image_detected.png')