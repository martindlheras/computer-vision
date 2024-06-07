from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

KXSIZE = 10
KYSIZE = 10
DETECTION_THRESHOLD = 245

if __name__ == '__main__':
    pic = Image.open('img/square.png')
    image = np.array(pic)

    xsize, ysize, nchannels = image.shape

    r_image = np.array(image[:, :, 0])
    r_filter = np.ones((KXSIZE, KYSIZE))

    tensor_product = np.tensordot(r_filter, r_image, axes=0)

    distance_matrix = np.sum(tensor_product, axis=(0,1))
    distance_matrix /= KXSIZE * KYSIZE

    distance_matrix = np.array([distance_matrix, distance_matrix, distance_matrix])
    distance_matrix = np.transpose(distance_matrix, (1, 2, 0))

    x_centers, y_centers = np.where(distance_matrix[:, :, 0] > DETECTION_THRESHOLD)

    for x_pixel in range(xsize):
        for y_pixel in range(ysize):
            if x_pixel in x_centers or y_pixel in y_centers:
                distance_matrix[x_pixel][y_pixel] = [255, 0, 0]

                image[x_pixel][y_pixel] = [255, 0, 0]

    pic = Image.fromarray(image.astype('uint8'), 'RGB')
    pic.save('img/image_detected_tensor.png')