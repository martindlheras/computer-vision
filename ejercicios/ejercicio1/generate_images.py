from PIL import Image
import numpy as np
import random

XSIZE = 10
YSIZE = 10

if __name__ == '__main__':
    img = np.random.randint(0, 256, (1000, 1000, 3))
    print(img.shape)

    pic = Image.fromarray(img.astype('uint8'), 'RGB')
    pic.save('images/random_noise.png')

    zeros = np.zeros((1000, 1000, 3))
    pic = Image.fromarray(zeros.astype('uint8'), 'RGB')
    pic.save('images/black.png')

    ones = 255 * np.ones((1000, 1000, 3))
    pic = Image.fromarray(ones.astype('uint8'), 'RGB')
    pic.save('images/white.png')

    for _ in range(1):

        x_center = random.randrange(0, 900)
        y_center = random.randrange(0, 900)

        myimg = np.zeros((1000, 1000, 3))

        for x_pixel in range(1000):
            for y_pixel in range(1000):
                for channel in range(3):
                    if x_center < x_pixel < x_center + XSIZE and y_center < y_pixel < y_center + YSIZE:
                        myimg[x_pixel][y_pixel][channel] = 255

        mypic = Image.fromarray(myimg.astype('uint8'), 'RGB')
        mypic.save('images/square.png')