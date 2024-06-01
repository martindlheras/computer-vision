from PIL import Image, ImageDraw
import numpy as np
import random
import scipy
import seaborn as sns

XSIZE = 100
YSIZE = 100

def gaussian_2d(A, x_center, y_center, sigma=10, height=XSIZE, width=YSIZE):
    """
    Generate a gaussian image with a given center and sigma
    
    Args:
        A (int): The amplitude of the gaussian
        x_center (int): The x coordinate of the center of the gaussian
        y_center (int): The y coordinate of the center of the gaussian
        sigma (int): The sigma of the gaussian
        height (int): The height of the image
        width (int): The width of the image

    Returns:
        np.array: The gaussian image normalized between 0 and A
    """
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    xy = np.vstack((x.ravel(), y.ravel()))
    x, y = xy

    gaussian = np.exp(-((x - x_center)**2 + (y - y_center)**2) / (2 * sigma**2))

    normalized_gaussian = gaussian / np.max(gaussian) * A

    return normalized_gaussian

def generate_gaussian_image(A, x_center, y_center, sigma=10, height=XSIZE, width=YSIZE):
    """
    Generate a gaussian image with a given center and sigma
    
    Args:
        A (int): The amplitude of the gaussian
        x_center (int): The x coordinate of the center of the gaussian
        y_center (int): The y coordinate of the center of the gaussian
        sigma (int): The sigma of the gaussian
        height (int): The height of the image
        width (int): The width of the image
    
    Returns:
        Image: The gaussian image
    """
    gaussian = gaussian_2d(A, x_center, y_center, sigma, height, width)
    gaussian_rgb = np.stack((gaussian.reshape(height, width), gaussian.reshape(height, width), gaussian.reshape(height, width)), axis=-1)
    return Image.fromarray(gaussian_rgb.astype('uint8'), 'RGB')

if __name__ == '__main__':

    x_center_original = random.randrange(XSIZE)
    y_center_original = random.randrange(YSIZE)
    A = random.randint(1, 255)

    print('We generate a gaussian image with the following parameters:')
    print(f'\tCenter: ({x_center_original}, {y_center_original})')
    print(f'\tA: {A}')

    gaussian = gaussian_2d(A, x_center_original, y_center_original)
    pic = generate_gaussian_image(A, x_center_original, y_center_original)
    pic.save('img/gaussian.png')

    distance = np.zeros((XSIZE, YSIZE))
    solutions = np.zeros((XSIZE, YSIZE, 3))

    for x_pixel in range(XSIZE):
        for y_pixel in range(YSIZE):
            def func(A):
                return np.sum(np.abs(gaussian_2d(A, x_pixel, y_pixel) - gaussian))
            opt = scipy.optimize.least_squares(func, [255])
            distance[x_pixel][y_pixel] = opt.cost
            solutions[x_pixel][y_pixel] = opt.x

    
    x_center, y_center = np.where(distance == np.min(distance))
    
    print('We detect the following values:')
    print(f'\tCenter: ({x_center[0]}, {y_center[0]})')
    print(f'\tA: {solutions[x_center[0]][y_center[0]][0]}')

    heatmap = sns.heatmap(distance.T, xticklabels=False, yticklabels=False, cbar=False)
    heatmap.get_figure().savefig('img/heatmap.png')

    gaussian_detection = Image.open('img/gaussian.png')
    painter = ImageDraw.Draw(gaussian_detection)
    painter.line([(x_center[0], 0), (x_center[0], YSIZE)], fill='red')
    painter.line([(0, y_center[0]), (XSIZE, y_center[0])], fill='red')
    painter.line([(x_center_original, 0), (x_center_original, YSIZE)], fill='green')
    painter.line([(0, y_center_original), (XSIZE, y_center_original)], fill='green')
    gaussian_detection.save('img/gaussian_detection.png')

    noise_levels = [10, 50, 100, 500]
    print(f'Now we add Poisson noise to the image in 4 levels: {noise_levels} and try again.')

    for level in noise_levels:
        noise = np.random.poisson(level, XSIZE*YSIZE)
        noisy_gaussian = (gaussian + noise) / np.max(gaussian + noise) * A
        noisy_gaussian_rgb = np.stack((noisy_gaussian.reshape(XSIZE, YSIZE), noisy_gaussian.reshape(XSIZE, YSIZE), noisy_gaussian.reshape(XSIZE, YSIZE)), axis=-1)
        noisy_gaussian_image = Image.fromarray(noisy_gaussian_rgb.astype('uint8'), 'RGB')
        noisy_gaussian_image.save(f'img/noisy_gaussian_{level}.png')
        distance = np.zeros((XSIZE, YSIZE))
        solutions = np.zeros((XSIZE, YSIZE, 3))

        for x_pixel in range(XSIZE):
            for y_pixel in range(YSIZE):
                def func(A):
                    return np.sum(np.abs(gaussian_2d(A, x_pixel, y_pixel) - noisy_gaussian))
                opt = scipy.optimize.least_squares(func, [255])
                distance[x_pixel][y_pixel] = opt.cost
                solutions[x_pixel][y_pixel] = opt.x

        x_center, y_center = np.where(distance == np.min(distance))

        print(f'Level {level} values:')
        print(f'\tCenter: ({x_center[0]}, {y_center[0]})')
        print(f'\tA: {solutions[x_center[0]][y_center[0]][0]}')

        noisy_gaussian_rgb = Image.open(f'img/noisy_gaussian_{level}.png')
        painter = ImageDraw.Draw(noisy_gaussian_rgb)
        painter.line([(x_center[0], 0), (x_center[0], YSIZE)], fill='red')
        painter.line([(0, y_center[0]), (XSIZE, y_center[0])], fill='red')
        painter.line([(x_center_original, 0), (x_center_original, YSIZE)], fill='green')
        painter.line([(0, y_center_original), (XSIZE, y_center_original)], fill='green')
        noisy_gaussian_rgb.save(f'img/noisy_gaussian_detection_{level}.png')
        