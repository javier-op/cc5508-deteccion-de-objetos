import matplotlib.pyplot as plt
import numpy as np
import sys

from bw import get_boundary, get_ccomponents, remove_small_components
from pai_io import imread, imsave
from utils import get_threshold_otsu, threshold

def get_yellow(image):
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]
    yellow = 2. * red + 4. * green - 3. * blue
    return np.interp(yellow, (yellow.min(), yellow.max()), (0, 255)).astype(np.uint8)

def circle_formula_coef(cc):
    boundary = cc['boundary']
    p1 = boundary[int(len(boundary) * 0.33)]
    p2 = boundary[int(len(boundary) * 0.66)]
    p3 = boundary[int(len(boundary) * 0.99)]
    a = np.array([[p1[0], p1[1], 1], [p2[0], p2[1], 1], [p3[0], p3[1], 1]])
    b = np.array([-p1[0]**2-p1[1]**2, -p2[0]**2-p2[1]**2, -p3[0]**2-p3[1]**2])
    s = np.linalg.solve(a, b)
    x = -s[0]/2
    y = -s[1]/2
    r = np.sqrt((s[0]**2 + s[1]**2)/4 - s[2])
    return x, y, r

def add_error_to_circle(cc):
    x, y, r = circle_formula_coef(cc)
    boundary = cc['boundary']
    error = 0
    for point in boundary:
        r_calc = np.sqrt((point[0] - x)**2 + (point[1] - y)**2)
        error += np.abs(r_calc - r) / (r + 0.001)
    error /= len(boundary)
    cc['error'] = error
    return cc

def filter_by_error(ccs, threshold):
    output = []
    for cc in ccs:
        if cc['error'] < threshold:
            output.append(cc)
    return output

def write_cc(cc, f):
    for key in cc:
        f.write('{}: {}\n'.format(key, cc[key]))
    f.write('\n')

def draw_all_lbbs(image, ccs, box_rgb_color):
    for cc in ccs:
        y, x, h, w = cc['bbox']
        for i in range(y, y+h):
            image[i, x] = box_rgb_color
            image[i, x+w] = box_rgb_color
        for j in range(x, x+w):
            image[y, j] = box_rgb_color
            image[y+h, j] = box_rgb_color
    return image

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SyntaxError('usage: yellow_circles.py [path to image]')
    input_image = imread(sys.argv[1])
    yellow = get_yellow(input_image)
    yellow_binary = threshold(yellow, get_threshold_otsu(yellow))
    imsave('yellow_binary.png', np.interp(yellow_binary, (yellow_binary.min(), yellow_binary.max()), (0, 255)).astype(np.uint8))
    yellow_ccs = get_ccomponents(yellow_binary)
    yellow_ccs = remove_small_components(yellow_ccs, 50)
    with open('yellow_ccs.txt', 'w') as f:
        for cc in yellow_ccs:
            write_cc(cc, f)
    yellow_ccs = [add_error_to_circle(cc) for cc in yellow_ccs]
    yellow_circle_ccs = filter_by_error(yellow_ccs, 0.1)
    image_with_boxes = draw_all_lbbs(input_image, yellow_circle_ccs, (255, 0, 0))
    imsave('output.png', image_with_boxes)
    plt.imshow(image_with_boxes)
    plt.axis('off')
    plt.show()
