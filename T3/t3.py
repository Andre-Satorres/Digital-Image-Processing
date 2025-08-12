import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt

GRANULARITY = 360

def horizontal_proj_technique(img):
    angles = np.linspace(0, 360, GRANULARITY, endpoint=True, retstep=False)
    values = np.zeros_like(angles)
    for i in range(len(angles)):
        profile = np.sum(rotate_image(img, angles[i]), axis=1)
        values[i] = goal_function(profile)

    return angles[np.argmax(values)]


def plot_img_histogram(img, hist):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
    # Plot array values
    ax1.plot(np.arange(len(hist)), hist, marker='o', linestyle='-', color='b', label='Values')
    ax1.set_title('Histogram')
    ax1.set_xlabel('Line')
    ax1.set_ylabel('Value')
    ax1.grid(True)

    ax2.plot(np.arange(len(hist) - 1), (hist[1:] - hist[:-1]) ** 2, marker='o', linestyle='-', color='b', label='Values')
    ax2.set_title("Square of differences")
    ax2.set_xlabel('Line')
    ax2.set_ylabel('Value')
    ax2.grid(True)


    ax3.imshow(img)
    ax3.set_title('Image')

    print(goal_function(hist))
    plt.show()


def hough_technique(img, f):
    edges = cv.Canny(img, 100, 150, apertureSize=3)
    cv.imwrite(f'out/canny/{f}', edges)
    lines = cv.HoughLinesWithAccumulator(edges, 1, 2*np.pi/GRANULARITY, 50)
    angle = lines[0, np.argmax(lines[:,:,2]), 1]
    return np.rad2deg(angle) - 90


def goal_function(profile):
    # Max instead of sum works way better
    return np.max((profile[1:] - profile[:-1]) ** 2)


def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])

    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    return cv.warpAffine(image, M, (new_w, new_h), borderMode=cv.BORDER_CONSTANT, borderValue=(255, 255, 255))


def main():
    mode = input('Mode:')
    # Load the image
    for f in os.listdir('images'):
        image = cv.cvtColor(cv.imread(f'images/{f}'), cv.COLOR_BGR2GRAY)
        if mode == 'proj':
            angle = horizontal_proj_technique(image)
        elif mode == 'hough':
            angle = hough_technique(image, f)
        else:
            raise NotImplementedError(f'Mode {mode} not implemented')
        
        path = f'out/{mode}'
        os.makedirs(path, exist_ok=True)
        cv.imwrite(f'{path}/{f}', rotate_image(image, angle))


main()