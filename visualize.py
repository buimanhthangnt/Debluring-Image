from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import imutils
import cv2 as cv


def compare(list_image):
    for i, image in enumerate(list_image):
        if i == 2: continue
        list_image[i] = image.reshape((image.shape[1], image.shape[2]))

    fig = plt.figure(figsize=(10, 5))
    num_col = 3
    num_row = 1
    for i in range(1, num_col * num_row +1):
        fig.add_subplot(num_row, num_col, i)
        plt.imshow(list_image[i-1], cmap='gray')
    plt.show()


def resize(image, new_height, new_width):
    width = image.shape[1]
    height = image.shape[0]
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.resize(image, (new_width, new_height))
    return image