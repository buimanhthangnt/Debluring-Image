from __future__ import division
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import imutils

def show(image):
    plt.imshow(image, cmap='gray')
    plt.show()


def read_pgm(pgmf):
    pgmf.readline()
    (width, height) = [int(i) for i in pgmf.readline().split()]
    depth = int(pgmf.readline())
    assert depth <= 255

    raster = []
    for y in range(height):
        row = []
        for y in range(width):
            row.append(ord(pgmf.read(1)))
        raster.append(row)
    return np.array(raster).astype(np.uint8)


def read_face_dataset(low, high):
    dataset = []
    folder = "orl_faces"
    for i in range(low, high):
        for file in os.listdir(os.path.join(folder, 's' + str(i))):
            filepath = os.path.join(folder, 's' + str(i), file)
            pgm = open(filepath, 'rb')
            image = read_pgm(pgm)
            pgm.close()
            dataset.append(image)
    return dataset


def generate_data(dataset):
    data = []
    labels = []
    for idx, image in enumerate(dataset):
        image = imutils.resize(image, width=46)
        labels += [image, image]
        
        img = cv.GaussianBlur(image, (5, 5), 0)
        data.append(img)
        
        img = cv.GaussianBlur(image, (7, 7), 0)
        data.append(img)
        
    return np.array(data), np.array(labels)


def get_data():
    if os.path.isfile('data.pkl'):
        print("Loading generated data")
        return pickle.load(open('data.pkl', 'rb'))
    else:
        print("Generating data")
        train_dataset = read_face_dataset(1, 36)
        x_train, y_train = generate_data(train_dataset)
        x_train = x_train.reshape((-1, x_train.shape[1], x_train.shape[2], 1))
        y_train = y_train.reshape((-1, y_train.shape[1], y_train.shape[2], 1))

        test_dataset = read_face_dataset(36, 41)
        x_test, y_test = generate_data(test_dataset)
        x_test = x_test.reshape((-1, x_test.shape[1], x_test.shape[2], 1))
        y_test = y_test.reshape((-1, y_test.shape[1], y_test.shape[2], 1))

        pickle.dump([x_train, x_test, y_train, y_test], open('data.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

        return x_train, x_test, y_train, y_test


def write_images():
    dataset = read_face_dataset(1, 41)
    for idx, image in enumerate(dataset):
        cv.imwrite('images/' + str(idx) + '.jpg', image)
