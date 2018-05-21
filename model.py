from keras.models import Model, optimizers, load_model
from keras.layers import Conv2D, Input, Deconv2D, Add
from data_utils import get_data
import os
import cv2 as cv
from visualize import resize, compare

class AutoEncoderDecoder:
    def __init__(self, height, width, channel):
        self.height = height
        self.width = width
        self.channel = channel


    def build(self, lr=0.0002):
        self.input_img = Input(shape=(self.height, self.width, self.channel))

        conv1 = Conv2D(48, (7, 7), activation='relu', padding='same')(self.input_img)
        conv1 = Conv2D(48, (7, 7), activation='relu', padding='same')(conv1)

        conv2 = Conv2D(64, (5, 5), activation='relu', padding='same')(conv1)
        conv2 = Conv2D(64, (5, 5), activation='relu', padding='same')(conv2)

        conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)

        conv4 = Conv2D(96, (3, 3), activation='relu', padding='same')(conv3)
        conv4 = Conv2D(96, (3, 3), activation='relu', padding='same')(conv4)

        dconv1 = Deconv2D(96, (3,3), activation='relu', padding='same')(conv4)
        dconv1 = Deconv2D(96, (3,3), activation='relu', padding='same')(dconv1)
        dconv1 = Add()([dconv1, conv4])

        dconv2 = Deconv2D(64, (3,3), activation='relu', padding='same')(dconv1)
        dconv2 = Deconv2D(64, (3,3), activation='relu', padding='same')(dconv2)
        dconv2 = Add()([dconv2, conv3])

        dconv3 = Deconv2D(64, (5,5), activation='relu', padding='same')(dconv2)
        dconv3 = Deconv2D(64, (5,5), activation='relu', padding='same')(dconv3)
        dconv3 = Add()([dconv3, conv2])

        dconv4 = Deconv2D(48, (7,7), activation='relu', padding='same')(dconv3)
        dconv4 = Deconv2D(48, (7,7), activation='relu', padding='same')(dconv4)
        dconv4 = Add()([dconv4, conv1])

        self.output_img = Conv2D(self.channel, (7, 7), activation='relu', padding='same')(dconv4)

        optimizer = optimizers.Adam(lr=lr)
        self.model = Model(self.input_img, self.output_img)
        self.model.compile(optimizer=optimizer, loss='mse')
        self.model.summary()


    def train(self):
        x_train, x_test, y_train, y_test = get_data()
        self.model.fit(x_train, y_train,
                        epochs=36,
                        batch_size=32,
                        shuffle=True,
                        validation_data=(x_test, y_test))


    def load_weights(self, path='weights.h5'):
        self.model.set_weights(load_model(path).get_weights())


    def sample(self):
        folder = 'test'
        for image in os.listdir(folder):
            image = cv.imread(os.path.join(folder, image))
            origin = resize(image, 56, 46)
            blur = cv.GaussianBlur(origin, (7,7), 0)
            blur = blur.reshape((1, blur.shape[0], blur.shape[1], 1))
            delur_img = self.model.predict(blur)
            compare([blur, delur_img, origin])
