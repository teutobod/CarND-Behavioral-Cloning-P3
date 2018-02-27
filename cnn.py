from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda
from keras.layers import Convolution2D, Cropping2D

def  nvidia():
    image_shape = (160, 320, 3)

    model = Sequential()

    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50, 20), (0, 0))))


    kernel_size = 5
    stride = (2, 2)

    model.add(Convolution2D(nb_filter=24, nb_row=kernel_size, nb_col=kernel_size, subsample=stride, activation='relu'))
    model.add(Convolution2D(nb_filter=36, nb_row=kernel_size, nb_col=kernel_size, subsample=stride, activation='relu'))
    model.add(Convolution2D(nb_filter=48, nb_row=kernel_size, nb_col=kernel_size, subsample=stride, activation='relu'))

    kernel_size = 3
    stride = (1, 1)
    model.add(Convolution2D(nb_filter=64, nb_row=kernel_size, nb_col=kernel_size, subsample=stride, activation='relu'))
    model.add(Convolution2D(nb_filter=64, nb_row=kernel_size, nb_col=kernel_size, subsample=stride, activation='relu'))

    model.add(Flatten())

    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))

    model.add(Dense(1))

    return model
