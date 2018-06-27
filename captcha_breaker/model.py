from keras.models import *
from keras.layers import *
from captcha_breaker import setting


def simple():
    input_tensor = Input((setting.HEIGHT, setting.WIDTH, 1))
    x = input_tensor
    for i in range(3):
        x = Convolution2D(32*2**i, (3, 3), activation='relu')(x)
        x = Convolution2D(32*2**i, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dropout(0.25)(x)
    x = [Dense(setting.CHAR_SET_LEN, activation='softmax', name='c%d'%(i+1))(x) for i in range(4)]
    model = Model(input=input_tensor, output=x)
    return model
