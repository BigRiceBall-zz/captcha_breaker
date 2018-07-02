from keras.models import *
from keras.layers import *
from captcha_breaker import setting
from keras import regularizers


def simple():
    input_tensor = Input((setting.HEIGHT, setting.WIDTH, 1))
    x = input_tensor
    x = noise.GaussianNoise(10)(x)
    for i in range(3):
        x = Convolution2D(32*2**i, (3, 3), activation='relu', 
            kernel_regularizer=regularizers.l2(0.003)
            # bias_regularizer=regularizers.l2(0.01),
            # activity_regularizer=regularizers.l2(0.01))(x)
            )(x)
        x = noise.GaussianNoise(10)(x)
        # x = Dropout(0.5)(x)
        x = Convolution2D(32*2**i, (3, 3), activation='relu', 
            kernel_regularizer=regularizers.l2(0.003)
            # bias_regularizer=regularizers.l2(0.01),
            # activity_regularizer=regularizers.l2(0.01))(x)
            )(x)
        x = noise.GaussianNoise(10)(x)
        # x = Dropout(0.5)(x)
        x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dropout(0.25)(x)
    x = [Dense(setting.CHAR_SET_LEN, activation='softmax', name='c%d'%(i+1))(x) for i in range(4)]
    model = Model(input=input_tensor, output=x)
    return model



from keras import backend as K
from keras.layers.normalization import BatchNormalization

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def CTC():
    rnn_size = 128
    input_tensor = Input((setting.WIDTH, setting.HEIGHT, 1))
    x = input_tensor
    for i in range(2):
        x = Convolution2D(32, (3, 3), activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(1 - 0.1 * i)(x)
        x = Convolution2D(32, (3, 3), activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.9 - 0.1 * i)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = BatchNormalization()(x)

    conv_shape = x.get_shape()
    x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2]*conv_shape[3])))(x)

    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(x)
    gru_1 = BatchNormalization()(gru_1)

    gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(x)
    gru_1b = BatchNormalization()(gru_1b)
    # print(123)
    gru1_merged = add([gru_1, gru_1b])
    gru1_merged = BatchNormalization()(gru1_merged)

    # print(456)
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2 = BatchNormalization()(gru_2)

    gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
    x = concatenate([gru_2, gru_2b])
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x = Dense(setting.CHAR_SET_LEN + 1, kernel_initializer='he_normal', activation='softmax')(x)
    base_model = Model(inputs=input_tensor, outputs=x)

    labels = Input(name='the_labels', shape=[setting.MAX_CAPTCHA], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])
    model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=[loss_out])
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta', metrics=['accuracy'])
    return model, base_model, conv_shape
