from breaker import model_builder
from breaker import dataset
import tensorflow as tf
import time
import numpy as np

def accuracy(captcha_alphabet_length, y_true, y_pred):
    first_pred = tf.argmax(y_pred[:, 0*captcha_alphabet_length:1*captcha_alphabet_length], axis=1)
    secon_pred = tf.argmax(y_pred[:, 1*captcha_alphabet_length:2*captcha_alphabet_length], axis=1)
    third_pred = tf.argmax(y_pred[:, 2*captcha_alphabet_length:3*captcha_alphabet_length], axis=1)
    fourt_pred = tf.argmax(y_pred[:, 3*captcha_alphabet_length:4*captcha_alphabet_length], axis=1)

    first_true = tf.argmax(y_true[:, 0*captcha_alphabet_length:1*captcha_alphabet_length], axis=1)
    secon_true = tf.argmax(y_true[:, 1*captcha_alphabet_length:2*captcha_alphabet_length], axis=1)
    third_true = tf.argmax(y_true[:, 2*captcha_alphabet_length:3*captcha_alphabet_length], axis=1)
    fourt_true = tf.argmax(y_true[:, 3*captcha_alphabet_length:4*captcha_alphabet_length], axis=1)
    accuracy = tf.constant(0)
#     for idx in range(first_pred.shape[0]):
    first = tf.equal(first_pred, first_true)
    secon = tf.equal(secon_pred, secon_true)
    third = tf.equal(third_pred, third_true)
    fourt = tf.equal(fourt_pred, fourt_true)
    f_s = tf.logical_and(first, secon)
    t_f = tf.logical_and(third, fourt)
    f_s_t_f = tf.logical_and(f_s, t_f)
    accuracy = tf.reduce_mean(tf.cast(f_s_t_f, dtype=np.float16))
    return accuracy

def acc(captcha_alphabet_length):
    def acc_for_captch(x, y):
        return accuracy(captcha_alphabet_length, x, y)
    return acc_for_captch

def train(height, width, captcha_length, captcha_alphabet, batch_size=128, epochs=10):
    now = str(int(time.time()))
    model = model_builder.build(height, width, captcha_length, len(captcha_alphabet))
    model.compile(loss='categorical_crossentropy',
                optimizer='adadelta',
                metrics=[acc(len(captcha_alphabet))])
    X, Y = dataset.read(captcha_length, captcha_alphabet)
    model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    model.compile(loss='categorical_crossentropy',
            optimizer='adadelta')
    model.save("model/model_" + now + ".h5")

