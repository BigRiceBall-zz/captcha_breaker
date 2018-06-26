
from captcha.image import ImageCaptcha
from captcha_breaker import setting
import matplotlib.pyplot as plt
import numpy as np
import random

def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([setting.CHARACTERS[x] for x in y])

def generator_4_multiple_types(batch_size=32):
    X = np.zeros((batch_size, setting.HEIGHT, setting.WIDTH, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, setting.CHAR_SET_LEN), dtype=np.uint8) for i in range(setting.MAX_CAPTCHA)]
    generator = ImageCaptcha(width=setting.WIDTH, height=setting.HEIGHT)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(setting.CHARACTERS) for j in range(4)])
            X[i] = generator.generate_image(random_str)
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, setting.CHARACTERS.find(ch)] = 1
        yield X, y