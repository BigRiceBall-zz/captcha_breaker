from skimage.transform import resize
from keras.models import load_model
import cv2
import numpy as np

# 验证码中的字符
# string.digits + string.ascii_uppercase
NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

CHAR_SET = NUMBER + ALPHABET
CHAR_SET_LEN = len(CHAR_SET)
CHARACTERS = ''.join(CHAR_SET)

MAX_CAPTCHA = 4

# 图像大小
HEIGHT = 36
WIDTH = 150

def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([CHARACTERS[x] for x in y])

def predict(path):
    model = load_model('models/model_1530163862.h5')
    image = resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY), (36, 150))
    _, image = cv2.threshold(image,0.5,1,cv2.THRESH_BINARY) 
    image = np.expand_dims(image, axis=2)
    image = np.expand_dims(image, axis=0)
    predicted_text = decode(model.predict(image))
    return predicted_text


