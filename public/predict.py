from skimage.transform import resize
from keras.models import load_model
from keras import backend as K
import cv2
import numpy as np
import sys

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

def decode(y, CTC=False):
    if not CTC:
        y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([CHARACTERS[x] for x in y])

def predict(path):
    model = load_model('models/model_1530170070.h5')
    # model_1530170070.h5
    # model_1530167168.h5
    image = resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY), (36, 150))
    _, image = cv2.threshold(image,0.5,1,cv2.THRESH_BINARY) 
    image = np.expand_dims(image, axis=2)
    image = np.expand_dims(image, axis=0)
    predicted_text = decode(model.predict(image))
    return predicted_text


def predict_CTC(path):
    model = load_model("models/model_CTC_base_model_1530176456.h5")
    image = resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY), (36, 150))
    _, image = cv2.threshold(image,0.5,1,cv2.THRESH_BINARY) 
    image = np.expand_dims(image, axis=2)
    image = image.transpose(1, 0, 2)
    image = np.expand_dims(image, axis=0)
    predicted_encode_text = model.predict(image)
    predicted_encode_text = predicted_encode_text[:,2:,:]
    out = K.get_value(
        K.ctc_decode(predicted_encode_text, 
        input_length=np.ones(predicted_encode_text.shape[0])*predicted_encode_text.shape[1], )[0][0])[0, :4]
    return decode(out, CTC=True)
    
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--path", dest="path", required=True,
                        help="image path")
    parser.add_argument("-m", "--model", dest="model", default="simple",
                        help="choose 'simple' or 'ctc' model")
    args = parser.parse_args()
    if args.model == "simple":
        print(predict(args.path))
    elif args.model == "ctc":
        print(predict_CTC(args.path))
    else:
        print("not support yet")
