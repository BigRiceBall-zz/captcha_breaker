import pytesseract
from captcha.image import ImageCaptcha
import numpy as  np
import matplotlib.pyplot as  plt
from  PIL import Image
import random
import cv2 as cv2
import os
from tqdm import tqdm

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
Alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

char_set = number + alphabet + Alphabet

##图片高
IMAGE_HEIGHT = 60
##图片宽
IMAGE_WIDTH = 160
##验证码长度
MAX_CAPTCHA = 4
##验证码选择空间
CHAR_SET_LEN = len(char_set)
##提前定义变量空间

##生成n位验证码字符 这里n=4
def random_captcha_text(char_set=char_set, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


##使用ImageCaptcha库生成验证码
def gen_captcha_text_and_image():
    image = ImageCaptcha()
    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)
    captcha = image.generate(captcha_text)
    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image


def parse_image_to_text(path): 
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.fastNlMeansDenoising(image,None,50,10,7)
    _, image = cv2.threshold(image,100,255,cv2.THRESH_BINARY) 
    # plt.imshow(image, cmap="gray")
    text = pytesseract.image_to_string(image)
    for char in text:
        if char not in char_set:
            text = text.replace(char, "")
    # print(text)
    # plt.show()
    return text


def parse_images_to_text():
    filenames = os.listdir("images/jd/captcha/jd/")
    length = len(filenames)
    count = 0
    print(length)
    for filename in tqdm(filenames):
        # print(filename)
        if (filename.endswith(".jpg") or filename.endswith(".jpeg") or
            filename.endswith(".png") or filename.endswith(".gif")):
            predicted_text = parse_image_to_text("images/jd/captcha/jd/" + filename)
            if filename[0:4] == predicted_text:
                count+=1
                # print(count)
            print("accuracy: " + str(count/length))
    print("accuracy: " + str(count/length))
