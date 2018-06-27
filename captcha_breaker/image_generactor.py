
from captcha.image import ImageCaptcha
from captcha_breaker import setting
# import matplotlib.pyplot as plt
import numpy as np
import random
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.transform import resize
import numpy as np
import skimage.io as io
import cv2 

def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([setting.CHARACTERS[x] for x in y])

def generator_4_multiple_types(batch_size=32, nb_type=1):
    X = np.zeros((batch_size, setting.HEIGHT, setting.WIDTH, 1), dtype=np.float32)
    y = [np.zeros((batch_size, setting.CHAR_SET_LEN), dtype=np.uint8) for i in range(setting.MAX_CAPTCHA)]
    generator = ImageCaptcha(width=170, height=80)
    model_image = cv2.imread("./images/models.jpeg")
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(setting.CHARACTERS) for j in range(4)])
            X[i] = generate_different_type(random_str, model_image, generator, 3)
            # print(X[i])
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, setting.CHARACTERS.find(ch)] = 1
        yield X, y

# def encode(text):
def _generate_type_1_model_image():
    image1 = cv2.imread("./images/model1.jpg")
    image2 = cv2.imread("./images/model2.jpg")
    model_image = cv2.hconcat([image1, image2, image1, image2, image1, image2, image1, image2, image1])
    cv2.imwrite("./images/models.jpeg", model_image)
    plt.imshow(model_image)
    plt.show()

def generate_type_1_captcha(model_image, text):
    image = model_image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = cv2.putText(image, text, (35,28), font, 1, (0,0,0), 2, cv2.LINE_AA)
    # plt.imshow(model_image)
    # plt.show()
    return image

def generate_different_type(text, model_image, generator, nb_type=1):
    p = np.random.uniform(0,1)
    type_range = np.linspace(0, 1, nb_type + 1, endpoint=True)
    if 0 <= p and p < type_range[1]:
        image = resize(np.asarray(generator.generate_image(text)), (setting.HEIGHT, setting.WIDTH))
        # print(rgb2gray(image).shape)
        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.imshow('image',image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # plt.imshow(rgb2gray(image), cmap="gray")
        # plt.show()
        return np.expand_dims(rgb2gray(image), axis=2)
    elif type_range[1] <= p and p < type_range[2]:
        image = resize(cv2.cvtColor(generate_type_1_captcha(model_image, text), cv2.COLOR_BGR2GRAY), (setting.HEIGHT, setting.WIDTH))
        # plt.imshow(image, cmap="gray")
        # plt.show()
        return np.expand_dims(image, axis=2)
    elif type_range[2] <= p and p < type_range[3]:
        image = resize(cv2.cvtColor(generate_type_1_captcha(model_image, text), cv2.COLOR_BGR2GRAY), (setting.HEIGHT, setting.WIDTH))
        pp = np.random.uniform(-0.005, 0.005)
        _, image = cv2.threshold(image,0.5 + pp,1,cv2.THRESH_BINARY) 
        # plt.imshow(image, cmap="gray")
        # plt.show()
        return np.expand_dims(image, axis=2)

# from captcha_breaker import image_generactor
# import cv2
# image1 = cv2.imread("./images/models.jpeg")
# image_generactor.generate_type_1_captcha(image1, "ABCD")