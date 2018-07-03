
from captcha.image import ImageCaptcha
from captcha_breaker import setting
# import matplotlib.pyplot as plt
import numpy as np
import random
from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np
import skimage.io as io
import cv2 
import threading
from skimage import util

def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([setting.CHARACTERS[x] for x in y])


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
    # print(text)
    p = np.random.randint(10, size=2) - 4
    interval = np.random.randint(6, size=1) + 14
    height = 37 + p[0]
    for index, char in enumerate(text):
        image = cv2.putText(image, char, (height + index*interval, 28 + p[1]), font, 1, (0,0,0), 2, cv2.LINE_AA)
    # plt.imshow(image)
    # print(image)
    # plt.show()
    return image



# from captcha_breaker import image_generactor
# image_generactor.generate_type_1_captcha(model_image, "XA8B")

def generate_different_type(text, model_image, generator, true_images_labels, nb_type=6):
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
        return np.expand_dims(rgb2gray(image), axis=2), text
    elif type_range[1] <= p and p < type_range[2]:
        image = resize(cv2.cvtColor(generate_type_1_captcha(model_image, text), cv2.COLOR_BGR2GRAY), (setting.HEIGHT, setting.WIDTH))
        # plt.imshow(image, cmap="gray")
        # plt.show()
        return np.expand_dims(image, axis=2), text
    elif type_range[2] <= p and p < type_range[3]:
        image = resize(cv2.cvtColor(generate_type_1_captcha(model_image, text), cv2.COLOR_BGR2GRAY), (setting.HEIGHT, setting.WIDTH))
        pp = np.random.uniform(-0.005, 0.005)
        _, image = cv2.threshold(image,0.4 + pp,1,cv2.THRESH_BINARY) 
        # plt.imshow(image, cmap="gray")
        # plt.show()
        return np.expand_dims(image, axis=2), text
    elif type_range[3] <= p and p < type_range[4]:
        # np.set_printoptions(threshold=np.nan)
        image = resize(cv2.cvtColor(generate_type_1_captcha(model_image, text), cv2.COLOR_BGR2GRAY), (setting.HEIGHT, setting.WIDTH))
        pp = np.random.uniform(-0.005, 0.005)
        _, image = cv2.threshold(image,0.4 + pp,1,cv2.THRESH_BINARY)
        image = util.invert(image)
        # print(image)
        # plt.imshow(image, cmap="gray")
        # plt.show()
        return np.expand_dims(image, axis=2), text
    elif type_range[4] <= p and p < type_range[5]:
        # np.set_printoptions(threshold=np.nan)
        image = resize(cv2.cvtColor(generate_type_1_captcha(model_image, text), cv2.COLOR_BGR2GRAY), (setting.HEIGHT, setting.WIDTH))
        pp = np.random.uniform(-0.005, 0.005)
        _, image = cv2.threshold(image,0.4 + pp,1,cv2.THRESH_BINARY)
        image = util.invert(image)
        pp = np.random.uniform(0, 0.11)
        image = add_saltnpeppar_noise(image, pp)

        # print(image)
        # plt.imshow(image, cmap="gray")
        # plt.show()
        return np.expand_dims(image, axis=2), text
    elif type_range[5] <= p and p < type_range[6]:
        image = resize(cv2.cvtColor(generate_type_1_captcha(model_image, text), cv2.COLOR_BGR2GRAY), (setting.HEIGHT, setting.WIDTH))
        pp = np.random.uniform(-0.005, 0.005)
        _, image = cv2.threshold(image,0.4 + pp,1,cv2.THRESH_BINARY) 
        pp = np.random.uniform(0, 0.11)
        image = add_saltnpeppar_noise(image, pp)

        # print(image)
        # plt.imshow(image, cmap="gray")
        # plt.show()
        return np.expand_dims(image, axis=2), text

    elif type_range[6] <= p and p < type_range[7]:
        p = np.random.randint(0, true_images_labels[0].shape[0])
        # print(p)
        pp = np.random.uniform(0.0001, 0.12)
        image = true_images_labels[0][p]

        image = add_gaussian_noise(image, pp, 0.1+pp*pp)

        # print(image)
        # plt.imshow(image.reshape(36, 150), cmap="gray")
        # print(true_images_labels[1][p].decode("ascii"))
        # plt.show()
        return image, true_images_labels[1][p].decode("ascii")
    elif type_range[7] <= p and p < type_range[8]:
        p = np.random.randint(0, true_images_labels[0].shape[0])
        # print(p)
        pp = np.random.uniform(-0.005, 0.005)
        image = true_images_labels[0][p].reshape(setting.HEIGHT, setting.WIDTH)
        _, image = cv2.threshold(image,0.5 + pp,1,cv2.THRESH_BINARY) 

        image = add_saltnpeppar_noise(image, 0.1 + pp)
        
        # print(image)
        # plt.imshow(image, cmap="gray")
        # print(true_images_labels[1][p].decode("ascii"))
        # plt.show()
        return np.expand_dims(image, axis=2), true_images_labels[1][p].decode("ascii")
    # elif 

def add_saltnpeppar_noise(im,prop):
    N = int(np.round(np.prod(im.shape)*prop))
    index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N],im.shape)
    im2 = np.copy(im)
    im2[index] = 1-im2[index]
    return im2
    
def add_gaussian_noise(im,prop,varSigma):
    N = int(np.round(np.prod(im.shape)*prop))
    index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N],im.shape)
    e = varSigma*np.random.randn(np.prod(im.shape)).reshape(im.shape)
    im2 = np.copy(im)
    im2[index] += e[index]
    return im2
# from captcha_breaker import image_generactor
# import cv2
# image1 = cv2.imread("./images/models.jpeg")
# image_generactor.generate_type_1_captcha(image1, "ABCD")





import threading

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def generator_4_multiple_types(batch_size=32, nb_type=6):
    X = np.zeros((batch_size, setting.HEIGHT, setting.WIDTH, 1), dtype=np.float32)
    y = [np.zeros((batch_size, setting.CHAR_SET_LEN), dtype=np.uint8) for i in range(setting.MAX_CAPTCHA)]
    generator = ImageCaptcha(width=170, height=80)
    model_image = cv2.imread("./images/models.jpeg")
    h5f = h5py.File('images/jd/captcha/origin_jd_captcha_train.h5', 'r')
    images = h5f["X"].value
    texts = h5f["Y"].value

    # print(true_images)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(setting.CHARACTERS) for j in range(4)])
            X[i], text = generate_different_type(random_str, model_image, generator, 
                                                (images, texts), nb_type)
            # print(X[i])
            for j, ch in enumerate(text):
                y[j][i, :] = 0
                y[j][i, setting.CHARACTERS.find(ch)] = 1
        yield X, y

@threadsafe_generator
def generate_true_test_captcha_simple(batch_size=128):
    X = np.zeros((batch_size, setting.HEIGHT, setting.WIDTH, 1), dtype=np.float32)
    y = [np.zeros((batch_size, setting.CHAR_SET_LEN), dtype=np.uint8) for i in range(setting.MAX_CAPTCHA)]
    h5f = h5py.File('images/jd/captcha/origin_jd_captcha_test.h5', 'r')
    images = h5f["X"].value
    texts = h5f["Y"].value
    length = len(images)
    while True:
        for i in range(batch_size):
            p = np.random.randint(0, length)
            X[i] = images[p]
            for j, ch in enumerate(texts[p].decode("ascii")):
                y[j][i, :] = 0
                y[j][i, setting.CHARACTERS.find(ch)] = 1
        yield X, y    


@threadsafe_generator
def generator_4_multiple_types_CTC(conv_shape, batch_size=128, nb_type=6):
    X = np.zeros((batch_size, setting.WIDTH, setting.HEIGHT, 1), dtype=np.float32)
    y = np.zeros((batch_size, setting.MAX_CAPTCHA), dtype=np.uint8)
    generator = ImageCaptcha(width=170, height=80)
    model_image = cv2.imread("./images/models.jpeg")
    h5f = h5py.File('images/jd/captcha/origin_jd_captcha_train.h5', 'r')
    images = h5f["X"].value
    texts = h5f["Y"].value
    # print(conv_shape)
    while True:
        # texts = list() 
        for i in range(batch_size):
            random_str = ''.join([random.choice(setting.CHARACTERS) for j in range(4)])
            # random_str = "4E4U"
            image, text = generate_different_type(random_str, model_image, generator, 
                                                (images, texts), nb_type)
            y[i] = [setting.CHARACTERS.find(x) for x in text]
            X[i] = image.transpose(1, 0, 2)
            # texts.append(text)
            # print(y[i])
            # print(len(np.ones(batch_size)*setting.MAX_CAPTCHA))
            # break
        yield [X, y, np.ones(batch_size)*int(conv_shape[1]-2), np.ones(batch_size)*setting.MAX_CAPTCHA], np.ones(batch_size)
        # yield texts
@threadsafe_generator
def generate_true_test_captcha(conv_shape, batch_size=128):
    X = np.zeros((batch_size, setting.WIDTH, setting.HEIGHT, 1), dtype=np.float32)
    y = np.zeros((batch_size, setting.MAX_CAPTCHA), dtype=np.uint8)
    h5f = h5py.File('images/jd/captcha/origin_jd_captcha_test.h5', 'r')
    images = h5f["X"].value
    texts = h5f["Y"].value
    length = len(images)
    while True:
        for i in range(batch_size):
            p = np.random.randint(0, length)
            # print(texts[p])
            y[i] = [setting.CHARACTERS.find(x) for x in texts[p].decode("ascii")]
            X[i] = images[p].transpose(1, 0, 2)
            # print(y[i])
            # print(len(np.ones(batch_size)*setting.MAX_CAPTCHA))
            # break
        yield [X, y, np.ones(batch_size)*int(conv_shape[1]-2), np.ones(batch_size)*setting.MAX_CAPTCHA], np.ones(batch_size)



import h5py
from tqdm import tqdm
def true_image2H5():
    import os
    from random import shuffle
    h5file = h5py.File('images/jd/captcha/origin_jd_captcha_train.h5', 'w')
    filenames = os.listdir("images/jd/captcha/jd/")
    length = len(filenames)
    print(length)
    shuffle(filenames)
    filenames_train_length = int(length * 0.75)
    X = list()
    Y = list()
    for index in tqdm(range(filenames_train_length)):
        filename = filenames[index]
        if (filename.endswith(".jpg") or filename.endswith(".jpeg") or
            filename.endswith(".png") or filename.endswith(".gif")):
            image = resize(cv2.cvtColor(cv2.imread("images/jd/captcha/jd/" + filename), cv2.COLOR_BGR2GRAY), (36, 150))
            plt.imshow(image, cmap="gray")
            image = image.astype(np.float32)
            image = np.expand_dims(image, axis=2)
            X.append(image)
            Y.append(filename[0:4].encode("ascii", "error"))
    X = np.asarray(X)
    print(len(X))
    h5file.create_dataset("X", (X.shape), data=X)
    h5file.create_dataset("Y", data=Y, dtype="S10")
    h5file.close()
    
    h5file = h5py.File('images/jd/captcha/origin_jd_captcha_test.h5', 'w')
    X = list()
    Y = list()
    for index in tqdm(range(filenames_train_length, length)):
        filename = filenames[index]
        if (filename.endswith(".jpg") or filename.endswith(".jpeg") or
            filename.endswith(".png") or filename.endswith(".gif")):
            image = resize(cv2.cvtColor(cv2.imread("images/jd/captcha/jd/" + filename), cv2.COLOR_BGR2GRAY), (36, 150))
            plt.imshow(image, cmap="gray")
            image = image.astype(np.float32)
            image = np.expand_dims(image, axis=2)
            X.append(image)
            Y.append(filename[0:4].encode("ascii", "error"))
    X = np.asarray(X)
    print(len(X))
    h5file.create_dataset("X", (X.shape), data=X)
    h5file.create_dataset("Y", data=Y, dtype="S10")
    h5file.close()



# from captcha_breaker import image_generactor
# image_generactor.true_image2H5()