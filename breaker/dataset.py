from tqdm import tqdm
from skimage.color import rgb2gray
from skimage.transform import resize
import cv2
import numpy as np

def build(height, width, captcha_length):
    import os
    from random import shuffle
    dataset_dir_X = 'images/all/captcha_dataset_X.npy'
    dataset_dir_Y = 'images/all/captcha_dataset_Y.npy'
    filenames = os.listdir("images/")
    length = len(filenames)
    print(length)
    shuffle(filenames)
    X = list()
    Y = list()
    print("building dataset!")
    for index in tqdm(range(length)):
        filename = filenames[index]
        if (filename.endswith(".jpg") or filename.endswith(".jpeg") or
            filename.endswith(".png") or filename.endswith(".gif")):
            image = resize(cv2.cvtColor(cv2.imread("images/" + filename), cv2.COLOR_BGR2GRAY), (height, width))
            # plt.imshow(image, cmap="gray")
            image = image.astype(np.float32)
            image = np.expand_dims(image, axis=2)
            X.append(image)
            Y.append(filename[0:captcha_length])
    X = np.asarray(X)
    Y = np.asarray(Y)
    np.save(dataset_dir_X, X)
    np.save(dataset_dir_Y, Y)
    print("Done!")


def read(captcha_length, captcha_alphabet):
    X = np.load('images/all/captcha_dataset_X.npy')
    texts = np.load('images/all/captcha_dataset_Y.npy')
    Y = np.zeros((len(texts), captcha_length * len(captcha_alphabet)))
    print("reading and converting!")
    for idx in tqdm(range(len(texts))):
        for j, char in enumerate(texts[idx]):
            Y[idx, j*len(captcha_alphabet) + captcha_alphabet.find(char)] = 1
    print("Done")
    return X, Y
