from breaker import trainer
from breaker import dataset
from argparse import ArgumentParser
from keras.models import load_model
import cv2
from skimage.transform import resize
import numpy as np

def train(height, width, captcha_length, captcha_alphabet, batch_size=128, epochs=10):
    trainer.train(height, width, captcha_length, captcha_alphabet, batch_size=batch_size, epochs=epochs)

def build_dataset(height, width, captcha_length):
    dataset.build(height, width, captcha_length)

def predict(height, width, captcha_length, captcha_alphabet, path, model_path):
    model = load_model(model_path)
    image = resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY), (height, width))
    image = np.expand_dims(image, axis=2)
    image = np.expand_dims(image, axis=0)
    data = model.predict(image)
    predicted_text = []
    for idx in range(captcha_length):
        predicted_text.append(captcha_alphabet[np.argmax(np.array(data[:, idx*len(captcha_alphabet):(idx+1)*len(captcha_alphabet)]))])
    predicted_text = "".join(predicted_text)
    return predicted_text

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-ht", "--height", dest="height", required=False, default=36,
                        help="image height")
    parser.add_argument("-wh", "--width", dest="width", required=False, default=150,
                        help="image width")
    parser.add_argument("-cl", "--captchalength", dest="captcha_length", required=False, default=4,
                        help="the number of character in image")
    parser.add_argument("-cal", "--alphabetlength", dest="captcha_alphabet", required=False, default="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                        help="the number of character in image")
    parser.add_argument("-b", "--batchsize", dest="batch_size", required=False, default=128,
                        help="batch size")
    parser.add_argument("-m", "--mode", dest="mode", default="train",
                        help="train or predict or dataset_builder")
    parser.add_argument("-ep", "--epochs", dest="epochs", default=50,
                        help="the number of iteration")
    parser.add_argument("-mp", "--model", dest="model_path", default="model/model_1533799477.h5",
                        help="model path")
    parser.add_argument("-imp", "--image", dest="image_path", required=False,
                        help="image path")
    args = parser.parse_args()

    if args.mode == "train":
        train(args.height, args.width, args.captcha_length, args.captcha_alphabet, args.batch_size, args.epochs)
    elif args.mode == "dataset_builder":
        build_dataset(args.height, args.width, args.captcha_length)
    elif args.mode == "predict":
        if args.image_path == None:
            parser.error("The 'predict mode' requires the -imp (image path)")
        print(predict(args.height, args.width, args.captcha_length, args.captcha_alphabet, args.image_path, args.model_path))


