
import captcha_breaker.model as model_builder
from captcha_breaker import image_generactor
import time
from keras import backend as K
import numpy as np
from tqdm import tqdm

def train(batch_size=32, nb_type=3):
    now = str(int(time.time()))
    model = model_builder.simple()
    model.compile(loss='categorical_crossentropy',
                optimizer='adadelta',
                metrics=['accuracy'])
                
    model.fit_generator(image_generactor.generator_4_multiple_types(batch_size=batch_size, nb_type=nb_type), 
                        samples_per_epoch=10240, nb_epoch=5,
                        nb_worker=28,
                        validation_data=image_generactor.generator_4_multiple_types(batch_size=batch_size, nb_type=nb_type), nb_val_samples=1280)
    model.save("models/model_" + now + ".h5")

from keras.callbacks import *

def evaluate_training(base_model, conv_shape, batch_num=128, nb_type=6):
    batch_acc = 0
    generator = image_generactor.generator_4_multiple_types_CTC(conv_shape, batch_size=batch_num, nb_type=nb_type)
    for i in tqdm(range(batch_num)):
        # print(i)
        [X_test, y_test, _, _], _  = next(generator)
        y_pred = base_model.predict(X_test)
        shape = y_pred[:,2:,:].shape
        out = K.get_value(K.ctc_decode(y_pred[:,2:,:], input_length=np.ones(shape[0])*shape[1])[0][0])[:, :4]
        if out.shape[1] == 4:
            batch_acc += ((y_test == out).sum(axis=1) == 4).mean()
    return batch_acc / batch_num

def evaluate_testing(base_model, generator, conv_shape, batch_size=128):
    batch_acc = 0
    print(batch_size)
    for i in tqdm(range(batch_size)):
        [X_test, y_test, _, _], _  = next(generator)
        y_pred = base_model.predict(X_test)
        shape = y_pred[:,2:,:].shape
        out = K.get_value(K.ctc_decode(y_pred[:,2:,:], input_length=np.ones(shape[0])*shape[1])[0][0])[:, :4]
        if out.shape[1] == 4:
            batch_acc += ((y_test == out).sum(axis=1) == 4).mean()
    return batch_acc / batch_size
    
# def evaluate_training(base_model, batch_num=10):
#     batch_acc = 0
#     generator = gen(128)
#     for i in range(batch_num):
#         [X_test, y_test, _, _], _  = next(generator)
#         y_pred = base_model.predict(X_test)
#         shape = y_pred[:,2:,:].shape
#         out = K.get_value(K.ctc_decode(y_pred[:,2:,:], input_length=np.ones(shape[0])*shape[1])[0][0])[:, :4]
#         if out.shape[1] == 4:
#             batch_acc += ((y_test == out).sum(axis=1) == 4).mean()
#     return batch_acc / batch_num

class Evaluate(Callback):
    def __init__(self, base_model, conv_shape, batch_size=128, nb_type=6):
        self.accs = []
        self._base_model = base_model
        self._conv_shape = conv_shape
        self._batch_size = batch_size
        self._nb_type = 6
        self._test_generator = image_generactor.generate_true_test_captcha(conv_shape, batch_size=1)

    
    def on_epoch_end(self, epoch, logs=None):
        print("calculating accuracy")
        acc = evaluate_training(self._base_model, self._conv_shape, self._batch_size, self._nb_type)*100
        acc_test = evaluate_testing(self._base_model, self._test_generator, self._conv_shape, self._batch_size)*100
        self.accs.append(acc)
        self.accs.append(acc_test)
        print()
        print(str(epoch) + ' training acc: %f%%'%acc)
        print(str(epoch) + 'testing acc: %f%%'%acc_test)

def train_CTC(batch_size=32, nb_type=3):
    now = str(int(time.time()))
    model, base_model, conv_shape = model_builder.CTC()

    evaluator = Evaluate(base_model, conv_shape)                
    # model.fit_generator(image_generactor.generator_4_multiple_types_CTC(conv_shape, batch_size=batch_size, nb_type=nb_type), 
    #                     samples_per_epoch=51200, nb_epoch=4,
    #                     callbacks=[EarlyStopping(patience=10), evaluator],
    #                     nb_worker=28,
    #                     validation_data=
    #                     image_generactor.generator_4_multiple_types_CTC
    #                     (conv_shape, batch_size=batch_size, nb_type=nb_type), nb_val_samples=1280)
    model.fit_generator(image_generactor.generator_4_multiple_types_CTC(conv_shape, batch_size=batch_size, nb_type=nb_type), 
                        samples_per_epoch=1280, nb_epoch=100,
                        callbacks=[evaluator],
                        nb_worker=28,
                        validation_data=
                        image_generactor.generate_true_test_captcha
                        (conv_shape, batch_size=batch_size), nb_val_samples=128)

    model.save("models/model_CTC_" + now + ".h5")
    base_model.save("models/model_CTC_base_model_" + now + ".h5")


def continue_2_train_CTC(batch_size=32, nb_type=3):
    now = str(int(time.time()))
    model, base_model, conv_shape = model_builder.CTC()
    model = load_model('models/model_CTC_1530353300.h5', custom_objects = {'<lambda>': lambda y_true, y_pred: y_pred})

    evaluator = Evaluate(base_model, conv_shape)                
    # model.fit_generator(image_generactor.generator_4_multiple_types_CTC(conv_shape, batch_size=batch_size, nb_type=nb_type), 
    #                     samples_per_epoch=1280, nb_epoch=40,
    #                     callbacks=[EarlyStopping(patience=10), evaluator],
    #                     nb_worker=28,
    #                     validation_data=
    #                     image_generactor.generator_4_multiple_types_CTC
    #                     (conv_shape, batch_size=batch_size, nb_type=nb_type), nb_val_samples=10)
    model.fit_generator(image_generactor.generator_4_multiple_types_CTC(conv_shape, batch_size=batch_size, nb_type=nb_type), 
                        samples_per_epoch=1280, nb_epoch=40,
                        callbacks=[EarlyStopping(patience=10), evaluator],
                        nb_worker=28,
                        validation_data=
                        image_generactor.generate_true_test_captcha
                        (conv_shape, batch_size=batch_size), nb_val_samples=1280)

    model.save("models/model_CTC_" + now + ".h5")
    base_model.save("models/model_CTC_base_model_" + now + ".h5")

def continue_2_train(batch_size=32, nb_type=3):
    from keras.models import load_model
    now = str(int(time.time()))
    model = load_model('models/model_1530167168.h5')
    model.fit_generator(image_generactor.generator_4_multiple_types(batch_size=batch_size, nb_type=nb_type), 
                        samples_per_epoch=8192, nb_epoch=1,
                        nb_worker=28,
                        validation_data=image_generactor.generator_4_multiple_types(batch_size=batch_size, nb_type=nb_type), nb_val_samples=1280)
    model.save("models/model_" + now + ".h5")

def test():
    import matplotlib.pyplot as plt
    from keras.models import load_model
    model = load_model('models/model_1530170070.h5')
    generator = image_generactor.generator_4_multiple_types(batch_size=1, nb_type=5)
    X, y = next(generator)
    ture_y = image_generactor.decode(y)
    predicted_text = image_generactor.decode(model.predict(X))
    X = X[0].reshape((36, 150))
    plt.imshow(X, cmap="gray")
    # print(y)
    print("true: " + ture_y + " predict: " + predicted_text)
    plt.show()

def test_JD():
    import cv2 
    from skimage.transform import resize
    import matplotlib.pyplot as plt
    from keras.models import load_model
    import numpy as np
    import os
    import h5py
    from tqdm import tqdm
    model = load_model('models/model_1530163862.h5')
    now = time.time()
    h5f = h5py.File('images/jd/captcha/origin_jd_captcha_test.h5', 'r')
    images = h5f["X"].value
    texts = h5f["Y"].value
    count = 0
    length = len(images)
    print(images.shape)
    for index, image in enumerate(tqdm(images)):
        # print(image.shape)
        _, image = cv2.threshold(image,0.4,1,cv2.THRESH_BINARY)
        # plt.imshow(image)
        image = np.expand_dims(image, axis=2) 
        image = np.expand_dims(image, axis=0)
        predicted_text = image_generactor.decode(model.predict(image))
        text = texts[index].decode("ascii")
        # print(predicted_text)
        # plt.show()
        if text == predicted_text:
            count += 1
    print("elapsed: " + str(time.time() - now))
    print("accuracy: " + str(count/length))




def predict():
    import cv2 
    from skimage.transform import resize
    import matplotlib.pyplot as plt
    from keras.models import load_model
    import numpy as np
    model = load_model('models/model_1530170070.h5')
    now = time.time()

    image = resize(cv2.cvtColor(cv2.imread("./images/image3.jpeg"), cv2.COLOR_BGR2GRAY), (36, 150))
    _, image = cv2.threshold(image,0.5,1,cv2.THRESH_BINARY) 

    image1 = np.expand_dims(image, axis=2)
    image1 = np.expand_dims(image1, axis=0)
    predicted_text = image_generactor.decode(model.predict(image1))
    print("elapsed: " + str(time.time() - now))

    plt.imshow(image, cmap="gray")
    print(" predict: " + predicted_text)
    plt.show()

def predict_jd():
    import cv2 
    from skimage.transform import resize
    import matplotlib.pyplot as plt
    from keras.models import load_model
    import numpy as np
    import os
    from tqdm import tqdm
    model = load_model('models/model_1530170070.h5')
    filenames = os.listdir("images/jd/captcha/jd/")
    length = len(filenames)
    count = 0
    print(length)
    for filename in tqdm(filenames):
        # print(filename)
        if (filename.endswith(".jpg") or filename.endswith(".jpeg") or
            filename.endswith(".png") or filename.endswith(".gif")):
            image = resize(cv2.cvtColor(cv2.imread("images/jd/captcha/jd/" + filename), cv2.COLOR_BGR2GRAY), (36, 150))
            _, image = cv2.threshold(image,0.5,1,cv2.THRESH_BINARY) 

            image1 = np.expand_dims(image, axis=2)
            image1 = np.expand_dims(image1, axis=0)
            # print(image)
            predicted_text = image_generactor.decode(model.predict(image1))
            # plt.imshow(image, cmap="gray")
            # print(y)
            # print("true: " + filename[0:4] + " predict: " + predicted_text)
            if filename[0:4] == predicted_text:
                count+=1
                # print(count)
    print("accuracy: " + str(count/length))
        # plt.show()

from keras.models import *
from keras.layers import *
def test2():
    import cv2 
    from skimage.transform import resize
    import matplotlib.pyplot as plt
    from keras.models import load_model
    import numpy as np

    model = load_model('models/model_1530071864.h5')
    model.get_layer('conv2d_1').kernel_regularizer = regularizers.l2(0.01) 
    model.get_layer('conv2d_1').bias_regularizer = regularizers.l2(0.01)
    model.get_layer('conv2d_1').activity_regularizer = regularizers.l2(0.01) 
    model.get_layer('conv2d_2').kernel_regularizer = regularizers.l2(0.01) 
    model.get_layer('conv2d_2').bias_regularizer = regularizers.l2(0.01)
    model.get_layer('conv2d_2').activity_regularizer = regularizers.l2(0.01) 
    model.get_layer('conv2d_3').kernel_regularizer = regularizers.l2(0.01) 
    model.get_layer('conv2d_3').bias_regularizer = regularizers.l2(0.01)
    model.get_layer('conv2d_3').activity_regularizer = regularizers.l2(0.01) 
    model.get_layer('conv2d_4').kernel_regularizer = regularizers.l2(0.01) 
    model.get_layer('conv2d_4').bias_regularizer = regularizers.l2(0.01)
    model.get_layer('conv2d_4').activity_regularizer = regularizers.l2(0.01) 
    model.get_layer('conv2d_5').kernel_regularizer = regularizers.l2(0.01) 
    model.get_layer('conv2d_5').bias_regularizer = regularizers.l2(0.01)
    model.get_layer('conv2d_5').activity_regularizer = regularizers.l2(0.01) 
    model.get_layer('conv2d_6').kernel_regularizer = regularizers.l2(0.01) 
    model.get_layer('conv2d_6').bias_regularizer = regularizers.l2(0.01)
    model.get_layer('conv2d_6').activity_regularizer = regularizers.l2(0.01) 
    model.compile(loss='categorical_crossentropy',
                optimizer='adadelta',
                metrics=['accuracy'])
    print(model.get_config()['layers'])
    model.save('models/model_1530090310_1.h5')

    # layers = [layer for layer in model.layers]
    # new_conv = Dropout(0.5)(layers[1].output)

    # x = new_conv
    # for i in range(3, len(layers) - 10):
    #     print(i)
    #     x = layers[i](x)
    # # for i in range(0, len(layers)):
    # #     x = layers[i](x)
    # # print(model.layers)
    # model = Model(input=layers[0].input, output=x)

    model.summary()

def test_CTC():
    import cv2 
    from skimage.transform import resize
    import matplotlib.pyplot as plt
    from keras.models import load_model
    import numpy as np
    from keras.models import model_from_json
    # model = load_model("models/model_CTC_1530238160.h5", custom_objects = {'<lambda>': lambda y_true, y_pred: y_pred})
    # print(models.layers)
    # print(model.to_json())
    # from keras.models import model_from_json
    # json_model = model.to_json()

    _, _, conv_shape = model_builder.CTC()

    model = load_model("models/model_CTC_base_model_1530353300.h5")

    # model.summary()
    # generator = image_generactor.generator_4_multiple_types(batch_size=1, nb_type=5)
    NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    CHAR_SET = NUMBER + ALPHABET
    CHAR_SET_LEN = len(CHAR_SET)
    CHARACTERS = ''.join(CHAR_SET)
    # characters2 = characters + ' '
    [X_test, y_test, _, _], _  = next(image_generactor.generator_4_multiple_types_CTC(conv_shape, batch_size=1, nb_type=8))
    now = time.time()

    # image = resize(cv2.cvtColor(cv2.imread("./images/jd/captcha/jd/unknown/tmp153008438483711886.jpg"), cv2.COLOR_BGR2GRAY), (36, 150))
    # _, image = cv2.threshold(image,0.4,1,cv2.THRESH_BINARY) 

    # image = np.expand_dims(image, axis=2)
    # image = image.transpose(1, 0, 2)
    # X_test = np.expand_dims(image, axis=0)

    y_pred = model.predict(X_test)
    y_pred = y_pred[:,2:,:]
    out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0])*y_pred.shape[1], top_paths=1)[0][0])[:, :4]
    print(out)
    out = ''.join([CHARACTERS[x] for x in out[0]])   
    print("elapsed: " + str(time.time() - now))

    # y_true = ''.join([CHARACTERS[x] for x in y_test[0]])
    print(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0])*y_pred.shape[1], top_paths=2))
    print(" predict: " + out)
    # print(X_test[0].transpose(1, 0, 2).shape)
    import matplotlib.pyplot as plt
    # plt.imshow(X_test, cmap="gray")
    plt.imshow(X_test[0].transpose(1, 0, 2)[:, :, 0], cmap="gray")
    # plt.title('pred:' + str(out) + '\ntrue: ' + str(y_true))
    plt.show()

    # argmax = np.argmax(y_pred, axis=2)[0]
    # list(zip(argmax, ''.join([characters2[x] for x in argmax])))


def predict_JD_CTC():
    import cv2 
    from skimage.transform import resize
    import matplotlib.pyplot as plt
    from keras.models import load_model
    import numpy as np

    model = load_model("models/model_CTC_1530176456.h5")

    NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    CHAR_SET = NUMBER + ALPHABET
    CHAR_SET_LEN = len(CHAR_SET)
    CHARACTERS = ''.join(CHAR_SET)

    from tqdm import tqdm
    filenames = os.listdir("images/jd/captcha/jd/")
    length = len(filenames)
    count = 0
    print(length)
    for filename in tqdm(filenames):
        # print(filename)
        if (filename.endswith(".jpg") or filename.endswith(".jpeg") or
            filename.endswith(".png") or filename.endswith(".gif")):
            image = resize(cv2.cvtColor(cv2.imread("images/jd/captcha/jd/" + filename), cv2.COLOR_BGR2GRAY), (36, 150))
            _, image = cv2.threshold(image,0.5,1,cv2.THRESH_BINARY) 

            image = np.expand_dims(image, axis=2)
            # print(image.shape)
            image = image.transpose(1, 0, 2)
            X_test = np.expand_dims(image, axis=0)
            
            
            # print(X_test.shape)
            y_pred = base_model.predict(X_test)
            y_pred = y_pred[:,2:,:]
            out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0])*y_pred.shape[1], )[0][0])[:, :4]
            out = ''.join([CHARACTERS[x] for x in out[0]])

            if filename[0:4] == out:
                count+=1
                # print(count)
    print("accuracy: " + str(count/length))
        # plt.show()

def CTC_model_2_base_model():
    model = load_model("models/model_CTC_1530176456.h5", custom_objects = {'<lambda>': lambda y_true, y_pred: y_pred})
    model.summary()
    model = Model(model.inputs, model.layers[-5].output)  # assuming you want the 3rd layer from the last
    model.summary()
    model.save("models/model_CTC_base_model_1530176456.h5")

def test_JD_CTC():
    import cv2 
    from skimage.transform import resize
    import matplotlib.pyplot as plt
    from keras.models import load_model
    import numpy as np
    import os
    import h5py
    from tqdm import tqdm

    model = load_model("models/model_CTC_base_model_1530503600.h5")

    # print(model.get_config())
    NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    CHAR_SET = NUMBER + ALPHABET
    CHAR_SET_LEN = len(CHAR_SET)
    CHARACTERS = ''.join(CHAR_SET)

    h5f = h5py.File('images/jd/captcha/origin_jd_captcha_test.h5', 'r')
    images = h5f["X"].value
    texts = h5f["Y"].value
    count = 0
    length = len(images)
    outs = []
    for index, image in enumerate(tqdm(images)):
        _, image = cv2.threshold(image,0.4,1,cv2.THRESH_BINARY)
        # print(image)
        # plt.imshow(image)
        image = np.expand_dims(image, axis=2) 
        image1 = image.transpose(1, 0, 2)
        X_test = np.expand_dims(image1, axis=0)
        y_pred = model.predict(X_test)
        y_pred = y_pred[:,2:,:]
        # out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0])*y_pred.shape[1], )[0][0])[:, :4]
        # print(''.join([CHARACTERS[x] for x in out[:,:4][0]]))
        # plt.show()

        out = K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0])*y_pred.shape[1], )
        outs.append(out[0][0])
    predict_encode_texts = K.batch_get_value(outs)
    for index, encoded in enumerate(tqdm(predict_encode_texts)):
        predict_text = ''.join([CHARACTERS[x] for x in encoded[:,:4][0]])
        if predict_text == texts[index].decode("ascii"):
            count +=1 

    # for index, out in enumerate(tqdm(outs)):
    #     out = K.get_value(out[0][0])[:, :4]
        # out = ''.join([CHARACTERS[x] for x in out[0]])
        # if out == texts[index].decode("ascii"):
        #     count +=1

    #     text = texts[index].decode("ascii")
    #     if text == out:
    #         count += 1
    print("accuracy: " + str(count/length))



# def predict_JD_CTC():
#     import cv2 
#     from skimage.transform import resize
#     import matplotlib.pyplot as plt
#     from keras.models import load_model
#     import numpy as np
#     from keras.models import model_from_json
#     import json

#     # model, base_model, conv_shape = model_builder.CTC()
#     model = load_model("models/model_CTC_1530176456.h5")
#     # with open('data.json') as f:
#     #     data = json.load(f)
#     # model.load_weights('test.h5')
#     # model.summary()
#     # generator = image_generactor.generator_4_multiple_types(batch_size=1, nb_type=5)
#     NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#     ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

#     CHAR_SET = NUMBER + ALPHABET
#     CHAR_SET_LEN = len(CHAR_SET)
#     CHARACTERS = ''.join(CHAR_SET)

#     from tqdm import tqdm
#     filenames = os.listdir("images/jd/captcha/jd/")
#     length = len(filenames)
#     count = 0
#     print(length)
#     for filename in tqdm(filenames):
#         # print(filename)
#         if (filename.endswith(".jpg") or filename.endswith(".jpeg") or
#             filename.endswith(".png") or filename.endswith(".gif")):
#             image = resize(cv2.cvtColor(cv2.imread("images/jd/captcha/jd/" + filename), cv2.COLOR_BGR2GRAY), (36, 150))
#             _, image = cv2.threshold(image,0.5,1,cv2.THRESH_BINARY) 

#             image = np.expand_dims(image, axis=2)
#             # print(image.shape)
#             image = image.transpose(1, 0, 2)
#             X_test = np.expand_dims(image, axis=0)
            
            
#             # print(X_test.shape)
#             y_pred = base_model.predict(X_test)
#             y_pred = y_pred[:,2:,:]
#             out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0])*y_pred.shape[1], )[0][0])[:, :4]
#             out = ''.join([CHARACTERS[x] for x in out[0]])

#             if filename[0:4] == out:
#                 count+=1
#                 # print(count)
#     print("accuracy: " + str(count/length))