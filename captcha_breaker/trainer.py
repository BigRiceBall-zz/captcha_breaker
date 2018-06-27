
import captcha_breaker.model as model_builder
from captcha_breaker import image_generactor
import time

def train(batch_size=32, nb_type=3):
    now = str(int(time.time()))
    model = model_builder.simple()
    model.compile(loss='categorical_crossentropy',
                optimizer='adadelta',
                metrics=['accuracy'])
                
    model.fit_generator(image_generactor.generator_4_multiple_types(batch_size=batch_size, nb_type=nb_type), 
                        samples_per_epoch=10, nb_epoch=1,
                        nb_worker=28,
                        validation_data=image_generactor.generator_4_multiple_types(batch_size=batch_size, nb_type=nb_type), nb_val_samples=1280)
    model.save("models/model_" + now + ".h5")

def continue_2_train(batch_size=32, nb_type=3):
    from keras.models import load_model
    now = str(int(time.time()))
    model = load_model('models/model_1530090310.h5')
    model.fit_generator(image_generactor.generator_4_multiple_types(batch_size=batch_size, nb_type=nb_type), 
                        samples_per_epoch=128, nb_epoch=32,
                        nb_worker=28,
                        validation_data=image_generactor.generator_4_multiple_types(batch_size=batch_size, nb_type=nb_type), nb_val_samples=1280)
    model.save("models/model_" + now + ".h5")

def test():
    import matplotlib.pyplot as plt
    from keras.models import load_model
    model = load_model('models/model_1530090310.h5')
    generator = image_generactor.generator_4_multiple_types(batch_size=1, nb_type=4)
    X, y = next(generator)
    ture_y = image_generactor.decode(y)
    predicted_text = image_generactor.decode(model.predict(X))
    X = X[0].reshape((36, 150))
    plt.imshow(X, cmap="gray")
    # print(y)
    print("true: " + ture_y + " predict: " + predicted_text)
    plt.show()

def predict():
    import cv2 
    from skimage.transform import resize
    import matplotlib.pyplot as plt
    from keras.models import load_model
    import numpy as np
    image = resize(cv2.cvtColor(cv2.imread("./images/image3.jpeg"), cv2.COLOR_BGR2GRAY), (36, 150))
    # _, image = cv2.threshold(image,0.5,1,cv2.THRESH_BINARY) 

    image1 = np.expand_dims(image, axis=2)
    image1 = np.expand_dims(image1, axis=0)
    print(image1)
    model = load_model('models/model_1530090310.h5')
    predicted_text = image_generactor.decode(model.predict(image1))
    plt.imshow(image, cmap="gray")
    # print(y)
    print(" predict: " + predicted_text)
    plt.show()
    

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
    # model.get_layer('conv2d_1').bias_regularizer = regularizers.l2(0.01)
    model.get_layer('conv2d_1').activity_regularizer = regularizers.l2(0.01) 
    model.get_layer('conv2d_2').kernel_regularizer = regularizers.l2(0.01) 
    # model.get_layer('conv2d_2').bias_regularizer = regularizers.l2(0.01)
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

