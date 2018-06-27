
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
                        samples_per_epoch=51200, nb_epoch=5,
                        nb_worker=4,
                        validation_data=image_generactor.generator_4_multiple_types(batch_size), nb_val_samples=1280)
    model.save("models/model_" + now + ".h5")