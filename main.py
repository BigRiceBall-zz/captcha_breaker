from captcha_breaker import trainer

# trainer.train(batch_size=128, nb_type=6)
# trainer.predict()
trainer.continue_2_train(batch_size=128, nb_type=6)
# trainer.test2()
# trainer.test()
# def move():
# import os
# import shutil
# images = os.listdir("images/jd/captcha/jd/")
# print(len(images))
# count = 0
# for image in images:
#     if (image[0:3] == "tmp"):
#         shutil.move("images/jd/captcha/jd/" + image, "images/jd/captcha/jd/unknown/"+ image)
#         count+=1
#     print(float(count/len(images)))
# trainer.predict_jd()
# from captcha_breaker import image_generactor
# image_generactor.true_image2H5()

# import h5py
# h5file = h5py.File('images/jd/captcha/origin_jd_captcha.h5', 'r')
# print(h5file['Y'])
# print(h5file['Y'].value[0].decode("ascii"))

