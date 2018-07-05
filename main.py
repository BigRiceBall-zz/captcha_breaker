from captcha_breaker import trainer
# from captcha_breaker import image_generactor

# trainer.train_CTC(batch_size=128, nb_type=6)

# trainer.test_CTC()
# trainer.predict_JD_CTC()
# trainer.test_JD_CTC()
# trainer.continue_2_train_CTC(batch_size=128, nb_type=6)

# trainer.train(batch_size=128, nb_type=6)
trainer.continue_2_train(batch_size=128, nb_type=6)

# trainer.predict()
# trainer.continue_2_train(batch_size=128, nb_type=6)
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
# trainer.test_JD()


# from captcha_breaker import ocr

# ocr.parse_images_to_text()

# 1: normal, 8 types
# 2: continue_to_train 6 types
# 3: continue_to_train 8 types
# 4: ctc 6 types
# 5: normal, average 6 types