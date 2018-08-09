# Captcha Breaker

This is a pre-trained model to recognise characters in a CAPTCHA. Presently, It can handle one type of CAPTCHA at one time, i.e., it can only handle 4 charcter or 5 character per model. 

# Installation 
I recommand you to create a virtualenv to install packages locally.
```
pip install -r requirements.txt
```
# Data Preparation
Put images to images/, and run ```python -m dataset_builder``` it will automatically create ```.npy``` file to images/all/.

# Train the model
After data preparation, you can simply run ```python -m train```.


# Predict
After training, you can simply run ```python -m predict -mp MODEL_PATH -imp IMAGE_PATH```

# Usage
```
usage: main.py [-h] [-ht HEIGHT] [-wh WIDTH] [-cl CAPTCHA_LENGTH]
               [-cal CAPTCHA_ALPHABET] [-b BATCH_SIZE] [-m MODE] [-ep EPOCHS]
               [-mp MODEL_PATH] [-imp IMAGE_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -ht HEIGHT, --height HEIGHT
                        image height
  -wh WIDTH, --width WIDTH
                        image width
  -cl CAPTCHA_LENGTH, --captchalength CAPTCHA_LENGTH
                        the number of character in image
  -cal CAPTCHA_ALPHABET, --alphabetlength CAPTCHA_ALPHABET
                        the number of character in image
  -b BATCH_SIZE, --batchsize BATCH_SIZE
                        batch size
  -m MODE, --mode MODE  train or predict or dataset_builder
  -ep EPOCHS, --epochs EPOCHS
                        the number of iteration
  -mp MODEL_PATH, --model MODEL_PATH
                        model path
  -imp IMAGE_PATH, --image IMAGE_PATH
                        image path
```
