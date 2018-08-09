from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding3D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical

def build(height, width, captcha_length, captcha_alphabet_length):
    #创建输入，结构为 高，宽，通道
    input_tensor = Input( shape=(height, width, 1))
    x = input_tensor
    #构建卷积网络
    #两层卷积层，一层池化层，重复3次。因为生成的验证码比较小，padding使用same
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = Conv2D(128, 3, padding='same',activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    #Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。
    x = Flatten()(x)
    #为输入数据施加Dropout。Dropout将在训练过程中每次更新参数时随机断开一定百分比（rate）的输入神经元，Dropout层用于防止过拟合。
    x = Dropout(0.25)(x)

    #Dense就是常用的全连接层
    #最后连接5个分类器，每个分类器是46个神经元，分别输出46个字符的概率。
    x = [Dense(captcha_alphabet_length, activation='softmax', name='c%d'%(i+1))(x) for i in range(captcha_length)]
    output = concatenate(x)
    model = Model(input=input_tensor, output=output)
    return model
