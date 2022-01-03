import tensorflow as tf


from tensorflow.keras.initializers import RandomNormal

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import datetime
import matplotlib
import copy
import numpy as np
import os




OUT_DIR = 'CNN_OUT_img/'
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
img_shape = (256, 256, 3) # 이미지 차원을 설정한다. 28*28*3
epoch = 10000 # 에폭은 1만번
batch_size = 1
noise = 100
sample_interval = 100
# 생성자 모델
generator_model = Sequential()
input_image = Input(shape=img_shape) #잡음
print(input_image)
conv_init = RandomNormal(0, 0.02)

def res_block(input_data, n_filter, kernel_size=3, strides=1):
    x = Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT'))(input_data)
    x = Conv2D(filters=n_filter, kernel_size=kernel_size, strides=strides,
               kernel_initializer=conv_init)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Lambda(lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT'))(x)
    x = Conv2D(filters=n_filter, kernel_size=kernel_size, strides=strides,
               kernel_initializer=conv_init)(x)
    x = BatchNormalization()(x)
    shortcut = input_data  ## x
    merged = Add()([x, shortcut])
    return merged
########################################### 생성자 모델 ###################################
# x = Sequential()
x = Lambda(lambda x: tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT'))(input_image)
x = Conv2D(filters=32, kernel_size=7, strides=1, kernel_initializer=conv_init)(x)

x = BatchNormalization()(x)
x = Activation("relu")(x)

x = Conv2D(filters=64, kernel_size=3, strides=2, padding="same", kernel_initializer=conv_init)(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = Conv2D(filters=128, kernel_size=3, strides=2, padding="same", kernel_initializer=conv_init)(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)


####### resnet_9blocks
R128 = res_block(x, 128)
R128 = res_block(R128, 128)
R128 = res_block(R128, 128)
R128 = res_block(R128, 128)
R128 = res_block(R128, 128)
R128 = res_block(R128, 128)
R128 = res_block(R128, 128)
R128 = res_block(R128, 128)
R128 = res_block(R128, 128)

x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding="same",kernel_initializer=conv_init)(R128)
x = BatchNormalization()(x)
x = Activation("relu")(x)

x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding="same",kernel_initializer=conv_init)(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)

u32 = Lambda(lambda x: tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT'))(x)
c731_3 = Conv2D(3, kernel_size=7, strides=1, activation='tanh',kernel_initializer=conv_init)(u32)  ## c731_3._keras_shape

output = c731_3

generator_model = Model(inputs=input_image, outputs=output)
generator_model.summary()

print("## Generator ##")

########### 판별자 생성 ####################
input_image = Input(shape=img_shape)
instance_norm = True
x = Conv2D(filters=64, kernel_size=4, strides=2, padding='same', kernel_initializer=conv_init)(input_image)

x = Conv2D(filters=128, kernel_size=4, strides=2, padding='same', kernel_initializer=conv_init)(input_image)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.2)(x)

x = Conv2D(filters=256, kernel_size=4, strides=2, padding='same', kernel_initializer=conv_init)(input_image)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.2)(x)

x = Conv2D(filters=512, kernel_size=4, strides=2, padding='same', kernel_initializer=conv_init)(input_image)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.2)(x)

output = Conv2D(1, kernel_size = 4, strides=1, padding = "same", kernel_initializer=conv_init)(x)

discriminator_model = Model(inputs = input_image, outputs = output)

print("## Discriminator ##")
discriminator_model.summary()
####################### 판별자 생성자 명명 ###############################################
optimizer = Adam(0.0002, 0.5)
D_A = discriminator_model   ## mone class를 구분하는 Discriminator
D_A.compile(loss='mse', optimizer=optimizer)
D_B = discriminator_model
D_B.compile(loss='mse', optimizer=optimizer)
lambda_cycle = 10

G_AB = generator_model  ## mone class --> B class  G 함수
G_BA = generator_model ## B class --> mone class  F 함수

############################ 이미지 명명 ########################################
# 잡음
image_A = Input(shape=img_shape)  ## mone class 이미지
image_B = Input(shape=img_shape)  ## B class 이미지
# 가짜 이미지
fake_B = G_AB(image_A)  ## B class 가짜 생성 이미지
fake_A = G_BA(image_B)  ## mone class 가짜 생성 이미지
#복원 이미지
reconstruct_A = G_BA(fake_B)
reconstruct_B = G_AB(fake_A)

#identity
identity_A = G_BA(image_A)
identity_B = G_AB(image_B)
############################## 학습 옵션 ###################################
## Discriminator의 Weight 학습을 안시킴
D_A.trainable = False
D_B.trainable = False

valid_A = D_A(fake_A)  ## 가짜 mone 이미지를 Discriminator_A에 넣었을 때  점수  D_A(F(B)) (0~1)
valid_B = D_B(fake_B)  ## 가짜 B 이미지를 Discriminator_B에 넣었을 때  점수

combined = Model(inputs=[image_A, image_B], outputs = [valid_A, valid_B, reconstruct_A, reconstruct_B, identity_A, identity_B])

combined.compile(loss=["mse", "mse","mae", "mae","mae", "mae"], loss_weights = [1, 1, lambda_cycle, lambda_cycle,1, 1],optimizer = optimizer)


########################### 학습 ###############################################
img_rows = 256
img_cols = 256
channels = 3

# cyclegan.train(epochs=200, batch_size=1, sample_interval=200)
start_time = datetime.datetime.now()
patch = int(img_rows / 2**4)
D_patch = (patch, patch, 1)

real = np.ones((batch_size,) +  D_patch) # 1로 채운다.
fake = np.zeros((batch_size,) + D_patch) # 0으로 채운다.
epochs = 200
batch_size =1
sample_interval = 200


X_train, X_test = np.load('./dataset/mone_image_data.npy', allow_pickle = True) # pickle 은 객체의 형태를 그대로 유지하며 저장
Y_train, Y_test = np.load('./dataset/picture_image_data.npy', allow_pickle = True) # pickle 은 객체의 형태를 그대로 유지하며 저장
for epoch in range(epochs):
    for batch_i, (imgs_A, imgs_B) in enumerate(zip(X_train, Y_train)):
        fake_B = G_AB.predict(imgs_A)
        fake_A = G_BA.predict(imgs_B)

        #--------------
        # 판별기 학습
        #--------------

        dA_loss_real = D_A.train_on_batch(imgs_A, real) # imgs_A를 real로 학습
        dA_loss_fake = D_A.train_on_batch(fake_A, fake) # fake_A를 fake로 학습
        dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

        dB_loss_real = D_B.train_on_batch(imgs_B, real)
        dB_loss_fake = D_B.train_on_batch(fake_B, fake)
        dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

        # Total disciminator loss
        d_loss = 0.5 * np.add(dA_loss, dB_loss)

        # ------------------
        #  Generator 학습
        # ------------------

        g_loss = combined.train_on_batch([imgs_A, imgs_B],
                                              [real, real,
                                               imgs_A, imgs_B,
                                               imgs_A, imgs_B])

        elapsed_time = datetime.datetime.now() - start_time
        # Plot the progress
        print("[Epoch %d/%d] [Batch %d] [D loss: %f] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
              % (epoch, epochs,
                 batch_i,
                 d_loss,
                 g_loss[0],
                 np.mean(g_loss[1:3]),
                 np.mean(g_loss[3:5]),
                 np.mean(g_loss[5:6]),
                 elapsed_time))

        # If at save interval => save generated image samples
        # if batch_i % sample_interval == 0:
        #     sample_images(epoch, batch_i)

generator_model.save('./generator_mnist_{}.h5'.format(1))
