import matplotlib.pyplot as plt
import numpy as np
import os

import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


OUT_DIR = 'CNN_OUT_img/'
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
img_shape = (28, 28, 1) # 이미지 차원을 설정한다. 28*28*1 < 마지막은 배치
epoch = 10000 # 에폭은 1만번
batch_size = 128
noise = 100
sample_interval = 100

#build generator 생성자 모델
generator_model = Sequential()
generator_model.add(Dense(256*7*7, input_dim=noise)) # input_dim 은 noise(입력)를 준다. 출력은 256*7*7로 한다.
generator_model.add(Reshape((7, 7, 256))) # 차원 재배열 (7,7,256)으로 재배열한다.
generator_model.add(Conv2DTranspose(128, kernel_size=3, # 이미지 크기를 다시 크게한다. 파라미터 수 3*3*256(채널)+128
                strides=2, padding='same'))
generator_model.add(BatchNormalization()) # 128*4   (gamma, beta, mean, variance)
generator_model.add(LeakyReLU(alpha=0.01))
generator_model.add(Conv2DTranspose(64, kernel_size=3,
                strides=1, padding='same'))
generator_model.add(BatchNormalization())
generator_model.add(LeakyReLU(alpha=0.01))
generator_model.add(Conv2DTranspose(1, kernel_size=3,
                strides=2, padding='same')) # 28*28*1 로 출력된다.
generator_model.add(Activation('tanh'))
generator_model.summary()

# build discriminator
discriminator_model = Sequential()
discriminator_model.add(Conv2D(32, kernel_size=3,
                strides=2, padding='same', input_shape=img_shape)) # 입력은 (28,28,1)이 되어야 한다.
#discriminator_model.add(BatchNormalization())
discriminator_model.add(LeakyReLU(alpha=0.01))
discriminator_model.add(Conv2D(64, kernel_size=3,
                strides=2, padding='same'))
#discriminator_model.add(BatchNormalization())
discriminator_model.add(LeakyReLU(alpha=0.01))
discriminator_model.add(Conv2D(128, kernel_size=3,
                strides=2, padding='same'))
#discriminator_model.add(BatchNormalization())
discriminator_model.add(LeakyReLU(alpha=0.01))
discriminator_model.add(Flatten())
discriminator_model.add(Dense(1, activation='sigmoid'))
discriminator_model.summary()

discriminator_model.compile(loss='binary_crossentropy',
                optimizer='adam', metrics=['accuracy']) # 맞냐 틀리냐
discriminator_model.trainable = False

# build GAN
gan_model = Sequential()
gan_model.add(generator_model)
gan_model.add(discriminator_model)
print(gan_model.summary())
gan_model.compile(loss='binary_crossentropy', optimizer='adam') # gan 모델 생성 맞냐 틀리냐

(X_train, Y_train), (X_test, Y_test) = mnist.load_data() # 데이터를 받는다.

print(X_train.shape, Y_train.shape)
MY_NUMBER = 9
X_train = X_train[Y_train == MY_NUMBER]
print(len(X_train))

_, axs = plt.subplots(4, 4, figsize=(4, 4),
            sharey=True, sharex=True)
cnt = 0
for i in range(4):
    for j in range(4):
        axs[i, j].imshow(X_train[cnt, :, :], cmap='gray')
        axs[i, j].axis('off')
        cnt += 1
plt.show()


X_train = X_train / 127.5 - 1 #0-1사이 값을 갖도록 한다. 넘피 임
print(X_train[0]) # 한차원빠지고 출력된다.
X_train = np.expand_dims(X_train, axis=3) #np 확장시켜서 넣는다.
print(X_train.shape)

real = np.ones((batch_size, 1)) # 1로 채운다.
print(real.shape) #(128,1) 차원
print(real[0])
fake = np.zeros((batch_size, 1)) # 0으로 채운다.

for itr in range(epoch):
    idx = np.random.randint(0, X_train.shape[0], batch_size) # 배치사이즈 만큼 0~xtrain.shape[0] 까지 수에서 batch 사이즈 만큼(128)개수의 난수를 뽑아내 넘피 어레이로 만든다.

    real_imgs = X_train[idx] # 각각의 인덱스된 값을 real_imgs로 만든다. (128,28,28,1)



    z = np.random.normal(0, 1, (batch_size, noise)) # 0부터 1까지 배치 128짜리 총 100개의 입력값을 가진 z를 만든다. shape 128,100

    fake_imgs = generator_model.predict(z) # 생성자 모델에 넣어 예측값을 구한다. -> 28*28이 된다. (출력1) 총 128개가 만들어진다. 차원이 하나 더생겨 저장됨 (128, 28, 28, 1) 무작위이미지에서 새로운 이미지를 생성


    d_hist_real = discriminator_model.train_on_batch(real_imgs, real) # 실제 이미지를 학습시키면서 가중치를 업데이트한다 1로서 학습시킨다.

    d_hist_fake = discriminator_model.train_on_batch(fake_imgs, fake) # 가짜 이미지를 학습시키면서 가중치를 업데이트한다 0으로서 학습시킨다.

    d_loss, d_acc = 0.5 * np.add(d_hist_real, d_hist_fake) # 손실값과 정확도를 0.5씩 가중치를 두고 d_loss와 d_acc를 생성한다.
    discriminator_model.trainable = False # 판별자는 학습하지 않는다.

    z = np.random.normal(0, 1, (batch_size, noise)) # 잡음생성
    gan_hist = gan_model.train_on_batch(z, real) # 생성자와 판별자를 통과시키고 예측된 z를 실제가 되도록 학습한다. 그렇지만 생성자만 학습된다. # loss값이 나온다. metrics=['accruacy']는 정확도도나온다.


    if itr % sample_interval == 0:
        print('%d [D loss: %f, acc.: %.2f%%] [G loss: %f]'%(
            itr, d_loss, d_acc * 100, gan_hist))
        row = col = 4
        z = np.random.normal(0, 1, (row * col, noise))
        fake_imgs = generator_model.predict(z)
        fake_imgs = 0.5 * fake_imgs + 0.5
        _, axs = plt.subplots(row, col, figsize=(row, col),
                              sharey=True, sharex=True)
        cnt = 0
        for i in range(row):
            for j in range(col):
                axs[i, j].imshow(fake_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        path = os.path.join(OUT_DIR, 'img-{}'.format(itr+1))
        plt.savefig(path)
        plt.close()

generator_model.save('./generator_mnist_{}.h5'.format(MY_NUMBER))










