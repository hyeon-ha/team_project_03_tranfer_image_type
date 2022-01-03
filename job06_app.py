import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PIL import Image # pillow 설치
import numpy as np
from tensorflow.keras.models import load_model
from PyQt5.QtGui import *
import matplotlib.pyplot as plt
import tensorflow as tf

form_window = uic.loadUiType('./job07_gan_app.ui')[0]

class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('Gan지 미술관')
        self.pixmap_1.hide()
        self.buttons = [self.output1, self.output2, self.output3, self.output4]
        self.buttons_flag = [True, True, True, True]
        self.pixmap_1
        self.setWindowIcon(QIcon())
        for i in range(len(self.buttons)):
            self.buttons[i].hide()
        self.image_muse.clicked.connect(self.image_gan)
        self.next.clicked.connect(self.next_button_slot)
        self.open_file.clicked.connect(self.open_file_slot)
        self.mone_style.clicked.connect(self.mone_style_slot)
        # self.mone_style.clicked.connect(self.image_generator(file))
    def image_gan(self):
        self.pixmap_1.hide()
        self.next.show()
        for i in range(len(self.buttons)):
            self.buttons[i].hide()
        self.buttons[0].show()
        self.buttons_flag[0] = False

    def next_button_slot(self):
        self.buttons
        for i in range(len(self.buttons_flag)):
            if self.buttons_flag[i] == False:
                self.buttons[i+1].show()
                self.buttons_flag[i+1] = False
                self.buttons[i].hide()
                self.buttons_flag[i] = True
                break

    def open_file_slot(self):
        for i in range(len(self.buttons)):
            self.buttons[i].hide()
        self.fname = QFileDialog.getOpenFileName(self)
        print(self.fname[0])
        pm = QPixmap(self.fname[0])

        self.pixmap_1.setPixmap(pm)
        self.pixmap_1.show()
        self.next.hide()
        # pixmap_1.setPixmap(pixmap)
        # self.image_generator(fname[0])

    def mone_style_slot(self):
        img = Image.open(self.fname[0])
        img = img.convert('RGB')  # RGB모드로 변환
        img = img.resize((256, 256))  # 사이즈는 튜플로
        data = np.asarray(img) / 127.5 - 1  # 이미지를 어레이로 바꾼다.
        data = np.expand_dims(data, axis=0)  # np 확장시켜서 넣는다.
        print('a1')
        G_AB = load_model('./model_colab/cycle_ganA_epoch115.h5')
        print('a2')
        fake_B=G_AB.predict(data)
        print('a3')
        gen_imgs = np.concatenate([data, fake_B])
        print('a4')
        gen_imgs = 0.5 * gen_imgs + 0.5
        plt.imsave('filename.jpeg', gen_imgs[1])
        pm = QPixmap('./filename.jpeg')
        self.pixmap_1.setPixmap(pm)
        self.pixmap_1.show()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = Exam()
    mainWindow.show()
    sys.exit(app.exec_())