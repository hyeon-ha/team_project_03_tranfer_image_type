import scipy
from glob import glob
import numpy as np

class DataLoader():
    def __init__(self, dataset_name, img_res=(256, 256)):
        self.dataset_name = dataset_name
        self.img_res = img_res
############################## 데이터 불러오기 및 전처리 #####################################
    def load_data(self, domain, batch_size=1, is_testing=False):
        data_type = "train%s" % domain if not is_testing else "test%s" % domain
        ### #s문자열 데이터를 포맷,
        # is_testing이 False이면 traindomin, True이면 testdomain
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))
        # 로드할 데이터 path 설정
        batch_images = np.random.choice(path, size=batch_size)
        # batch_images들을 뽑아낸다 배치사이즈는 1임
        imgs = []
        for img_path in batch_images: # 각각의 이미지에 대하여 리스이즈와
            img = self.imread(img_path)
            if not is_testing: # 즉 테스트 파일이 아니라면
                img = scipy.misc.imresize(img, self.img_res)
                ###이미지 크기 조정
                if np.random.random() > 0.5: # 랜덤으로 해서 좌우를 바꾼다.
                    img = np.fliplr(img)
            else:
                img = scipy.misc.imresize(img, self.img_res)
            imgs.append(img) # 이미지들을 e다시 imgs에 넣는다.

        imgs = np.array(imgs)/127.5 - 1.  ## -1 과 1 사이로 scale 스케일링

        return imgs

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        path_A = glob('./dataset/%s/%sA/*' % (self.dataset_name, data_type))
        # 만약에 테스트용 데이터가 아니라면 train 테스트용 데이터라면 val
        path_B = glob('./dataset/%s/%sB/*' % (self.dataset_name, data_type))

        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size) # 실행할 배치의 개수
        total_samples = self.n_batches * batch_size # 총 샘플수

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_A = np.random.choice(path_A, total_samples, replace=False) # 랜덤으로 골라 path_A로 만든다. 비복원 추출
        path_B = np.random.choice(path_B, total_samples, replace=False) # 랜덤으로  골라 path_B로 만든다. 비복원 추출

        for i in range(self.n_batches-1): #
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
                img_A = self.imread(img_A)
                img_B = self.imread(img_B)

                img_A = scipy.misc.imresize(img_A, self.img_res)
                img_B = scipy.misc.imresize(img_B, self.img_res)

                if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B

    def load_img(self, path):
        img = self.imread(path)
        img = scipy.misc.imresize(img, self.img_res)
        img = img/127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
