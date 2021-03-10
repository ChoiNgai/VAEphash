import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# 感知哈希算法
def pHash(image):
    image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     cv2.imshow('image', image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # 将灰度图转为浮点型，再进行dct变换
    dct = cv2.dct(np.float32(image))
    #     print(dct)
    # 取左上角的8*8，这些代表图片的最低频率
    # 这个操作等价于c++中利用opencv实现的掩码操作
    # 在python中进行掩码操作，可以直接这样取出图像矩阵的某一部分
    dct_roi = dct[0:8, 0:8]
    avreage = np.mean(dct_roi)
    hash = []
    for i in range(dct_roi.shape[0]):
        for j in range(dct_roi.shape[1]):
            if dct_roi[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash

# 计算汉明距离
def Hamming_distance(hash1, hash2):
    num = 0
    for index in range(len(hash1)):
        if hash1[index] != hash2[index]:
            num += 1
    return num


if __name__ == "__main__":

    str_images0 = os.listdir('datasets/images/')
    str_images1 = os.listdir('datasets/images1/')
    similar = np.empty([len(str_images0), 1], dtype=float, order='C')

    if(len(str_images0) <= len(str_images1)):           #按最小的长度比对
        lenmin = len(str_images0)
    else:
        lenmin = len(str_images1)

    list_simi = []

    for i in range(0, lenmin - 1, 1):
        images0_list = str_images0[i] # 读取文件夹里面的图片名
        images1_list = str_images1[i]  # 读取文件夹里面的图片名

        image_file1 = str('datasets/images/'+str_images0[i])  # 在此读取目标图像
        image_file2 = str('datasets/images/'+str_images1[i])

        img1 = cv2.imread(image_file1)
        img2 = cv2.imread(image_file2)
        hash1 = pHash(img1)
        hash2 = pHash(img2)
        dist = Hamming_distance(hash1, hash2)
        # 将距离转化为相似度
        similar[i] = 1 - dist * 1.0 / 64
        print("第",i+1,"张图片相似度:",similar[i])
        list_simi.append(similar[i])

    simi = np.empty([len(str_images0), 1], dtype=float, order='C')        #忘了这个是啥，算了，不删了

plt.plot(list_simi)
plt.xlabel('index')
plt.ylabel('similar')
plt.title('Video similarity analysis results')
plt.savefig('./datasets/result.jpg')
plt.show()



