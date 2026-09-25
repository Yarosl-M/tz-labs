import numpy as np;
import cv2 as cv;
import matplotlib.pyplot as plt;


def buildHistogram(img):
    h = np.zeros(256, dtype=np.int32);
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            h[img[i, j]] = h[img[i, j]] + 1;
    return h;

images = ['img1.jpg', 'img2.jpg', 'img3.jpg']; # имена файлов для загрузки

for image_name in images:
    img_bgr = cv.imread(image_name);
    # копии монохромны
    gray2 = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY);
    h = buildHistogram(gray2);
    bins = [i for i in range(256)];
    plt.bar(bins, h);
    plt.tight_layout();
    plt.show();