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
labels = ['Исходное изображение', 'Недоэкспонированное изображение',
          'Переэкспонированное изображение'];

for (image_name, label) in zip(images, labels):
    img_bgr = cv.imread(image_name);
    # копии монохромны
    gray2 = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY);
    # два подграфика для двух разных гистограмм
    fig, ax = plt.subplots(1, 2);
    
    bins1 = [i for i in range(256)];
    # гистограмма по собственному методу
    h1 = buildHistogram(gray2);
    ax[0].bar(bins1, h1);
    ax[0].set_title('Собственная гистограмма' + '\n' + label);

    
    # гистограмма по методу из OpenCV
    h2 = cv.calcHist(
        [gray2],    # исходное изображение в виде списка
        [0],        # индекс канала для которого вычисляется гистограмма (0 — в оттенках серого, 1—3 — цвета BGR)
        None,       # маска (None — без маски)
        [256],      # массив размеров гистограммы
        [0, 256]    # диапазон значений яркости
    );
    # print(h2.ndim);
    ax[1].bar(bins1, h2);
    print(gray2.size);
    ax[1].set_title('Гистограмма cv.calcHist()' + '\n' + label);
 
    plt.tight_layout();
    plt.show();