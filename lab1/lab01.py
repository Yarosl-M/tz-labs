import numpy as np;
import cv2 as cv;
import matplotlib.pyplot as plt;


def makeCdf(histogram):
    integral = np.zeros_like(histogram);
    integral[0] = histogram[0];
    for i in range(1, integral.shape[0]):
        integral[i] = integral[i - 1] + histogram[i];
    return integral;

def histEqualization(img):
    h = cv.calcHist(
        [img],      # исходное изображение в виде списка
        [0],        # индекс канала для которого вычисляется гистограмма (0 — в оттенках серого, 1—3 — цвета BGR)
        None,       # маска (None — без маски)
        [256],      # массив размеров гистограммы
        [0, 256]    # диапазон значений яркости
    );
    numPixels = img.shape[0] * img.shape[1];
    
    cdf = makeCdf(h);
    # нормализация
    firstNonZero = 0;
    for el in h:
        if el != 0:
            firstNonZero = el;
            break;
    sourceRange = numPixels;
    destinationRange = 256;
    lut = np.zeros_like(h);
    for i in range(destinationRange):
        offset = cdf[i] - firstNonZero;
        if offset < 0:
            offset = 0;
        lut[i] = (offset / sourceRange) * destinationRange;
    return cv.LUT(img, lut).astype(np.uint8);
    return None
    #



images = ['img1.jpg', 'img2.jpg', 'img3.jpg']; # имена файлов для загрузки
labels = ['Исходное изображение', 'Недоэкспонированное изображение',
          'Переэкспонированное изображение'];

for image_name, label in zip(images, labels):
    # чтение файла изображения 
    img_bgr = cv.imread(image_name);
    
    # copies are monochrome
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY);
    
    fig, ax = plt.subplots(1, 2);
    
    # отображение изображения в оттенках серого
    ax[0].imshow(gray, cmap='gray', vmin=0, vmax=255);
    ax[0].set_title('Изображение в оттенках серого');
    ax[0].axis('off');

    bins = [i for i in range(256)];
    # построение гистограммы через OpenCV
    h = cv.calcHist(
        [gray],    # исходное изображение в виде списка
        [0],        # индекс канала для которого вычисляется гистограмма (0 — в оттенках серого, 1—3 — цвета BGR)
        None,       # маска (None — без маски)
        [256],      # массив размеров гистограммы
        [0, 256]    # диапазон значений яркости
    );
    # 
    ax[1].bar(bins, h);
    ax[1].set_title('Гистограмма cv.calcHist()');
    
    fig.suptitle(label);
    plt.tight_layout();
    plt.show();