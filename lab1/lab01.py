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
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY);
    
    # эквализация изображения
    equalized = histEqualization(gray);
    
    fig, ax = plt.subplots(2, 2);
    
    bins = [i for i in range(256)];
    
    # отображение исходного изображения в оттенках серого
    ax[0, 0].imshow(gray, cmap='gray', vmin=0, vmax=255);
    ax[0, 0].set_title('Исходное изображение');
    ax[0, 0].axis('off');
    
    # построение гистограммы исходного изображения
    h = cv.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    );
    ax[0, 1].bar(bins, h);
    ax[0, 1].set_title('Исходная гистограмма');
    
    # отображение изображения после эквализации
    ax[1, 0].imshow(equalized, cmap='gray', vmin=0, vmax=255);
    ax[1, 0].set_title('После эквализации');
    ax[1, 0].axis('off');
    
    # построение гистограммы изображения после эквализации
    h_equalized = cv.calcHist(
        [equalized],
        [0],
        None,
        [256],
        [0, 256]
    );
    ax[1, 1].bar(bins, h_equalized);
    ax[1, 1].set_title('Гистограмма после эквализации');
    
    fig.suptitle(label);
    plt.tight_layout();
    plt.show();