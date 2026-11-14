import numpy as np;
import cv2 as cv;
import matplotlib.pyplot as plt;
from sys import argv;


FILENAME = ['bees.png', 'img.jpg', 'img2.webp',
            'aaaaaaaaaaaaaaaaaaaa.jpg',
            '388.jpg', '394.jpg']

def convolution2(image, kernel):
    # расстояние от ядра
    d = int((kernel.shape[0] - 1)/2);
    output = np.zeros_like(image, dtype=np.float32);
    for i in range(0, image.shape[0]): # строки изображения
        for j in range(0, image.shape[1]): # столбцы изображения
            kernel_i = -1; 
            kernel_j = -1;
            for ni in range(i-d, i+d+1): # итерация по строкам ядра
                ni_new = ni;
                kernel_i += 1;
                # отражение при выходе за края изображения
                if ni < 0:
                    ni_new = -ni - 1;
                elif ni >= image.shape[0]:
                    ni_new = image.shape[0] - (ni - image.shape[0] + 1)
                for nj in range(j-d, j+d+1): # итерация по столбцам ядра
                    nj_new = nj;
                    kernel_j += 1;
                    if nj < 0:
                        nj_new = -nj - 1;
                    elif nj >= image.shape[1]:
                        nj_new = image.shape[1] - (nj - image.shape[1] + 1)
                    value = image[ni_new, nj_new]* kernel[kernel_i, kernel_j];
                    output[i, j] = output[i, j] + value;
                kernel_j = -1;
        if output[i, j] < 0:
            output[i, j] = 0;
        elif output[i, j] > 255:
            output[i, j] = 255;
    return np.round(output).astype(np.uint8);

KERNEL_LOWPASS = np.ndarray(shape=(67,67), dtype=np.float32);
# midpoint = (KERNEL_LOWPASS.shape[0] // 2, KERNEL_LOWPASS.shape[1] // 2);
# for i in range(KERNEL_LOWPASS.shape[0]):
#     for j in range(KERNEL_LOWPASS.shape[1]):
#         # kernel coef = 1 / (manhattan distance from center + 1) (just for fun lol)
#         distance = abs(i - midpoint[0]) + abs(j - midpoint[1]);
#         KERNEL_LOWPASS[i, j] = 1 / (distance + 1);
KERNEL_LOWPASS /= np.sum(KERNEL_LOWPASS);
KERNEL_LOWPASS = np.ones_like(KERNEL_LOWPASS);
KERNEL_LOWPASS /= np.sum(KERNEL_LOWPASS);

###############################################################################

# 1) Загрузить изображение
img_bgr = cv.imread('./stuff/' + FILENAME[-1]);
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB);
gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY);

# 2) Выполнить бинаризацию
THRESH = 120;
BLOCKSIZE_MEAN = 25;
C_MEAN = 6;
BLOCKSIZE_GAUSS = 37;
C_GAUSS = 5;

gray = convolution2(gray, KERNEL_LOWPASS);

thresh_manual, img_manual = cv.threshold(gray, THRESH, 255, cv.THRESH_BINARY);
thresh_otsu, img_otsu = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
img_mean = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY, blockSize=BLOCKSIZE_MEAN, C=C_MEAN);
img_gauss = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY, blockSize=BLOCKSIZE_GAUSS, C=C_GAUSS);

# гистограмма ч/б изображения
bins = np.arange(256);
histogram = cv.calcHist([gray], [0], None, [256], [0, 256]);

# показать исходное изображение и ч/б
fig, ax = plt.subplots(1, 3, figsize=(14,4), dpi=200);
ax[0].imshow(img_rgb);
ax[0].axis('off');

ax[1].imshow(gray, cmap='gray', vmin=0, vmax=255);
ax[1].set_title('Исходное изображение\nи изображение в оттенках серого');
ax[1].axis('off');

ax[2].bar(bins, histogram);
ax[2].set_title('Гистограмма изображения в оттенках серого');
ax[2].set_xlim(0, 255);
ax[2].set_xlabel('Уровень яркости');
ax[2].set_ylabel('Количество пикселей');

fig.tight_layout();
plt.show();

# результаты разных методов бинаризации и гистограммы
binary_images = [
    ('Ручная бинаризация', img_manual, thresh_manual),
    ('Бинаризация методом Отцу', img_otsu, thresh_otsu),
    ('Адаптивная бинаризация\nпо среднему значению', img_mean, None),
    ('Адаптивная бинаризация по Гауссу', img_gauss, None),
];

for title, binary_img, threshold in binary_images:
    fig, ax = plt.subplots(1, 2, figsize=(10,4),dpi=200);
    binary_img = np.bitwise_not(binary_img);
    ax[0].imshow(binary_img, cmap='gray', vmin=0, vmax=255);
    ax[0].set_title(title);
    ax[0].axis('off');

    ax[1].bar(bins, histogram);
    if threshold is not None:
        ax[1].axvline(
            threshold,
            color='red',
            linestyle='--',
            label='Порог = %d' % threshold
        );
        ax[1].legend();
    ax[1].set_title(f'Гистограмма изображения {"\n(порог — адаптивный)" if threshold is None else ""}');
    ax[1].set_xlim(0, 255);
    ax[1].set_xlabel('Уровень яркости');
    ax[1].set_ylabel('Количество пикселей');

    fig.tight_layout();
    plt.show();