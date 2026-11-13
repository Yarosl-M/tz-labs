import numpy as np;
import cv2 as cv;
import matplotlib.pyplot as plt;
from sys import argv;


FILENAME = 'bees.png';
FILENAME_NEW = 'img.jpg';

###############################################################################

# 1) Загрузить изображение
img_bgr = cv.imread(FILENAME);
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB);
gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY);

# 2) Выполнить бинаризацию
THRESH = 72;

thresh_manual, img_manual = cv.threshold(gray, THRESH, 255, cv.THRESH_BINARY);
thresh_otsu, img_otsu = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
img_mean = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY, blockSize=43, C=1);
img_gauss = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY, blockSize=55, C=1);

# TODO: нужно будет также отображать гистограммы и порог бинаризации
fig, ax = plt.subplots(3, 2, dpi=200);


ax[0,0].imshow(img_rgb);
ax[0,0].set_title('Исходное изображение');
ax[0,0].axis('off');

ax[0, 1].imshow(gray, cmap='gray', vmin=0, vmax=255);
ax[0, 1].set_title('Изображение в оттенках серого');
ax[0, 1].axis('off');

# ручная бинаризация с пороговым значением = THRESH
ax[1,0].imshow(img_manual, cmap='gray', vmin=0, vmax=255);
ax[1,0].set_title('Ручная бинаризация');
ax[1,0].axis('off');

# бинаризация методом Отцу
ax[1, 1].imshow(img_otsu, cmap='gray', vmin=0, vmax=255);
ax[1, 1].set_title('Бинаризация методом Отцу (порог = %d)' % thresh_otsu);
ax[1, 1].axis('off');

# адаптивная бинаризация № 1
ax[2,0].imshow(img_mean, cmap='gray', vmin=0, vmax=255);
ax[2,0].set_title('Адаптивная бинаризация\nпо среднему значению');
ax[2,0].axis('off');

# адаптивная бинаризация № 2
ax[2, 1].imshow(img_gauss, cmap='gray', vmin=0, vmax=255);
ax[2, 1].set_title('Адаптивная бинаризация по Гауссу');
ax[2, 1].axis('off');

fig.tight_layout();
plt.show();