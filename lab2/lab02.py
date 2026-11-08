import numpy as np;
import cv2 as cv;
import matplotlib.pyplot as plt;
from sys import argv;


###############################################################################

# 1) Загрузить изображение
img_bgr = cv.imread('bees.png');
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB);
gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY);

# 2) Выполнить бинаризацию
# сначала размытие изображения
#img_blur_median = None;
#img_blur_gauss = None;

THRESH = 127;

stuff = cv.threshold(gray, THRESH, 255, cv.THRESH_BINARY_INV);

# TODO: нужно будет также отображать гистограммы и порог бинаризации
fig, ax = plt.subplots(3, 2);
ax[0,0].imshow(img_rgb);
ax[0,0].set_title('Исходное изображение');
ax[0,0].axis('off');

ax[0, 1].imshow(gray, cmap='gray', vmin=0, vmax=255);
ax[0, 1].set_title('Изображение в оттенках серого');
ax[0, 1].axis('off');
##############################################3
ax[1,0].imshow(img_rgb);
ax[1,0].set_title('Исходное изображение');
ax[1,0].axis('off');

ax[1, 1].imshow(gray, cmap='gray', vmin=0, vmax=255);
ax[1, 1].set_title('Изображение в оттенках серого');
ax[1, 1].axis('off');
##############################################
ax[2,0].imshow(img_rgb);
ax[2,0].set_title('Исходное изображение');
ax[2,0].axis('off');

ax[2, 1].imshow(gray, cmap='gray', vmin=0, vmax=255);
ax[2, 1].set_title('Изображение в оттенках серого');
ax[2, 1].axis('off');