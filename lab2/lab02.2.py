import numpy as np;
import cv2 as cv;
import matplotlib.pyplot as plt;
from random import randint;


FILENAME = '394.jpg';

KERNEL_SIZE = 15;

THRESH = 120;
BLOCKSIZE_MEAN = 25;
C_MEAN = 6;

MIN_AREA = 0;
MAX_AREA = 12251997;

###############################################################################

img_bgr = cv.imread('./stuff/' + FILENAME);
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB);
gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY);

img_blur = cv.GaussianBlur(gray, (KERNEL_SIZE, KERNEL_SIZE), 0);   

img_thresh = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY, blockSize=BLOCKSIZE_MEAN, C=C_MEAN);

img_segment = img_thresh;
# затем выводится img_segment опять но с прямоугольниками окда

# показать исходное изображение и ч/б
fig, ax = plt.subplots(1, 2, figsize=(10,4), dpi=200);
ax[0].imshow(img_rgb);
ax[0].set_title('Исходное изображение')
ax[0].axis('off');

ax[1].imshow(gray, cmap='gray', vmin=0, vmax=255);
ax[1].set_title('Изображение в оттенках серого');
ax[1].axis('off');

fig.tight_layout();
plt.show();

# wait a sec
# might have cut too much off?
# whatever
# в общем, что надо
# в первой строке два столбца: оригинал цветной и ч/б
# затем:
# 1) размытие
# 2) бинаризация
# 3) маркировка (would be fun to make random HSV colours)
# 4) окаймление прямоугольниками
# ну всё равно можно каждый отдельно выводить
# можно и без цикла действительно

plt.imshow(img_blur, cmap='gray', vmin=0, vmax=255);
plt.title('Размытие по Гауссу');
plt.axis('off');

for i in range(21):
    # basically a mockup of generating a random color okay
    h = randint(0, 179);
    s = randint(180, 255);
    v = randint(200, 255);

plt.show();

