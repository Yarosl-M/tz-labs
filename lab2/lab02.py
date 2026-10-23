import numpy as np;
import cv2 as cv;
import matplotlib.pyplot as plt;
from sys import argv;


###############################################################################

# 1) Загрузить изображение
img_bgr = cv.imread('bees.png');
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB);
img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY);

# 2) Выполнить бинаризацию
# сначала размытие изображения
img_blur_median = None;
img_blur_gauss = None;

