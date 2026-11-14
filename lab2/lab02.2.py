import numpy as np;
import cv2 as cv;
import matplotlib.pyplot as plt;


FILENAME = ['bees.png', 'img.jpg', 'img2.webp',
            'aaaaaaaaaaaaaaaaaaaa.jpg',
            '388.jpg', ]#'394.jpg']

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
        # if output[i, j] < 0:
        #     output[i, j] = 0;
        # elif output[i, j] > 255:
        #     output[i, j] = 255;
    return np.round(output);

KERNEL_HIGHPASS_X = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]]);
KERNEL_HIGHPASS_Y = np.array([[-1, -2, -1],
                              [0,0,0],
                              [1, 2, 1]]);

def sobel(img):
    dx = convolution2(img, KERNEL_HIGHPASS_X);
    dy = convolution2(img, KERNEL_HIGHPASS_Y);
    s = np.sqrt(np.square(dx) + np.square(dy));
    return (dx, dy, s);

###############################################################################

# 1) Загрузить изображение
img_bgr = cv.imread('./stuff/' + FILENAME[-1]);
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB);
gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY);

dx, dy, g = sobel(gray);

fig, ax = plt.subplots(2, 2);
ax[0, 0].imshow(gray, cmap="gray", vmin=0, vmax=255);
ax[0, 0].set_title("Исходное изображение");

ax[0, 1].imshow(g, cmap="gray");
ax[0, 1].set_title("Результат применения\nоператора Собеля");

ax[1, 0].imshow(dx, cmap="gray");
ax[1, 0].set_title("Производная по оси X");

ax[1, 1].imshow(dy, cmap="gray");
ax[1, 1].set_title("Производная по оси Y")

fig.tight_layout();
plt.show();