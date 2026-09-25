import numpy as np;
import cv2 as cv;
import matplotlib.pyplot as plt;

print("Hello Console World!");

def rgbToGrayScale(img):
    output = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8);
    for i in range(img.shape[0]):       # строки
        for j in range(img.shape[1]):   # столбцы
            # предоставленная формула для перевода в оттенки серого
            # 0.114 синий канал
            # 0.587 зелёный
            # 0.299 красный
            output[i, j] = (
                0.114 * img[i, j, 0]
                + 0.587 * img[i, j, 1]
                + 0.299 * img[i, j, 2]
            );
    return output;

images = ['img1.jpg', 'img2.jpg', 'img3.jpg']; # имена файлов для загрузки

for image_name in images:
    img_bgr = cv.imread(image_name); # opencv сначала загружает в BGR
    # перевод в RGB стандартной функцией OpenCV
    # так как matplotlib использует RGB
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB);
    # перевод в оттенки серого собственным методом
    gray1 = rgbToGrayScale(img_bgr);
    # и методом в OpenCV
    gray2 = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY);
    # рисунок и оси (функции subplots() для отрисовки изображений)
    fig, ax = plt.subplots(1, 3);
    # исходное изображение (цветное)
    ax[0].imshow(img_rgb);
    ax[0].set_title('Цветное изображение');
    # копии монохромны
    ax[1].imshow(gray1, cmap='gray', vmin=0, vmax=255); # cmap=gray для вывода в градациях серого
    ax[1].set_title('Собственный метод');
    
    ax[2].imshow(gray2, cmap='gray', vmin=0, vmax=255);
    ax[2].set_title('cv.cvtColor()');
    # убрать координатные оси с пометками (незачем для данного случая)
    for a in ax:
        a.axis('off');
    fig.suptitle(image_name);
    plt.tight_layout();
    plt.show();
##    break;
