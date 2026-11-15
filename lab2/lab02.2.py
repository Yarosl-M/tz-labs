import numpy as np;
import cv2 as cv;
import matplotlib.pyplot as plt;
from matplotlib.patches import Rectangle as PlotRectangle;
from random import randint;
from classes import find_bounding_box;


FILENAME = '394.jpg';

KERNEL_SIZE = 15;

THRESH = 120;
BLOCKSIZE_MEAN = 25;
C_MEAN = 4;

MIN_AREA = 350;
MAX_AREA = 999999999999999999999;

###############################################################################

img_bgr = cv.imread('./stuff/' + FILENAME);
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB);
gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY);

img_blur = cv.GaussianBlur(gray, (KERNEL_SIZE, KERNEL_SIZE), 0);   

img_thresh = cv.adaptiveThreshold(img_blur,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY_INV, blockSize=BLOCKSIZE_MEAN, C=C_MEAN);

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

# после размытия
plt.figure(dpi=200);
plt.imshow(img_blur, cmap='gray', vmin=0, vmax=255);
plt.title('Размытие по Гауссу');
plt.axis('off');
plt.show();

# после бинаризации
plt.figure(dpi=200);
plt.imshow(img_thresh, cmap='gray');
plt.title('После адаптивной бинаризации');
plt.axis('off');
plt.show();

# num_labels — количество меток
# labels — изображение, где каждому пикселю
# соответствует определённая метка
# (метка 0 — для фона)
num_labels, labels = cv.connectedComponents(img_thresh);

bb = find_bounding_box(labels, num_labels);
# print(bb[1:4])

label_color = [(randint(0,179), randint(150,200), randint(200,255))
               for i in range(num_labels)];
label_color[0] = (20,20,20);
img_objects = np.zeros((img_thresh.shape[0],img_thresh.shape[1],3), dtype=np.uint8);
for i in range(img_thresh.shape[0]):
    for j in range(img_thresh.shape[1]):
        img_objects[i, j] = label_color[labels[i, j]];
img_objects = cv.cvtColor(img_objects, cv.COLOR_HSV2RGB);
# отдельные промаркированные компоненты
plt.figure(dpi=200);
plt.imshow(img_objects);
plt.title('Размеченные связные компоненты изображения');
plt.axis('off');
plt.show();

plt.figure(dpi=200);
img_rect = np.copy(img_objects);
i = 0;
for b in filter(lambda box: box.area >= MIN_AREA and box.area <= MAX_AREA, bb[1:]):
    cv.rectangle(img_rect,
                 pt1=(b.upper_left.x, b.upper_left.y),
                 pt2=(b.lower_right.x, b.lower_right.y),
                 color=(255,0,0),
                 thickness=2);
print(max([b.area for b in bb[1:]]));
print(max([b.area for b in bb]));
plt.imshow(img_rect);
plt.title('Объекты, выделенные прямоугольниками');
plt.axis('off');
# и с прямоугольниками

    # plt.gca().add_patch(PlotRectangle(
    #     (b.upper_left.x - 0.5, b.upper_left.y - 0.5),
    #     b.width + 1,
    #     b.height + 1,
    #     fill=False,
    #     edgecolor='red',
    #     linewidth=1
    # ));
plt.show();