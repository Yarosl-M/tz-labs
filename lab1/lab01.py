import numpy as np;
import cv2 as cv;
import matplotlib.pyplot as plt;
from sys import argv;

def adjust_hsv(img, hue, sat, br):
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV);
    h, s, v = cv.split(hsv);
    print(h.dtype);
    h1 = cv.add(h.astype(np.int16), hue // 2) % 180;
    s = cv.add(s, sat);
    v = cv.add(v, br);
    final_hsv = cv.merge((h1.astype(np.uint8), s, v));
    return cv.cvtColor(final_hsv, cv.COLOR_HSV2RGB);


######################################################################

if len(argv) != 4:
    print(f"Использование: python {argv[0]} <hue> <saturation> <brightness>");
    exit(-1);

images = ['img1.jpg', 'img2.jpg', 'img3.jpg']; # имена файлов для загрузки
labels = ['Исходное изображение', 'Недоэкспонированное изображение',
          'Переэкспонированное изображение'];

for image_name, label in zip(images[:1], labels[:1]):
    img_bgr = cv.imread(image_name);
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB);
    dh = int(argv[1]);
    ds = int(argv[2]);
    dv = int(argv[3]);

    img_corrected = adjust_hsv(img_rgb, dh, ds, dv);
    
    bins = [i for i in range(256)];
    
    hist1 = cv.calcHist(
        [cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)],
        [0],
        None,
        [256],
        [0, 256]
    );
    hist2 = cv.calcHist(
        [cv.cvtColor(img_corrected, cv.COLOR_RGB2GRAY)],
        [0],
        None,
        [256],
        [0, 256]
    );

    fig, ax = plt.subplots(2, 2);

    
    ax[0, 0].imshow(img_rgb);
    ax[0, 0].set_title('Изображение до коррекции');
    ax[0, 0].axis('off');

    ax[0, 1].bar(bins, hist1, color='black');
    ax[0, 1].set_title('Гистограмма до коррекции');


    ax[1, 0].imshow(img_corrected);
    ax[1, 0].set_title('Изображение после коррекции');
    ax[1, 0].axis('off');
    
    ax[1, 1].bar(bins, hist2, color='black');   
    ax[1, 1].set_title('Гистограмма после коррекции');
    
    fig.suptitle(label);
    plt.tight_layout();
    plt.show();