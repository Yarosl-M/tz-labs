import cv2 as cv;
import matplotlib.pyplot as plt;
from sys import argv;


def brightness_up(img, c):
    r, g, b = cv.split(img);
    lim = 255 - c;
    r[r > lim] = 255;
    r[r <= lim] += c;
    g[g > lim] = 255;
    g[g <= lim] += c;
    b[b > lim] = 255;
    b[b <= lim] += c;
    return cv.merge((r, g, b));


def brightness_down(img, c):
    r, g, b = cv.split(img);
    lim = c;
    r[r < lim] = 0;
    r[r >= lim] -= c;
    g[g < lim] = 0;
    g[g >= lim] -= c;
    b[b < lim] = 0;
    b[b >= lim] -= c;
    return cv.merge((r, g, b));

######################################################################

if len(argv) != 3:
    print(f"Использование: python {argv[0]} <alpha1> <alpha2>");
    exit(-1);

images = ['img1.jpg', 'img2.jpg', 'img3.jpg']; # имена файлов для загрузки
labels = ['Исходное изображение', 'Недоэкспонированное изображение',
          'Переэкспонированное изображение'];

for image_name, label in zip(images[1:], labels[1:]):
    img_bgr = cv.imread(image_name);
    # gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY);
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB);
    img_rgb_corrected = brightness_up(img_rgb, int(argv[1])) if image_name == 'img2.jpg' else brightness_down(img_rgb, int(argv[2]));
    
    bins = [i for i in range(256)];
    
    hist1 = cv.calcHist(
        [cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)],
        [0],
        None,
        [256],
        [0, 256]
    );
    hist2 = cv.calcHist(
        [cv.cvtColor(img_rgb_corrected, cv.COLOR_RGB2GRAY)],
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


    ax[1, 0].imshow(img_rgb_corrected);
    ax[1, 0].set_title('Изображение после коррекции');
    ax[1, 0].axis('off');
    
    ax[1, 1].bar(bins, hist2, color='black');   
    ax[1, 1].set_title('Гистограмма после коррекции');
    
    fig.suptitle(label);
    plt.tight_layout();
    plt.show();