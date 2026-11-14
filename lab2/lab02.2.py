import numpy as np;
import cv2 as cv;
import matplotlib.pyplot as plt;
from sys import argv;


FILENAME = ['bees.png', 'img.jpg', 'img2.webp',
            'aaaaaaaaaaaaaaaaaaaa.jpg',
            '388.jpg', '394.jpg']

###############################################################################

# 1) Загрузить изображение
img_bgr = cv.imread('./stuff/' + FILENAME[-1]);
img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB);
gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY);

# 2) Выполнить бинаризацию
THRESH = 128;
BLOCKSIZE_MEAN = 25;
C_MEAN = 6;
BLOCKSIZE_GAUSS = 37;
C_GAUSS = 5;

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
        # ax[1].legend();
    ax[1].set_title(f'Гистограмма изображения {"\n(порог — адаптивный)" if threshold is None else ""}');
    ax[1].set_xlim(0, 255);
    ax[1].set_xlabel('Уровень яркости');
    ax[1].set_ylabel('Количество пикселей');

    if title == 'Бинаризация методом Отцу':
        thresholds = np.arange(255);
        variances = [];
        for value in thresholds:
            group1 = gray[gray <= value];
            group2 = gray[gray > value];
            # if group1.size == 0 or group2.size == 0:
                # variances.append(np.nan);
            if group1.size == 0:
                variances.append(np.var(group2));
            elif group2.size == 0:
                variances.append(np.var(group1));
            else:
                variance = (
                    group1.size * np.var(group1)
                    + group2.size * np.var(group2)
                ) / gray.size;
                # variance = np.var(group1) + np.var(group2);
                variances.append(variance);

        variance_ax = ax[1].twinx();
        variance_ax.plot(
            thresholds,
            variances,
            color='blue',
            linestyle='--',
            alpha=0.75,
            label='Взвешенная сумма дисперсий'
        );
        hist_handles, hist_labels = ax[1].get_legend_handles_labels()
        var_handles, var_labels = variance_ax.get_legend_handles_labels()

        ax[1].legend(
            hist_handles + var_handles,
            hist_labels + var_labels,
            loc='upper left'
        )
        variance_ax.set_ylabel('Взвешенная сумма дисперсий');
        # variance_ax.legend(loc='lower left');
    elif threshold is not None:
        ax[1].legend();

    fig.tight_layout();
    plt.show();