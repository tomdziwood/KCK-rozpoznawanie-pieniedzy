from skimage import io, img_as_ubyte
from matplotlib import pyplot as plt
import numpy as np
import os

def plot_hist_caly(img):
    img = img_as_ubyte(img)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    histo, x = np.histogram(img[:, :, 0], range(0, 256), density=True)
    ax[0].plot(histo)
    ax[0].set_xlim(0, 255)
    ax[0].set_ylim(0, max(histo))

    histo, x = np.histogram(img[:, :, 1], range(0, 256), density=True)
    ax[1].plot(histo)
    ax[1].set_xlim(0, 255)
    ax[1].set_ylim(0, max(histo))

    histo, x = np.histogram(img[:, :, 2], range(0, 256), density=True)
    ax[2].plot(histo)
    ax[2].set_xlim(0, 255)
    ax[2].set_ylim(0, max(histo))

    plt.show()


def plot_hist_okrag(img):
    img = img_as_ubyte(img)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    dane = img[:, :, 0]
    histo, x = np.histogram(dane[dane != 255], range(0, 256), density=True)
    ax[0].plot(histo)
    ax[0].set_xlim(0, 255)
    ax[0].set_ylim(0, max(histo))

    dane = img[:, :, 1]
    histo, x = np.histogram(dane[dane != 255], range(0, 256), density=True)
    ax[1].plot(histo)
    ax[1].set_xlim(0, 255)
    ax[1].set_ylim(0, max(histo))

    dane = img[:, :, 2]
    histo, x = np.histogram(dane[dane != 255], range(0, 256), density=True)
    ax[2].plot(histo)
    ax[2].set_xlim(0, 255)
    ax[2].set_ylim(0, max(histo))

    plt.show()


def przypadek01(sciezkaZrodlowa, tytul):
    image = io.imread(sciezkaZrodlowa + tytul[45])

    print("\t\nTrwa rysowanie obrazu...")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image, aspect='equal', interpolation='bilinear')
    plt.show()
    print("Rysowanie histogramu obrazu...\n")
    plot_hist_caly(image)

    print("Trwa szykowanie wycinka...")
    wycinek = image[140:415, 715:990]
    xlen = wycinek.shape[0]
    ylen = wycinek.shape[1]
    x = np.zeros((xlen, ylen))
    y = np.zeros((xlen, ylen))
    for i in range(xlen):
        for j in range(ylen):
            x[i, j] = i
            y[i, j] = j
    r = xlen / 2
    wycinek[(x - xlen / 2) ** 2 + (y - ylen / 2) ** 2 > r ** 2] = [255, 255, 255]

    print("Trwa rysowanie wycinku...")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(wycinek, aspect='equal', interpolation='bilinear')
    plt.show()

    print("Rysowanie histogramu wycinka...")
    plot_hist_okrag(wycinek)


def przypadek02(sciezkaZrodlowa, tytul):
    image = io.imread(sciezkaZrodlowa + tytul[45])

    """
    print("\t\nTrwa rysowanie obrazu...")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image, aspect='equal', interpolation='bilinear')
    plt.show()
    print("Rysowanie histogramu obrazu...\n")
    plot_hist_caly(image)
    """

    print("Trwa szykowanie wycinka...")
    wycinek = image[280:550, 240:510]
    xlen = wycinek.shape[0]
    ylen = wycinek.shape[1]
    x = np.zeros((xlen, ylen))
    y = np.zeros((xlen, ylen))
    for i in range(xlen):
        for j in range(ylen):
            x[i, j] = i
            y[i, j] = j
    r = xlen / 2
    wycinek[(x - xlen / 2) ** 2 + (y - ylen / 2) ** 2 > r ** 2] = [255, 255, 255]

    print("Trwa rysowanie wycinku...")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(wycinek, aspect='equal', interpolation='bilinear')
    plt.show()

    print("Rysowanie histogramu wycinka...")
    plot_hist_okrag(wycinek)


def przypadek03(sciezkaZrodlowa, tytul):
    image = io.imread(sciezkaZrodlowa + tytul[45])

    """
    print("\t\nTrwa rysowanie obrazu...")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image, aspect='equal', interpolation='bilinear')
    plt.show()
    print("Rysowanie histogramu obrazu...\n")
    plot_hist_caly(image)
    """

    print("Trwa szykowanie wycinka...")
    wycinek = image[565:835, 467:737]
    xlen = wycinek.shape[0]
    ylen = wycinek.shape[1]
    x = np.zeros((xlen, ylen))
    y = np.zeros((xlen, ylen))
    for i in range(xlen):
        for j in range(ylen):
            x[i, j] = i
            y[i, j] = j
    r = xlen / 2
    wycinek[(x - xlen / 2) ** 2 + (y - ylen / 2) ** 2 > r ** 2] = [255, 255, 255]

    print("Trwa rysowanie wycinku...")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(wycinek, aspect='equal', interpolation='bilinear')
    plt.show()

    print("Rysowanie histogramu wycinka...")
    plot_hist_okrag(wycinek)


def main():
    print("Poczatek testowania 1zl.")

    sciezkaZrodlowa = "zdjecia/1280 x 960/"
    tytul = os.listdir(sciezkaZrodlowa)

    przypadek01(sciezkaZrodlowa, tytul)
    przypadek02(sciezkaZrodlowa, tytul)
    przypadek03(sciezkaZrodlowa, tytul)

    print("Koniec testowania 1zl.")

main()