from skimage import io, img_as_ubyte, filters
from skimage.morphology import disk
import skimage.morphology as mp
from skimage.color import rgb2gray, gray2rgb
from matplotlib import pyplot as plt
import numpy as np


def plot_hist(img):
    img = img_as_ubyte(img)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    histo, x = np.histogram(img[:, :, 0], range(0, 256), density=True)
    ax[0].plot(histo)
    ax[0].set_xlim(0, 255)
    histo, x = np.histogram(img[:, :, 1], range(0, 256), density=True)
    ax[1].plot(histo)
    ax[1].set_xlim(0, 255)
    histo, x = np.histogram(img[:, :, 2], range(0, 256), density=True)
    ax[2].plot(histo)
    ax[2].set_xlim(0, 255)
    plt.show()


def progowanie(image):
    img = np.copy(image)
    srednia = [0, 0, 0]
    odchylenie = [0, 0, 0]
    for i in range(3):
        srednia[i] = np.mean(img[:, :, i])
        odchylenie[i] = np.std(img[:, :, i])

    decyzja = np.ones((len(img), len(img[0])))
    for m in range(3):
        decyzja[img[:, :, m] < srednia[m] - 2.5 * odchylenie[m]] *= 0
    img[decyzja == 1] = [255, 255, 255]
    img[decyzja == 0] = [0, 0, 0]

    return img


def konwolucja(image):
    img = rgb2gray(image)
    img = filters.sobel(img)
    return gray2rgb(img)


def kontrastowanie(image):
    img = np.copy(image)
    srednia = [0, 0, 0]
    odchylenie = [0, 0, 0]
    for i in range(3):
        srednia[i] = np.mean(img[:, :, i])
        odchylenie[i] = np.std(img[:, :, i])

    for m in range(3):
        color = img[:, :, m]
        color[color >= srednia[m] + 3 * odchylenie[m]] = 1
        color[color < srednia[m] + 3 * odchylenie[m]] = 0
        img[:, :, m] = color

    return img

def odszumianie(image):
    img = rgb2gray(image)
    img =  filters.median(img, disk(2))
    return gray2rgb(img)

def dylatacja(image):
    return mp.dilation(image)

def erozja(image):
    return mp.erosion(image)


def wykrywanie10gr(image):
    return image

def wykrywanie20gr(image):
    return image

def wykrywanie50gr(image):
    return image

def wykrywanie1zl(image):
    img = np.copy(image)

    print("\t\tTrwa progowanie...")
    img = progowanie(img)

    print("\t\tTrwa konwolucja...")
    img = konwolucja(img)

    print("\t\tTrwa kontrastowanie...")
    img = kontrastowanie(img)

    print("\t\tTrwa odszumianie...")
    img = odszumianie(img)

    print("\t\tTrwa dylatacja...")
    img = dylatacja(img)
    """
    print("\t\tTrwa erozja...")
    img = erozja(img)
    """
    return img

def wykrywanie2zl(image):
    return image

def wykrywanie5zl(image):
    return image

def wykrywanie10zl(image):
    return image

def wykrywanie20zl(image):
    return image



def main():
    tytul = ['1.jpg', '2.jpg', '3.jpg',
               '4.jpg', '5.jpg', '6.jpg',
               '7.jpg', '8.jpg']

    """
    for i in range(5):
        image = io.imread(sciezka[i])
        plot_hist(image)
    """


    for i in range(8):
        print('Trwa przetwarzanie obrazu o indeksie i =', i)
        oryginal = io.imread('zdjecia/' + tytul[i])


        print("\tTrwa wyszukiwanie 1zl...")
        image = wykrywanie1zl(oryginal)


        print("\t\nTrwa rysowanie wykresow...")
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.5), subplot_kw={'xticks': [], 'yticks': []})
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)
        ax[0].imshow(oryginal, aspect='equal', interpolation='bilinear')
        ax[1].imshow(image, aspect='equal', interpolation='bilinear')

        print("\tTrwa zapisywanie do pliku jpg...")
        fig.savefig("wyniki/" + tytul[i], dpi=160)
        print("\tPomyslnie zapisano do pliku jpg.")
        print("\tPrzygotowywanie wyswietlenia obrazow w panelu bocznym...")
        plt.show()
        print("\tPomyslnie wyswietlono.\n\n")


main()