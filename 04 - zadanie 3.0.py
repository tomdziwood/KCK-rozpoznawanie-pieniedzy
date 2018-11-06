from skimage import io, img_as_ubyte, filters
from skimage.morphology import disk
import skimage.morphology as mp
from skimage.color import rgb2gray, gray2rgb
from matplotlib import pyplot as plt
import numpy as np


def plot_hist(img):
    img = img_as_ubyte(img)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    #print("img.shape = " + str(img.shape))
    histo, x = np.histogram(img[:, :, 0], range(0, 256), density=True)
    ax[0].plot(histo)
    ax[0].set_xlim(0, 255)
    histo, x = np.histogram(img[:, :, 1], range(0, 256), density=True)
    ax[1].plot(histo)
    ax[1].set_xlim(0, 255)
    histo, x = np.histogram(img[:, :, 2], range(0, 256), density=True)
    ax[2].plot(histo)
    ax[2].set_xlim(0, 255)
    #print("type(histo) = " + str(type(histo)))
    #print("type(x) = " + str(type(x)))
    #print("histo.shape = " + str(histo.shape))
    #print("x.shape = " + str(x.shape))
    plt.show()


def progowanie(image):
    srednia = [0, 0, 0]
    odchylenie = [0, 0, 0]
    for i in range(3):
        srednia[i] = np.mean(image[:, :, i])
        odchylenie[i] = np.std(image[:, :, i])

    #for m in range(3):
    #    color = image[:, :, m]
    #    color[color > srednia[m] - odchylenie[m]] = 255
    #    image[:, :, m] = color

    decyzja = np.ones((len(image), len(image[0])))
    for m in range(3):
        decyzja[image[:, :, m] < srednia[m] - 2.5 * odchylenie[m]] *= 0
    image[decyzja == 1] = [255, 255, 255]
    image[decyzja == 0] = [0, 0, 0]


def konwolucja(image):
    img = rgb2gray(image)
    img = filters.sobel(img)
    return gray2rgb(img)


def kontrastowanie(image):
    srednia = [0, 0, 0]
    odchylenie = [0, 0, 0]
    for i in range(3):
        srednia[i] = np.mean(image[:, :, i])
        odchylenie[i] = np.std(image[:, :, i])

    for m in range(3):
        color = image[:, :, m]
        color[color >= srednia[m] + 3 * odchylenie[m]] = 1
        color[color < srednia[m] + 3 * odchylenie[m]] = 0
        image[:, :, m] = color

def odszumianie(image):
    img = rgb2gray(image)
    img =  filters.median(img, disk(2))
    return gray2rgb(img)

def dylatacja(image):
    return mp.dilation(image)

def erozja(image):
    return mp.erosion(image)

def main():
    sciezka = ['dane/samolot00.jpg', 'dane/samolot01.jpg', 'dane/samolot02.jpg',
               'dane/samolot03.jpg', 'dane/samolot04.jpg', 'dane/samolot05.jpg',
               'dane/samolot06.jpg', 'dane/samolot07.jpg', 'dane/samolot08.jpg',
               'dane/samolot09.jpg', 'dane/samolot10.jpg', 'dane/samolot11.jpg',
               'dane/samolot12.jpg', 'dane/samolot13.jpg', 'dane/samolot14.jpg',
               'dane/samolot15.jpg', 'dane/samolot16.jpg', 'dane/samolot17.jpg',
               'dane/samolot18.jpg', 'dane/samolot19.jpg', 'dane/samolot20.jpg']

    """
    for i in range(5):
        image = io.imread(sciezka[i])
        plot_hist(image)
    """

    fig, ax = plt.subplots(nrows=7, ncols=3, figsize=(10, 17), subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)
    for i in range(21):
        print('Trwa przetwarzanie obrazu o indeksie i =', i)
        image = io.imread(sciezka[i])
        #print("type(image) = " + str(type(image)))
        #print(str(image.shape))

        print("\tTrwa progowanie...")
        progowanie(image)

        print("\tTrwa konwolucja...")
        image = konwolucja(image)

        print("\tTrwa kontrastowanie...")
        kontrastowanie(image)

        print("\tTrwa odszumianie...")
        image = odszumianie(image)

        print("\tTrwa dylatacja...")
        image = dylatacja(image)
        """
        print("\tTrwa erozja...")
        image = erozja(image)
        """
        ax[int(i / 3), i % 3].imshow(image,
                                     aspect='equal',
                                     interpolation='bilinear')



    print("\nTrwa zapisywanie do pliku jpg...")
    fig.savefig("wykresy/3/Zadanie 3.0.jpg", dpi = 480)
    print("Pomyslnie zapisano do pliku jpg.")
    print("\nPrzygotowywanie wyswietlenia obrazow w panelu bocznym...")
    plt.show()
    print("Pomyslnie wyswietlono.")


main()