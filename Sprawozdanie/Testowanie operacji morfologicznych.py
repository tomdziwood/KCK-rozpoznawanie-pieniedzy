import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import gc



def wykrywanieKonturow(image):
    print("\tTrwa wykrywanie konturow...")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


    blurred = cv2.GaussianBlur(gray, (5, 5), 0)



    threshMean = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    threshGauss = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    canny = cv2.Canny(blurred, 50, 100)


    print("\tKoniec wykrywania konturow.")
    return [threshMean, threshGauss, canny]



def morfologia(image):
    print("\tTrwa morfologia konturow...")

    maska = np.ones((5, 5), np.uint8)
    dyl1 = cv2.dilate(image, maska, iterations=1)
    dyl2 = cv2.dilate(image, maska, iterations=2)
    er1 = cv2.erode(image, maska, iterations=1)
    er2 = cv2.erode(image, maska, iterations=2)
    er1dyl1 = cv2.dilate(er1, maska, iterations=1)

    print("\tKoniec morfologii konturow.")
    return [[image, dyl1, dyl2, er1dyl1, er1, er2], ["Obraz podstawowy", "Jednokrotna dylatacja", "Podwójna dylatacja",
                                                     "Jednokrotna erozja + jednokrotna dylatacja", "Jednokrotna erozja", "Podwójna erozja"]]


def main():
    sciezkaZrodlowa = "zdjecia/1280 x 960/"
    sciezkaDocelowa = "Testowanie operacji morfologicznych/1280 x 960/"

    tytul = os.listdir(sciezkaZrodlowa)



    for i in range(len(tytul)):
    #for i in [0]:
        print('Trwa przetwarzanie obrazu o indeksie i =', i, " (" + tytul[i])
        oryginal = cv2.imread(sciezkaZrodlowa + tytul[i], cv2.IMREAD_COLOR)


        print("\tTrwa wykrywanie monet...")
        obrazy = wykrywanieKonturow(oryginal)
        metody = ["cv2.adaptiveThreshold( adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C )", "cv2.adaptiveThreshold( adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C )", "cv2.Canny()"]

        for j in range(len(obrazy)):
            images, tytulyWykresow = morfologia(obrazy[j])

            print("\tTrwa rysowanie wykresow...")
            fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 10), subplot_kw={'xticks': [], 'yticks': []})
            fig.subplots_adjust(left=0, right=1, bottom=0, top=0.9, hspace=0.1, wspace=0.1)
            fig.suptitle(metody[j], fontsize=20)
            for k in range(6):
                ax[int(k / 3), k % 3].set_title(tytulyWykresow[k])
                ax[int(k / 3), k % 3].imshow(images[k], cmap='gray', aspect='equal', interpolation='bilinear')


            print("\tTrwa zapisywanie do pliku jpg...")
            fig.savefig(sciezkaDocelowa + tytul[i][:-4] + "_" + str(j) + ".jpg")
            print("\tPomyslnie zapisano do pliku jpg.\n\n")

            """
            print("\tPrzygotowywanie wyswietlenia obrazow w panelu bocznym...")
            plt.show()
            print("\tPomyslnie wyswietlono.\n\n")
            """

            # Zwolnienie pamięci zajmowanej przez duże macierze z obrazami
            print("\tTrwa zwolnienie pamięci...")
            plt.close()
            gc.collect()
            print("\tPomyslnie zwoniono pamięć.\n")



main()