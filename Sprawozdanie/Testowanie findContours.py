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



def main():
    sciezkaZrodlowa = "zdjecia/1280 x 960/"
    sciezkaDocelowa = "Testowanie findContours/1280 x 960/"

    tytul = os.listdir(sciezkaZrodlowa)



    for i in range(len(tytul)):
    #for i in [0]:
        print('Trwa przetwarzanie obrazu o indeksie i =', i, " (" + tytul[i])
        oryginal = cv2.imread(sciezkaZrodlowa + tytul[i], cv2.IMREAD_COLOR)


        print("\tTrwa wykrywanie monet...")
        obrazy = wykrywanieKonturow(oryginal)

        print("\tTrwa morfologia konturow...")
        maska = np.ones((5, 5), np.uint8)
        for j in range(3):
            obrazy[j] = cv2.dilate(obrazy[j], maska, iterations=1)

        contours = [0, 0, 0]
        wynik = [0, 0, 0]
        metoda = ["cv2.ADAPTIVE_THRESH_MEAN_C", "cv2.ADAPTIVE_THRESH_GAUSSIAN_C", "cv2.Canny()"]
        for j in range(3):
            im2, contours[j], hierarchy = cv2.findContours(obrazy[j], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            wynik[j] = cv2.drawContours(oryginal.copy(), contours[j], -1, (0, 255, 0), 3)

        print("\tTrwa rysowanie wykresow...")
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 10), subplot_kw={'xticks': [], 'yticks': []})
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)
        for j in range(3):
            ax[0, j].imshow(obrazy[j], cmap='gray', aspect='equal', interpolation='bilinear')
            ax[0, j].set_title(metoda[j])
            ax[1, j].imshow(cv2.cvtColor(wynik[j], cv2.COLOR_BGR2RGB), aspect='equal', interpolation='bilinear')
            ax[1, j].set_title("Liczba konturów: " + str(len(contours[j])))

        """
        print("\tPrzygotowywanie wyswietlenia obrazow w panelu bocznym...")
        plt.show()
        print("\tPomyslnie wyswietlono.\n\n")
        """

        print("\tTrwa zapisywanie do pliku jpg...")
        fig.savefig(sciezkaDocelowa + tytul[i])
        print("\tPomyslnie zapisano do pliku jpg.\n\n")

        # Zwolnienie pamięci zajmowanej przez duże macierze z obrazami
        print("\tTrwa zwolnienie pamięci...")
        plt.close()
        gc.collect()
        print("\tPomyslnie zwoniono pamięć.\n")



main()