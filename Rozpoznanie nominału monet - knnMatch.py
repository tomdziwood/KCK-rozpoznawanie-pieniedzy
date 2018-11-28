import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import gc



def progowanie(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    canny = cv2.Canny(blurred, 50, 100)

    return canny



def main():
    sciezkaZrodlowa = "zdjecia/monety - latwe/"
    sciezkaWzorcow = "wzorce/"
    nominaly = ["5gr", "50gr", "1zl", "2zl", "5zl", "10zl", "20zl"]

    tytul = os.listdir(sciezkaZrodlowa)


    #for i in range(len(tytul)):
    #for i in [0, 20, 30, 40, 50, 60, 70]:
    for iTytul in [13]:
        print("\n\nTrwa przetwarzanie obrazu o indeksie i = " + str(iTytul) + "/" + str(len(tytul)) + " (" + tytul[iTytul] + ")")
        oryginal = cv2.imread(sciezkaZrodlowa + tytul[iTytul], cv2.IMREAD_COLOR)
        minimalnyPromien = int(min(oryginal.shape[:2])/20)


        print("\tTrwa wykrywanie konturow...")
        image = progowanie(oryginal)


        print("\tTrwa morfologia konturów...")
        maska = np.ones((5, 5), np.uint8)
        image = cv2.dilate(image, maska, iterations=1)


        im2, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        imageWszystkieKontury = cv2.drawContours(oryginal.copy(), contours, -1, (0, 255, 0), 3)

        print("\tTrwa identyfikacja konturów okręgów...")
        konturyOkregow = []
        for contour in contours:

            x = contour[:, 0, 0]
            y = contour[:, 0, 1]

            xmin = min(x)
            xmax = max(x)
            ymin = min(y)
            ymax = max(y)
            dx = xmax - xmin
            dy = ymax - ymin

            if(dx >= minimalnyPromien) and (dy >= minimalnyPromien):
                xmean = np.mean(x)
                ymean = np.mean(y)
                odlegloscOdSrodka = ((xmean - x) ** 2 + (ymean - y) ** 2) ** (0.5)
                odlegloscOdSrodkaMean = np.mean(odlegloscOdSrodka)
                odlegloscOdSrodkaStd = np.std(odlegloscOdSrodka)

                if(odlegloscOdSrodkaStd < 0.2 * odlegloscOdSrodkaMean):
                    konturyOkregow.append(contour)

                    wycinek = np.copy(oryginal[ymin : ymax, xmin : xmax])

                    #przejscie po folderach z nominałami
                    for iNominaly in range(len(nominaly)):
                        sciezkaWzorca = sciezkaWzorcow + str(nominaly[iNominaly]) + '/'
                        tytulWzorca = os.listdir(sciezkaWzorca)

                        #przejscie po wzorcach danego nominału
                        for iTytulWzorca in range(min(len(tytulWzorca), 10)):
                            wzorzec = cv2.imread(sciezkaWzorca + tytulWzorca[iTytulWzorca], cv2.IMREAD_COLOR)
                            print("\t\tWczytano wzorzec: " + sciezkaWzorca + tytulWzorca[iTytulWzorca])

                            orb = cv2.ORB_create()
                            kpWzorzec, desWzorzec = orb.detectAndCompute(wzorzec, None)
                            kpWycinek, desWycinek = orb.detectAndCompute(wycinek, None)
                            bf = cv2.BFMatcher()
                            matches = bf.knnMatch(desWzorzec, desWycinek, k=2)
                            good = []
                            for m, n in matches:
                                if m.distance < 0.75 * n.distance:
                                    good.append([m])
                            dopasowanie = cv2.drawMatchesKnn(wzorzec, kpWzorzec, wycinek, kpWycinek, good, outImg=None, flags=2)

                            print("\t\tTrwa rysowanie wykresów...")
                            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 7),subplot_kw={'xticks': [], 'yticks': []})
                            fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)
                            ax[0].set_title("Wzorzec")
                            ax[0].imshow(cv2.cvtColor(wzorzec, cv2.COLOR_BGR2RGB), aspect='equal',interpolation='bilinear')
                            ax[1].set_title("Wycinek")
                            ax[1].imshow(cv2.cvtColor(wycinek, cv2.COLOR_BGR2RGB), aspect='equal',interpolation='bilinear')
                            ax[2].set_title("Ilosc dopasowań: " + str(len(good)))
                            ax[2].imshow(cv2.cvtColor(dopasowanie, cv2.COLOR_BGR2RGB), aspect='equal',interpolation='bilinear')

                            print("\t\tPrzygotowywanie wyświetlenia obrazów w panelu bocznym...")
                            plt.show()
                            plt.close()


        print("\tIdentyfikacja konturów okręgów zakończona.")


        imageKonturyOkregow = cv2.drawContours(oryginal.copy(), konturyOkregow, -1, (0, 255, 0), 3)

        print("\tTrwa rysowanie wykresow...")
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 5), subplot_kw={'xticks': [], 'yticks': []})
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)
        ax[0].imshow(cv2.cvtColor(oryginal, cv2.COLOR_BGR2RGB), aspect='equal', interpolation='bilinear')
        ax[1].set_title("Ilosc konturów: " + str(len(contours)))
        ax[1].imshow(cv2.cvtColor(imageWszystkieKontury, cv2.COLOR_BGR2RGB), aspect='equal', interpolation='bilinear')
        ax[2].set_title("Ilosc konturów: " + str(len(konturyOkregow)))
        ax[2].imshow(cv2.cvtColor(imageKonturyOkregow, cv2.COLOR_BGR2RGB), aspect='equal', interpolation='bilinear')


        print("\tPrzygotowywanie wyświetlenia obrazów w panelu bocznym...")
        plt.show()


        """
        print("\tTrwa zapisywanie do pliku jpg...")
        fig.savefig(sciezkaDocelowa + tytul[i])
        print("\tPomyslnie zapisano do pliku jpg.\n\n")
        """


        #Zwolnienie pamięci zajmowanej przez duże macierze z obrazami
        print("\tTrwa zwolnienie pamięci...")
        plt.close()
        gc.collect()



main()