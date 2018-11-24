import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import gc



def progowanie(image):
    print("\tTrwa wykrywanie konturow...")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    canny = cv2.Canny(blurred, 50, 100)

    print("\tKoniec wykrywania konturow.")
    return canny



def main():
    sciezkaZrodlowa = "zdjecia/1280 x 960/"
    sciezkaDocelowa = "Rozpoznanie nominału monet/1280 x 960/"
    sciezkaWzorcow = "wzorce/"
    nominaly = ["5gr", "50gr", "1zl", "2zl", "5zl", "10zl", "20zl"]

    tytul = os.listdir(sciezkaZrodlowa)

    sciezkaZapisuWzorca = "wzorce/Znalezione/"
    licznikZnalezionychWzorcow = 0


    for i in range(len(tytul)):
    #for i in [0, 20, 30, 40, 50, 60, 70]:
    #for i in [13]:
        print("Trwa przetwarzanie obrazu o indeksie i = " + str(i) + "/" + str(len(tytul)) + " (" + tytul[i])
        oryginal = cv2.imread(sciezkaZrodlowa + tytul[i], cv2.IMREAD_COLOR)
        minimalnyPromien = int(min(oryginal.shape[:2])/20)
        maksymalnyPromien = int(min(oryginal.shape[:2])/3)


        image = progowanie(oryginal)

        maska = np.ones((5, 5), np.uint8)
        image = cv2.dilate(image, maska, iterations=1)
        im2, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
                """
                print("xmean = " + str(xmean))
                print("ymean = " + str(ymean))
                print("odlegloscOdSrodkaMean = " + str(odlegloscOdSrodkaMean))
                print("odlegloscOdSrodkaStd = " + str(odlegloscOdSrodkaStd))
                print("0.2 * odlegloscOdSrodkaMean = " + str(0.2 * odlegloscOdSrodkaMean))
                """
                if(odlegloscOdSrodkaStd < 0.2 * odlegloscOdSrodkaMean):
                    konturyOkregow.append(contour)

                    #wycinek = np.copy(oryginal[ymin : ymax, xmin : xmax])
                    wycinek = np.copy(oryginal[int(ymean - odlegloscOdSrodkaMean) : int(ymean + odlegloscOdSrodkaMean), int(xmean - odlegloscOdSrodkaMean) : int(xmean + odlegloscOdSrodkaMean)])
                    cv2.imwrite("wzorce/Znalezione/" + str(licznikZnalezionychWzorcow) + ".jpg", wycinek)
                    licznikZnalezionychWzorcow  += 1

        print("\tIdentyfikacja konturów okręgów zakończona.")


        #Zwolnienie pamięci zajmowanej przez duże macierze z obrazami
        print("\tTrwa zwolnienie pamięci...")
        plt.close()
        gc.collect()
        print("\tPomyslnie zwoniono pamięć.\n")



main()