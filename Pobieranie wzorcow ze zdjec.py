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
    sciezkaZrodlowa = "wzorce/zdjecia wzorcowe/3/"

    tytul = os.listdir(sciezkaZrodlowa)

    sciezkaZapisuWzorca = "wzorce/znalezione/"


    for iTytul in range(len(tytul)):
        print("\nTrwa przetwarzanie obrazu " + str(iTytul) + "/" + str(len(tytul)) + " (" + tytul[iTytul] + ")")
        oryginal = cv2.imread(sciezkaZrodlowa + tytul[iTytul], cv2.IMREAD_COLOR)
        minimalnyPromien = int(min(oryginal.shape[:2])/20)


        print("\tTrwa progowanie obrazu...")
        image = progowanie(oryginal)


        print("\tTrwa morfologia konturów...")
        maska = np.ones((5, 5), np.uint8)
        image = cv2.dilate(image, maska, iterations=1)


        print("\tTrwa obliczanie konturów..")
        im2, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        print("\tTrwa identyfikacja konturów okręgów...")
        licznikZnalezionychWzorcow = 1
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
                    wycinek = np.copy(oryginal[int(ymean - odlegloscOdSrodkaMean) : int(ymean + odlegloscOdSrodkaMean), int(xmean - odlegloscOdSrodkaMean) : int(xmean + odlegloscOdSrodkaMean)])
                    cv2.imwrite(sciezkaZapisuWzorca + tytul[iTytul][:-4] + "_" + str(licznikZnalezionychWzorcow) + ".jpg", wycinek)
                    licznikZnalezionychWzorcow  += 1


        #Zwolnienie pamięci zajmowanej przez duże macierze z obrazami
        print("\tTrwa zwolnienie pamięci...")
        plt.close()
        gc.collect()



main()