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
    sciezkaZrodlowa = "zdjecia/"
    sciezkaDocelowa = "Detekcja okręgów cv2.findContours/"

    kategoriaZdjec = ["banknoty - latwe/", "banknoty - srednie/", "banknoty - trudne/",
                      "monety - latwe/", "monety - srednie/", "monety - trudne/",
                      "challenge/"]


    for iKategoriaZdjec in range(len(kategoriaZdjec)):
        tytul = os.listdir(sciezkaZrodlowa + kategoriaZdjec[iKategoriaZdjec])

        for iTytul in range(len(tytul)):
            print("\nTrwa przetwarzanie obrazu " + str(iTytul) + "/" + str(len(tytul)) + " (" + kategoriaZdjec[iKategoriaZdjec] + tytul[iTytul] + ")")
            oryginal = cv2.imread(sciezkaZrodlowa + kategoriaZdjec[iKategoriaZdjec] + tytul[iTytul], cv2.IMREAD_COLOR)
            minimalnyPromien = int(min(oryginal.shape[:2])/20)


            print("\tTrwa progowanie obrazu...")
            image = progowanie(oryginal)


            print("\tTrwa morfologia konturów...")
            maska = np.ones((5, 5), np.uint8)
            image = cv2.dilate(image, maska, iterations=1)


            print("\tTrwa obliczanie konturów..")
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


            imageKonturyOkregow = cv2.drawContours(oryginal.copy(), konturyOkregow, -1, (0, 255, 0), 7)

            print("\tTrwa rysowanie wykresów...")
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 5), subplot_kw={'xticks': [], 'yticks': []})
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)
            ax[0].imshow(cv2.cvtColor(oryginal, cv2.COLOR_BGR2RGB), aspect='equal', interpolation='bilinear')
            ax[1].set_title("Ilosc konturów: " + str(len(contours)))
            ax[1].imshow(cv2.cvtColor(imageWszystkieKontury, cv2.COLOR_BGR2RGB), aspect='equal', interpolation='bilinear')
            ax[2].set_title("Ilosc okręgów: " + str(len(konturyOkregow)))
            ax[2].imshow(cv2.cvtColor(imageKonturyOkregow, cv2.COLOR_BGR2RGB), aspect='equal', interpolation='bilinear')

            """
            print("\tPrzygotowywanie wyświetlenia obrazów w panelu bocznym...")
            plt.show()
            """

            print("\tTrwa zapisywanie do pliku jpg...")
            fig.savefig(sciezkaDocelowa + kategoriaZdjec[iKategoriaZdjec] + tytul[iTytul])


            #Zwolnienie pamięci zajmowanej przez duże macierze z obrazami
            print("\tTrwa zwolnienie pamięci...")
            plt.close()
            gc.collect()



main()