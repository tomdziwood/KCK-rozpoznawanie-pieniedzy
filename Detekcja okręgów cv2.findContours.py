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
    sciezkaDocelowa = "Detekcja okręgów cv2.findContours/1280 x 960/"

    tytul = os.listdir(sciezkaZrodlowa)


    for i in range(len(tytul)):
    #for i in [0, 20, 30, 40, 50, 60, 70]:
    #for i in [268]:
        print("Trwa przetwarzanie obrazu o indeksie i = " + str(i) + "/" + str(len(tytul)) + " (" + tytul[i])
        oryginal = cv2.imread(sciezkaZrodlowa + tytul[i], cv2.IMREAD_COLOR)
        minimalnyPromien = int(min(oryginal.shape[:2])/20)
        maksymalnyPromien = int(min(oryginal.shape[:2])/3)


        image = progowanie(oryginal)

        maska = np.ones((5, 5), np.uint8)
        image = cv2.dilate(image, maska, iterations=1)
        im2, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        imageWszystkieKontury = cv2.drawContours(oryginal.copy(), contours, -1, (0, 255, 0), 3)

        print("\tTrwa identyfikacja konturów okręgów...")
        konturyOkregow = []
        for contour in contours:
            """
            print("len(contour) = " + str(len(contour)))
            print(contour)
            print("len(contour[0]) = " + str(len(contour[0])))
            print(contour[0])
            print("len(contour[0, 0]) = " + str(len(contour[0, 0])))
            print(contour[0, 0])
            #print("len(contour[0, 0, 0]) = " + str(len(contour[0, 0, 0])))
            print(contour[0, 0, 0])
            """
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


        """
        print("\tPrzygotowywanie wyswietlenia obrazow w panelu bocznym...")
        plt.show()
        print("\tPomyslnie wyswietlono.\n\n")
        """



        print("\tTrwa zapisywanie do pliku jpg...")
        fig.savefig(sciezkaDocelowa + tytul[i])
        print("\tPomyslnie zapisano do pliku jpg.\n\n")



        #Zwolnienie pamięci zajmowanej przez duże macierze z obrazami
        print("\tTrwa zwolnienie pamięci...")
        plt.close()
        gc.collect()
        print("\tPomyslnie zwoniono pamięć.\n")



main()