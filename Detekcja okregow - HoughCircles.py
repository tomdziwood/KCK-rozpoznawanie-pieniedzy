import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import gc



def wykrywanieKonturow(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)

    return thresh



def main():
    sciezkaZrodlowa = "zdjecia/"

    kategoriaZdjec = ["banknoty - latwe/", "banknoty - srednie/", "banknoty - trudne/",
                      "monety - latwe/", "monety - srednie/", "monety - trudne/",
                      "challenge/"]


    for iKategoriaZdjec in range(len(kategoriaZdjec)):
        tytul = os.listdir(sciezkaZrodlowa + kategoriaZdjec[iKategoriaZdjec])

        for iTytul in range(len(tytul)):
            print("\nTrwa przetwarzanie obrazu " + str(iTytul) + "/" + str(len(tytul)) + " (" + kategoriaZdjec[iKategoriaZdjec] + tytul[iTytul] + ")")
            oryginal = cv2.imread(sciezkaZrodlowa + kategoriaZdjec[iKategoriaZdjec] + tytul[iTytul], cv2.IMREAD_COLOR)


            print("\tTrwa progowanie obrazu...")
            image = wykrywanieKonturow(oryginal)


            print("\tTrwa rysowanie wykresów...")
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.5), subplot_kw={'xticks': [], 'yticks': []})
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)
            ax[0].imshow(cv2.cvtColor(oryginal, cv2.COLOR_BGR2RGB), aspect='equal', interpolation='bilinear')
            ax[1].imshow(image, cmap='gray', aspect='equal', interpolation='bilinear')


            print("\tPrzygotowywanie wyswietlenia obrazow w panelu bocznym...")
            plt.show()



            print("\tTrwa wykonywanie cv2.HoughCircles()...")
            minimalnyPromien = int(min(image.shape)/20)
            maksymalnyPromien = int(min(image.shape)/3)
            minimalnyDystans = int(1.5 * minimalnyPromien)
            parametr2 = 60
            while(parametr2 > 0):
                print("\tTrwa proba parametr2 = " + str(parametr2) + "...")
                circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, minDist=minimalnyDystans, param1=50, param2=parametr2, minRadius=minimalnyPromien, maxRadius=maksymalnyPromien)
                if(not(circles is None) and (len(circles[0]) >= 10)):
                    print("\tfunkcja cv2.HoughCircles() zatrzymana dla parametr2 = " + str(parametr2))
                    break
                parametr2 -= 1

            wynik = np.copy(oryginal)
            circles = np.uint16(np.around(circles))
            print("\tLiczba znalezionych okregow: " + str(len(circles[0])))
            print("\tTrwa nanoszenie okregow...")
            for j in circles[0, :]:
                # rysowanie okregu
                cv2.circle(wynik, (j[0], j[1]), j[2], (0, 255, 0), 2)
                # draw srodka okregu
                cv2.circle(wynik, (j[0], j[1]), 2, (0, 0, 255), 3)

            print("\tTrwa rysowanie wykresow...")
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.5), subplot_kw={'xticks': [], 'yticks': []})
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)
            ax[0].imshow(cv2.cvtColor(oryginal, cv2.COLOR_BGR2RGB), aspect='equal', interpolation='bilinear')
            ax[1].imshow(cv2.cvtColor(wynik, cv2.COLOR_BGR2RGB), cmap='gray', aspect='equal', interpolation='bilinear')

            print("\tPrzygotowywanie wyswietlenia obrazow w panelu bocznym...")
            plt.show()



            """
            print("\tTrwa zapisywanie do pliku jpg...")
            fig.savefig(sciezkaDocelowa + kategoriaZdjec[iKategoriaZdjec] + tytul[iTytul])
            """

            #Zwolnienie pamięci zajmowanej przez duże macierze z obrazami
            print("\tTrwa zwolnienie pamięci...")
            plt.close()
            gc.collect()



main()