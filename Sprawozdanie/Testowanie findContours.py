import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import gc



def progowanie(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    threshMean = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    threshGauss = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    canny = cv2.Canny(blurred, 50, 100)

    return [threshMean, threshGauss, canny]



def main():
    sciezkaZrodlowa = "../zdjecia/"
    sciezkaDocelowa = "Testowanie findContours/"

    kategoriaZdjec = ["banknoty - latwe/", "banknoty - srednie/", "banknoty - trudne/",
                      "monety - latwe/", "monety - srednie/", "monety - trudne/",
                      "challenge/"]


    for iKategoriaZdjec in range(len(kategoriaZdjec)):
        tytul = os.listdir(sciezkaZrodlowa + kategoriaZdjec[iKategoriaZdjec])

        for iTytul in range(len(tytul)):
            print("\nTrwa przetwarzanie obrazu " + str(iTytul) + "/" + str(len(tytul)) + " (" + kategoriaZdjec[iKategoriaZdjec] + tytul[iTytul] + ")")
            oryginal = cv2.imread(sciezkaZrodlowa + kategoriaZdjec[iKategoriaZdjec] + tytul[iTytul], cv2.IMREAD_COLOR)


            print("\tTrwa progowanie obrazu...")
            obrazy = progowanie(oryginal)


            print("\tTrwa morfologia konturow...")
            maska = np.ones((5, 5), np.uint8)
            for j in range(3):
                obrazy[j] = cv2.dilate(obrazy[j], maska, iterations=1)


            print("\tTrwa obliczanie konturow..")
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
            """

            print("\tTrwa zapisywanie do pliku jpg...")
            fig.savefig(sciezkaDocelowa + kategoriaZdjec[iKategoriaZdjec] + tytul[iTytul])

            # Zwolnienie pamięci zajmowanej przez duże macierze z obrazami
            print("\tTrwa zwolnienie pamięci...")
            plt.close()
            gc.collect()



main()