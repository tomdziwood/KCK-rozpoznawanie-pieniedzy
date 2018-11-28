import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import gc



def progowanie(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    threshMean2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    threshMean2blurred = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    threshGauss2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    threshGauss2blurred = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    canny = cv2.Canny(gray, 50, 100)
    cannyBlurred = cv2.Canny(blurred, 50, 100)

    return [[gray, blurred, threshMean2, threshMean2blurred, threshGauss2, threshGauss2blurred, canny, cannyBlurred], ["Oryginalny obraz", "Rozmyty obraz - cv2.GaussianBlur",
                                                                                                              "ADAPTIVE_THRESH_MEAN_C, blockSize=31,C=10", "GaussianBlur + cv2.ADAPTIVE_THRESH_MEAN_C, blockSize=11,C=2",
                                                                                                              "ADAPTIVE_THRESH_GAUSSIAN_C, blockSize=31,C=10", "GaussianBlur + ADAPTIVE_THRESH_GAUSSIAN_C, blockSize=31,C=10",
                                                                                                              "filtr Canny", "GaussianBlur + filtr Canny"]]



def main():
    sciezkaZrodlowa = "../zdjecia/"
    sciezkaDocelowa = "Testowanie blur/"

    kategoriaZdjec = ["banknoty - latwe/", "banknoty - srednie/", "banknoty - trudne/",
                      "monety - latwe/", "monety - srednie/", "monety - trudne/",
                      "challenge/"]


    for iKategoriaZdjec in range(len(kategoriaZdjec)):
        tytul = os.listdir(sciezkaZrodlowa + kategoriaZdjec[iKategoriaZdjec])

        for iTytul in range(len(tytul)):
            print("\nTrwa przetwarzanie obrazu " + str(iTytul) + "/" + str(len(tytul)) + " (" + kategoriaZdjec[iKategoriaZdjec] + tytul[iTytul] + ")")
            oryginal = cv2.imread(sciezkaZrodlowa + kategoriaZdjec[iKategoriaZdjec] + tytul[iTytul], cv2.IMREAD_COLOR)


            print("\tTrwa progowanie obrazu...")
            images, tytulyWykresow = progowanie(oryginal)


            print("\tTrwa rysowanie wykresów...")
            fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(12, 20.5), subplot_kw={'xticks': [], 'yticks': []})
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)
            for j in range(8):
                ax[int(j / 2), j % 2].set_title(tytulyWykresow[j])
                ax[int(j / 2), j % 2].imshow(images[j], cmap='gray', aspect='equal', interpolation='bilinear')


            print("\tTrwa zapisywanie do pliku jpg...")
            fig.savefig(sciezkaDocelowa + kategoriaZdjec[iKategoriaZdjec] + tytul[iTytul])

            """
            print("\tPrzygotowywanie wyświetlenia obrazów w panelu bocznym...")
            plt.show()
            """

            # Zwolnienie pamięci zajmowanej przez duże macierze z obrazami
            print("\tTrwa zwolnienie pamięci...")
            plt.close()
            gc.collect()



main()