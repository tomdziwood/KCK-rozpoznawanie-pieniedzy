import cv2
from matplotlib import pyplot as plt
import numpy as np
import os



def wykrywanieKonturow(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)[1]
    threshMean1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    threshGauss1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    threshMean2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 10)
    threshGauss2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
    threshMean3 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 20)
    threshGauss3 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 20)

    return [[gray, thresh, threshMean1, threshGauss1, threshMean2, threshGauss2, threshMean3, threshGauss3], ["Oryginalny obraz", "Globalne progowanie (127)",
                                                                                                              "Adaptacyjne progowanie (średnia), blockSize=11,C=2", "Adaptacyjne progowanie (Gauss), blockSize=11,C=2",
                                                                                                              "Adaptacyjne progowanie (średnia), blockSize=31,C=10", "Adaptacyjne progowanie (Gauss), blockSize=31,C=10",
                                                                                                              "Adaptacyjne progowanie (średnia), blockSize=51,C=20", "Adaptacyjne progowanie (Gauss), blockSize=51,C=20"]]



def main():
    sciezkaZrodlowa = "zdjecia/"
    sciezkaDocelowa = "Progowanie obrazow/"

    kategoriaZdjec = ["banknoty - latwe/", "banknoty - srednie/", "banknoty - trudne/",
                      "monety - latwe/", "monety - srednie/", "monety - trudne/",
                      "challenge/"]

    for iKategoriaZdjec in range(len(kategoriaZdjec)):
        tytul = os.listdir(sciezkaZrodlowa + kategoriaZdjec[iKategoriaZdjec])

        for iTytul in range(len(tytul)):
            print("\nTrwa przetwarzanie obrazu " + str(iTytul) + "/" + str(len(tytul)) + " (" + tytul[iTytul] + ")")
            oryginal = cv2.imread(sciezkaZrodlowa + kategoriaZdjec[iKategoriaZdjec] + tytul[iTytul], cv2.IMREAD_COLOR)


            print("\tTrwa progowanie obrazu...")
            images, tytulyWykresow = wykrywanieKonturow(oryginal)


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



main()