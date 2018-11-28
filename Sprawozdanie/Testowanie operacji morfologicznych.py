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



def morfologia(image):
    maska = np.ones((5, 5), np.uint8)
    dyl1 = cv2.dilate(image, maska, iterations=1)
    dyl2 = cv2.dilate(image, maska, iterations=2)
    er1 = cv2.erode(image, maska, iterations=1)
    er2 = cv2.erode(image, maska, iterations=2)
    er1dyl1 = cv2.dilate(er1, maska, iterations=1)

    return [[image, dyl1, dyl2, er1dyl1, er1, er2], ["Obraz podstawowy", "Jednokrotna dylatacja", "Podwójna dylatacja",
                                                     "Jednokrotna erozja + jednokrotna dylatacja", "Jednokrotna erozja", "Podwójna erozja"]]



def main():
    sciezkaZrodlowa = "../zdjecia/"
    sciezkaDocelowa = "Testowanie operacji morfologicznych/"

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
            metody = ["cv2.adaptiveThreshold( adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C )", "cv2.adaptiveThreshold( adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C )", "cv2.Canny()"]


            for j in range(len(obrazy)):
                print("\tTrwa morfologia konturów...")
                images, tytulyWykresow = morfologia(obrazy[j])

                print("\tTrwa rysowanie wykresów... (" + str(j) + "/" + str(len(obrazy)) + ")")
                fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 10), subplot_kw={'xticks': [], 'yticks': []})
                fig.subplots_adjust(left=0, right=1, bottom=0, top=0.9, hspace=0.1, wspace=0.1)
                fig.suptitle(metody[j], fontsize=20)
                for k in range(6):
                    ax[int(k / 3), k % 3].set_title(tytulyWykresow[k])
                    ax[int(k / 3), k % 3].imshow(images[k], cmap='gray', aspect='equal', interpolation='bilinear')


                print("\tTrwa zapisywanie do pliku jpg...")
                fig.savefig(sciezkaDocelowa + kategoriaZdjec[iKategoriaZdjec] + tytul[iTytul][:-4] + "_" + str(j) + ".jpg")

                """
                print("\tPrzygotowywanie wyświetlenia obrazów w panelu bocznym...")
                plt.show()
                """

                # Zwolnienie pamięci zajmowanej przez duże macierze z obrazami
                print("\tTrwa zwolnienie pamięci...")
                plt.close()
                gc.collect()



main()