import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import gc



def progowanie(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)

    canny = cv2.Canny(blurred, 50, 100)

    return [thresh, thresh.copy(), canny]


def morfologiczneTransformacje(image, oryginal):
    maska = np.ones((5, 5), np.uint8)
    dyl1 = cv2.dilate(image, maska, iterations=1)
    dyl2 = cv2.dilate(image, maska, iterations=2)
    er1 = cv2.erode(image, maska, iterations=1)
    er2 = cv2.erode(image, maska, iterations=1)
    er1dyl1 = cv2.dilate(er1, maska, iterations=1)
    er1dyl2 = cv2.dilate(er1dyl1, maska, iterations=1)
    er1dyl3 = cv2.dilate(er1dyl2, maska, iterations=1)

    print("\tTrwa rysowanie wykresów morfologii...")
    fig, ax = plt.subplots(nrows=8, ncols=2, figsize=(12, 36), subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)
    obrazy = [image, dyl1, dyl2, er1, er2, er1dyl1, er1dyl2, er1dyl3]
    for j in range(8):
        im2, contours, hierarchy = cv2.findContours(obrazy[j], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        wynik = cv2.drawContours(oryginal.copy(), contours, -1, (0, 255, 0), 3)
        ax[j, 0].imshow(obrazy[j], cmap='gray', aspect='equal', interpolation='bilinear')
        ax[j, 1].set_title("Ilość konturów: " + str(len(contours)))
        ax[j, 1].imshow(cv2.cvtColor(wynik, cv2.COLOR_BGR2RGB), aspect='equal', interpolation='bilinear')

    print("\tPrzygotowywanie wyświetlenia obrazów w panelu bocznym...")
    plt.show()



def main():
    sciezkaZrodlowa = "zdjecia/"
    sciezkaDocelowa = "Detekcja konturów cv2.findContours/"

    kategoriaZdjec = ["banknoty - latwe/", "banknoty - srednie/", "banknoty - trudne/",
                      "monety - latwe/", "monety - srednie/", "monety - trudne/",
                      "challenge/"]

    for iKategoriaZdjec in range(len(kategoriaZdjec)):
        tytul = os.listdir(sciezkaZrodlowa + kategoriaZdjec[iKategoriaZdjec])

        for iTytul in range(len(tytul)):
            print("\nTrwa przetwarzanie obrazu " + str(iTytul) + "/" + str(len(tytul)) + " (" + kategoriaZdjec[iKategoriaZdjec] + tytul[iTytul] + ")")
            oryginal = cv2.imread(sciezkaZrodlowa + kategoriaZdjec[iKategoriaZdjec] + tytul[iTytul], cv2.IMREAD_COLOR)


            print("\tTrwa progowanie obrazu...")
            image = progowanie(oryginal)


            #morfologiczneTransformacje(image, oryginal)


            contours = [0, 0, 0]
            wynik = [0, 0, 0]
            metoda = ["Threshold", "Threshold + dilate", "Canny + dilate"]
            maska = np.ones((5, 5), np.uint8)
            for j in range(3):
                if j > 0:
                    image[j] = cv2.dilate(image[j], maska, iterations=1)
                im2, contours[j], hierarchy = cv2.findContours(image[j], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                wynik[j] = cv2.drawContours(oryginal.copy(), contours[j], -1, (0, 255, 0), 3)

            print("\tTrwa rysowanie wykresów...")
            fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(18, 15), subplot_kw={'xticks': [], 'yticks': []})
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)
            for j in range(3):
                ax[j, 0].imshow(cv2.cvtColor(oryginal, cv2.COLOR_BGR2RGB), aspect='equal', interpolation='bilinear')
                ax[j, 1].imshow(image[j], cmap='gray', aspect='equal', interpolation='bilinear')
                ax[j, 2].set_title("Ilość konturów ( " + metoda[j] + "): " + str(len(contours[j])))
                ax[j, 2].imshow(cv2.cvtColor(wynik[j], cv2.COLOR_BGR2RGB), aspect='equal', interpolation='bilinear')

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