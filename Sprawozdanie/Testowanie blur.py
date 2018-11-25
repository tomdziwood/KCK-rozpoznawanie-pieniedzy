import cv2
from matplotlib import pyplot as plt
import numpy as np
import os



def wykrywanieKonturow(image):
    print("\tTrwa wykrywanie konturow...")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    """
    fig, ax = plt.subplots()
    ax.imshow(gray, cmap='gray')
    plt.show()
    """

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    """
    fig, ax = plt.subplots()
    ax.imshow(blurred, cmap='gray')
    plt.show()
    """


    #blockSize = 51
    #C = 20

    threshMean2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    threshMean2blurred = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    threshGauss2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    threshGauss2blurred = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    canny = cv2.Canny(gray, 50, 100)
    cannyBlurred = cv2.Canny(blurred, 50, 100)


    print("\tKoniec wykrywania konturow.")
    return [[gray, blurred, threshMean2, threshMean2blurred, threshGauss2, threshGauss2blurred, canny, cannyBlurred], ["Oryginalny obraz", "Rozmyty obraz - cv2.GaussianBlur",
                                                                                                              "ADAPTIVE_THRESH_MEAN_C, blockSize=31,C=10", "GaussianBlur + cv2.ADAPTIVE_THRESH_MEAN_C, blockSize=11,C=2",
                                                                                                              "ADAPTIVE_THRESH_GAUSSIAN_C, blockSize=31,C=10", "GaussianBlur + ADAPTIVE_THRESH_GAUSSIAN_C, blockSize=31,C=10",
                                                                                                              "filtr Canny", "GaussianBlur + filtr Canny"]]



def main():
    sciezkaZrodlowa = "zdjecia/1280 x 960/"
    sciezkaDocelowa = "Testowanie blur/1280 x 960/"

    tytul = os.listdir(sciezkaZrodlowa)



    #for i in range(len(tytul)):
    for i in range(len(tytul)):
        print('Trwa przetwarzanie obrazu o indeksie i =', i, " (" + tytul[i])
        oryginal = cv2.imread(sciezkaZrodlowa + tytul[i], cv2.IMREAD_COLOR)


        print("\tTrwa wykrywanie monet...")
        images, tytulyWykresow = wykrywanieKonturow(oryginal)



        print("\t\nTrwa rysowanie wykresow...")
        fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(12, 20.5), subplot_kw={'xticks': [], 'yticks': []})
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)
        for j in range(8):
            ax[int(j / 2), j % 2].set_title(tytulyWykresow[j])
            ax[int(j / 2), j % 2].imshow(images[j], cmap='gray', aspect='equal', interpolation='bilinear')


        print("\tTrwa zapisywanie do pliku jpg...")
        fig.savefig(sciezkaDocelowa + tytul[i])
        print("\tPomyslnie zapisano do pliku jpg.\n\n")
        """
        print("\tPrzygotowywanie wyswietlenia obrazow w panelu bocznym...")
        plt.show()
        print("\tPomyslnie wyswietlono.\n\n")
        """



main()