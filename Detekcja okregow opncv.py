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

    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    """
    fig, ax = plt.subplots()
    ax.imshow(blurred, cmap='gray')
    plt.show()
    """


    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)


    print("\tKoniec wykrywania konturow.")
    return thresh



def main():
    sciezkaZrodlowa = "zdjecia/1280 x 960/"
    sciezkaDocelowa = "okregi/1280 x 960/"

    tytul = os.listdir(sciezkaZrodlowa)


    #for i in range(len(tytul)):
    for i in [0, 10, 20, 30, 40, 50, 60]:
        print('Trwa przetwarzanie obrazu o indeksie i =', i, " (" + tytul[i])
        oryginal = cv2.imread(sciezkaZrodlowa + tytul[i], cv2.IMREAD_COLOR)


        print("\tTrwa wykrywanie monet...")
        image = wykrywanieKonturow(oryginal)
        """
        print("type(images) = " + str(type(images)))
        print("len(images) = " + str(len(images)))
        print("type(tytulyWykresow) = " + str(type(tytulyWykresow)))
        print("len(tytulyWykresow) = " + str(len(tytulyWykresow)))
        """

        print("\t\nTrwa rysowanie wykresow...")
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.5), subplot_kw={'xticks': [], 'yticks': []})
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)
        ax[0].imshow(cv2.cvtColor(oryginal, cv2.COLOR_BGR2RGB), aspect='equal', interpolation='bilinear')
        ax[1].imshow(image, cmap='gray', aspect='equal', interpolation='bilinear')
        print("\tPrzygotowywanie wyswietlenia obrazow w panelu bocznym...")
        plt.show()
        print("\tPomyslnie wyswietlono.\n\n")


        print("\tTrwa wykonywanie cv2.HoughCircles()...")
        minimalnyPromien = int(min(image.shape)/20)
        maksymalnyPromien = int(min(image.shape)/3)
        minimalnyDystans = int(1.5 * minimalnyPromien)
        parametr2 = 40
        while(parametr2 > 0):
            print("Trwa proba parametr2 = " + str(parametr2) + "...")
            circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, minDist=minimalnyDystans, param1=50, param2=parametr2, minRadius=minimalnyPromien, maxRadius=maksymalnyPromien)
            if(not(circles is None) and (len(circles[0]) >= 10)):
                print("funkcja cv2.HoughCircles() zatrzymana dla parametr2 = " + str(parametr2))
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
        print("\tPomyslnie wyswietlono.\n\n")


        """
        print("\tTrwa zapisywanie do pliku jpg...")
        fig.savefig(sciezkaDocelowa + tytul[i])
        print("\tPomyslnie zapisano do pliku jpg.\n\n")
        """



main()