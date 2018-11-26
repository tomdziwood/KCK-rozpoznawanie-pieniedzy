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



def zatwierdzenieKsztaltuProstokata(funkcjaLiniowa):
    for i in range(4):
        iloczyn = np.abs(funkcjaLiniowa[i][0] * funkcjaLiniowa[(i + 1) % 4][0])

        if((iloczyn < 0.5) or (iloczyn > 2)):
            return False
    return True



def main():
    sciezkaZrodlowa = "zdjecia/banknoty - latwe/"
    sciezkaDocelowa = "Detekcja prostokatow cv2.findContours/banknoty - latwe/"

    tytul = os.listdir(sciezkaZrodlowa)


    for i in range(len(tytul)):
    #for i in [0, 20, 30, 40, 50, 60, 70]:
    #for i in [17]:
        print("Trwa przetwarzanie obrazu o indeksie i = " + str(i) + "/" + str(len(tytul)) + " (" + tytul[i])
        oryginal = cv2.imread(sciezkaZrodlowa + tytul[i], cv2.IMREAD_COLOR)
        minimalnyBok = int(min(oryginal.shape[:2])/4)
        maksymalnyBok = int(max(oryginal.shape[:2]))


        image = progowanie(oryginal)

        maska = np.ones((5, 5), np.uint8)
        image = cv2.dilate(image, maska, iterations=1)
        im2, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        imageWszystkieKontury = cv2.drawContours(oryginal.copy(), contours, -1, (0, 255, 0), 3)

        print("\tTrwa identyfikacja konturów okręgów...")
        konturyProstokatow = []
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

            if(dx >= minimalnyBok) and (dy >= minimalnyBok):
                print("\tTrwa ustalanie położenia prostokąta...")
                wierzcholki = [[x[0], y[0]], [x[0], y[0]], [x[0], y[0]], [x[0], y[0]]]
                for iter in range(1, len(x)):
                    if(wierzcholki[0][1] > y[iter]):
                        wierzcholki[0] = [x[iter], y[iter]]
                    if(wierzcholki[1][0] < x[iter]):
                        wierzcholki[1] = [x[iter], y[iter]]
                    if(wierzcholki[2][1] < y[iter]):
                        wierzcholki[2] = [x[iter], y[iter]]
                    if(wierzcholki[3][0] > x[iter]):
                        wierzcholki[3] = [x[iter], y[iter]]

                funkcjaLiniowa = []
                for iter in range(4):
                    wspA = (wierzcholki[iter][1] - wierzcholki[(iter + 1) % 4][1]) / (wierzcholki[iter][0] - wierzcholki[(iter + 1) % 4][0])
                    wspB = wierzcholki[iter][1] - wspA * wierzcholki[iter][0]
                    funkcjaLiniowa.append([wspA, wspB])

                if(zatwierdzenieKsztaltuProstokata(funkcjaLiniowa) == True):
                    konturyProstokatow.append(np.array(wierzcholki))

        print("\tIdentyfikacja konturów okręgów zakończona.")


        imageKonturyProstokatow = cv2.drawContours(oryginal.copy(), konturyProstokatow, -1, (0, 255, 0), 3)

        print("\tTrwa rysowanie wykresow...")
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 5), subplot_kw={'xticks': [], 'yticks': []})
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)
        ax[0].imshow(cv2.cvtColor(oryginal, cv2.COLOR_BGR2RGB), aspect='equal', interpolation='bilinear')
        ax[1].set_title("Ilosc konturów: " + str(len(contours)))
        ax[1].imshow(cv2.cvtColor(imageWszystkieKontury, cv2.COLOR_BGR2RGB), aspect='equal', interpolation='bilinear')
        ax[2].set_title("Ilosc konturów: " + str(len(konturyProstokatow)))
        ax[2].imshow(cv2.cvtColor(imageKonturyProstokatow, cv2.COLOR_BGR2RGB), aspect='equal', interpolation='bilinear')


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