import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import gc



def progowanie(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    canny = cv2.Canny(blurred, 50, 100)

    return canny



def zatwierdzenieKsztaltuProstokata(funkcjaLiniowa):
    for i in range(4):
        iloczyn = np.abs(funkcjaLiniowa[i][0] * funkcjaLiniowa[(i + 1) % 4][0])

        if((iloczyn < 0.5) or (iloczyn > 2)):
            return False
    return True



def main():
    sciezkaZrodlowa = "zdjecia/"
    sciezkaDocelowa = "Detekcja prostokatow/"

    kategoriaZdjec = ["banknoty - latwe/", "banknoty - srednie/", "banknoty - trudne/",
                      "monety - latwe/", "monety - srednie/", "monety - trudne/",
                      "challenge/"]


    for iKategoriaZdjec in range(len(kategoriaZdjec)):
        tytul = os.listdir(sciezkaZrodlowa + kategoriaZdjec[iKategoriaZdjec])

        for iTytul in range(len(tytul)):
            print("\nTrwa przetwarzanie obrazu " + str(iTytul) + "/" + str(len(tytul)) + " (" + kategoriaZdjec[iKategoriaZdjec] + tytul[iTytul] + ")")
            oryginal = cv2.imread(sciezkaZrodlowa + kategoriaZdjec[iKategoriaZdjec] + tytul[iTytul], cv2.IMREAD_COLOR)
            minimalnyBok = int(min(oryginal.shape[:2])/4)


            print("\tTrwa progowanie obrazu...")
            image = progowanie(oryginal)


            print("\tTrwa morfologia konturów...")
            maska = np.ones((5, 5), np.uint8)
            image = cv2.dilate(image, maska, iterations=1)


            print("\tTrwa obliczanie konturów..")
            im2, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            imageWszystkieKontury = cv2.drawContours(oryginal.copy(), contours, -1, (0, 255, 0), 3)

            print("\tTrwa identyfikacja konturów prostokątów...")
            konturyProstokatow = []
            for contour in contours:

                x = contour[:, 0, 0]
                y = contour[:, 0, 1]

                xmin = min(x)
                xmax = max(x)
                ymin = min(y)
                ymax = max(y)
                dx = xmax - xmin
                dy = ymax - ymin

                if(dx >= minimalnyBok) and (dy >= minimalnyBok):
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



            imageKonturyProstokatow = cv2.drawContours(oryginal.copy(), konturyProstokatow, -1, (255, 0, 0), 7)

            print("\tTrwa rysowanie wykresow...")
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 5), subplot_kw={'xticks': [], 'yticks': []})
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)
            ax[0].imshow(cv2.cvtColor(oryginal, cv2.COLOR_BGR2RGB), aspect='equal', interpolation='bilinear')
            ax[1].set_title("Ilosc konturów: " + str(len(contours)))
            ax[1].imshow(cv2.cvtColor(imageWszystkieKontury, cv2.COLOR_BGR2RGB), aspect='equal', interpolation='bilinear')
            ax[2].set_title("Ilosc prostokątów: " + str(len(konturyProstokatow)))
            ax[2].imshow(cv2.cvtColor(imageKonturyProstokatow, cv2.COLOR_BGR2RGB), aspect='equal', interpolation='bilinear')


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