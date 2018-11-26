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

    return canny



def zatwierdzenieKsztaltuProstokata(funkcjaLiniowa):
    for i in range(4):
        iloczyn = np.abs(funkcjaLiniowa[i][0] * funkcjaLiniowa[(i + 1) % 4][0])

        if((iloczyn < 0.5) or (iloczyn > 2)):
            return False
    return True



def main():
    sciezkaZrodlowa = "zdjecia/banknoty - latwe/"
    sciezkaDocelowa = "Rozpoznanie nominalu banknotow - histogram/banknoty - latwe/"
    sciezkaWzorcow = "wzorce/"
    nominalyBanknotow = ["10zl", "20zl"]

    tytul = os.listdir(sciezkaZrodlowa)

    # przejscie po folderach z nominałami monet
    histogramyBanknotow = []
    for iNominalyBanknotow in range(len(nominalyBanknotow)):
        sciezkaWzorca = sciezkaWzorcow + str(nominalyBanknotow[iNominalyBanknotow]) + '/'
        tytulWzorca = os.listdir(sciezkaWzorca)

        # przejscie po wzorcach danego nominału
        #for iTytulWzorcaMonety in range(min(len(tytulWzorca), 10)):
        for iTytulWzorcaBanknota in range(len(tytulWzorca)):
            wzorzec = cv2.imread(sciezkaWzorca + tytulWzorca[iTytulWzorcaBanknota], cv2.IMREAD_COLOR)
            print("\t\tTrwa szykowanie wzorca... (" + str(iTytulWzorcaBanknota) + "/" + str(len(tytulWzorca)) + ") " + sciezkaWzorca + tytulWzorca[iTytulWzorcaBanknota])

            histoWzorzec = [0, 0, 0]
            for iter in range(3):
                dane = wzorzec[:, :, iter]
                histoWzorzec[iter], xSmieci = np.histogram(dane[dane != 255], range(0, 256), density=True)

            histogramyBanknotow.append([histoWzorzec, nominalyBanknotow[iNominalyBanknotow], sciezkaWzorca + tytulWzorca[iTytulWzorcaBanknota]])



    for i in range(len(tytul)):
    #for i in [42]:
        print("Trwa przetwarzanie obrazu o indeksie i = " + str(i) + "/" + str(len(tytul)) + " (" + tytul[i])
        oryginal = cv2.imread(sciezkaZrodlowa + tytul[i], cv2.IMREAD_COLOR)
        minimalnyBok = int(min(oryginal.shape[:2]) / 4)
        maksymalnyBok = int(max(oryginal.shape[:2]))

        image = progowanie(oryginal)

        maska = np.ones((5, 5), np.uint8)
        image = cv2.dilate(image, maska, iterations=1)
        im2, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        imageWszystkieKontury = cv2.drawContours(oryginal.copy(), contours, -1, (0, 255, 0), 3)
        wynikImage = oryginal.copy()

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

            if (dx >= minimalnyBok) and (dy >= minimalnyBok):
                print("\tTrwa ustalanie położenia prostokąta...")
                xmean = np.mean(x)
                ymean = np.mean(y)
                wierzcholki = [[x[0], y[0]], [x[0], y[0]], [x[0], y[0]], [x[0], y[0]]]
                for iter in range(1, len(x)):
                    if (wierzcholki[0][1] > y[iter]):
                        wierzcholki[0] = [x[iter], y[iter]]
                    if (wierzcholki[1][0] < x[iter]):
                        wierzcholki[1] = [x[iter], y[iter]]
                    if (wierzcholki[2][1] < y[iter]):
                        wierzcholki[2] = [x[iter], y[iter]]
                    if (wierzcholki[3][0] > x[iter]):
                        wierzcholki[3] = [x[iter], y[iter]]

                funkcjaLiniowa = []
                for iter in range(4):
                    wspA = (wierzcholki[iter][1] - wierzcholki[(iter + 1) % 4][1]) / (
                            wierzcholki[iter][0] - wierzcholki[(iter + 1) % 4][0])
                    wspB = wierzcholki[iter][1] - wspA * wierzcholki[iter][0]
                    funkcjaLiniowa.append([wspA, wspB])

                if (zatwierdzenieKsztaltuProstokata(funkcjaLiniowa) == True):
                    konturyProstokatow.append(np.array(wierzcholki))


                    wycinek = np.copy(oryginal)

                    print("\tTrwa szykowanie banknota...")
                    xlen = wycinek.shape[0]
                    ylen = wycinek.shape[1]
                    xWspolrzedna = np.zeros((xlen, ylen))
                    yWspolrzedna = np.zeros((xlen, ylen))

                    for ix in range(xlen):
                        xWspolrzedna[ix] = np.arange(ylen)
                        yWspolrzedna[ix] = np.ones(ylen) * ix

                    print("Trwa obcinanie obrazu...")
                    wycinek[yWspolrzedna < xWspolrzedna * funkcjaLiniowa[0][0] + funkcjaLiniowa[0][1]] = [255, 255, 255]
                    wycinek[yWspolrzedna > xWspolrzedna * funkcjaLiniowa[1][0] + funkcjaLiniowa[1][1]] = [255, 255, 255]
                    wycinek[yWspolrzedna > xWspolrzedna * funkcjaLiniowa[2][0] + funkcjaLiniowa[2][1]] = [255, 255, 255]
                    wycinek[yWspolrzedna < xWspolrzedna * funkcjaLiniowa[3][0] + funkcjaLiniowa[3][1]] = [255, 255, 255]




                    print("\tTrwa szykowanie histogramu monety...")
                    histoWycinek = [0, 0, 0]
                    for iter in range(3):
                        dane = wycinek[:, :, iter]
                        histoWycinek[iter], xSmieci = np.histogram(dane[dane != 255], range(0, 256), density=True)


                    wynikiStatystyk = []
                    for [histoWzorzec, nominalBanknotu, kompletnaSciezkaWzorca] in histogramyBanknotow:
                        roznicaHistogramow = 0
                        for iter in range(3):
                            roznicaHistogramow += np.sum(np.abs(histoWycinek[iter] - histoWzorzec[iter]))
                        wynikiStatystyk.append([roznicaHistogramow, nominalBanknotu, kompletnaSciezkaWzorca])


                    wynikiStatystyk.sort(key=lambda x: x[0])
                    #cv2.putText(wynikImage, wynikiStatystyk[0][1], (int(xmean), int(ymean)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)

                    """
                    for wynikStatystyk in wynikiStatystyk:
                        print(wynikStatystyk)
                    """



                    npWyniki = np.array(wynikiStatystyk)
                    jakosc = npWyniki[:, 0]
                    jakosc = jakosc.astype(np.float)
                    ileKandydatow = sum(jakosc < jakosc[0] + 0.04)

                    """
                    print("npWyniki:")
                    print(npWyniki)
                    """

                    nazwaNominalu = npWyniki[:ileKandydatow, 1]
                    """
                    print("jakosc:")
                    print(jakosc)
                    print("nazwaNominalu:")
                    print(nazwaNominalu)
                    """

                    npUnique = np.unique(nazwaNominalu, return_counts=True)
                    """
                    print("npUnique: ")
                    print(npUnique)
                    print("unique = ")
                    print(npUnique[0][:])
                    print("count = ")
                    print(npUnique[1][:])
                    """

                    zestawienie = []
                    for iter in range(len(npUnique[0])):
                        zestawienie.append([npUnique[0][iter], npUnique[1][iter]])

                    """
                    print("zestawienie = ")
                    print(zestawienie)
                    """
                    zestawienie.sort(key=lambda x: x[1], reverse=True)

                    """
                    print("zestawienie posortowane = ")
                    print(zestawienie)
                    """

                    cv2.putText(wynikImage, zestawienie[0][0], (int(xmean), int(ymean - 25)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
                    cv2.putText(wynikImage, zestawienie[0][0], (int(xmean), int(ymean + 25)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)
                    """
                    if(len(zestawienie) == 0):
                        cv2.putText(wynikImage, "???", (int(xmean), int(ymean + 50)),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
                    else:
                        cv2.putText(wynikImage, zestawienie[0][0], (int(xmean), int(ymean + 50)),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
                    """



        print("\tIdentyfikacja konturów okręgów zakończona.")


        imageKonturyProstokatow = cv2.drawContours(wynikImage.copy(), konturyProstokatow, -1, (0, 255, 0), 3)

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
        #fig.savefig(sciezkaDocelowa + tytul[i][:-4] + "_test.jpg")
        print("\tPomyslnie zapisano do pliku jpg.\n\n")


        #Zwolnienie pamięci zajmowanej przez duże macierze z obrazami
        print("\tTrwa zwolnienie pamięci...")
        plt.close()
        gc.collect()
        print("\tPomyslnie zwoniono pamięć.\n")



main()