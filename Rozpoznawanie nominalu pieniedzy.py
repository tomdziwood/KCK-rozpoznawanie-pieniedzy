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
    sciezkaDocelowa = "Rozpoznawanie nominalu pieniedzy/"
    sciezkaWzorcow = "wzorce/"
    nominalyMonet = ["5gr", "50gr", "1zl", "2zl", "5zl"]
    nominalyBanknotow = ["10zl", "20zl"]

    kategoriaZdjec = ["banknoty - latwe/", "banknoty - srednie/", "banknoty - trudne/",
                      "monety - latwe/", "monety - srednie/", "monety - trudne/",
                      "challenge/"]

    # przejscie po folderach z nominałami monet
    histogramyMonet = []
    for iNominalyMonet in range(len(nominalyMonet)):
        sciezkaWzorca = sciezkaWzorcow + str(nominalyMonet[iNominalyMonet]) + '/'
        tytulWzorca = os.listdir(sciezkaWzorca)

        # przejscie po wzorcach danego nominału
        for iTytulWzorca in range(len(tytulWzorca)):
            wzorzec = cv2.imread(sciezkaWzorca + tytulWzorca[iTytulWzorca], cv2.IMREAD_COLOR)
            print("\t\tTrwa szykowanie wzorca... (" + str(iTytulWzorca) + "/" + str(
                len(tytulWzorca)) + ") " + sciezkaWzorca + tytulWzorca[iTytulWzorca])

            xlen = wzorzec.shape[0]
            ylen = wzorzec.shape[1]
            xWspolrzedna = np.zeros((xlen, ylen))
            yWspolrzedna = np.zeros((xlen, ylen))

            for ix in range(xlen):
                xWspolrzedna[ix] = np.ones(ylen) * ix
                yWspolrzedna[ix] = np.arange(ylen)

            promien = xlen / 2
            wzorzec[(xWspolrzedna - promien) ** 2 + (yWspolrzedna - promien) ** 2 > promien ** 2] = [255, 255, 255]

            histoWzorzec = [0, 0, 0]
            for iter in range(3):
                dane = wzorzec[:, :, iter]
                histoWzorzec[iter], xSmieci = np.histogram(dane[dane != 255], range(0, 256), density=True)

            histogramyMonet.append([histoWzorzec, nominalyMonet[iNominalyMonet], sciezkaWzorca + tytulWzorca[iTytulWzorca]])

    # przejscie po folderach z nominałami monet
    histogramyBanknotow = []
    for iNominalyBanknotow in range(len(nominalyBanknotow)):
        sciezkaWzorca = sciezkaWzorcow + str(nominalyBanknotow[iNominalyBanknotow]) + '/'
        tytulWzorca = os.listdir(sciezkaWzorca)

        # przejscie po wzorcach danego nominału
        for iTytulWzorca in range(len(tytulWzorca)):
            wzorzec = cv2.imread(sciezkaWzorca + tytulWzorca[iTytulWzorca], cv2.IMREAD_COLOR)
            print("\t\tTrwa szykowanie wzorca... (" + str(iTytulWzorca) + "/" + str(len(tytulWzorca)) + ") " + sciezkaWzorca + tytulWzorca[iTytulWzorca])

            histoWzorzec = [0, 0, 0]
            for iter in range(3):
                dane = wzorzec[:, :, iter]
                histoWzorzec[iter], xSmieci = np.histogram(dane[dane != 255], range(0, 256), density=True)

            histogramyBanknotow.append([histoWzorzec, nominalyBanknotow[iNominalyBanknotow],sciezkaWzorca + tytulWzorca[iTytulWzorca]])



    for iKategoriaZdjec in range(len(kategoriaZdjec)):
        tytul = os.listdir(sciezkaZrodlowa + kategoriaZdjec[iKategoriaZdjec])

        for iTytul in range(len(tytul)):
            print("\nTrwa przetwarzanie obrazu " + str(iTytul) + "/" + str(len(tytul)) + " (" + kategoriaZdjec[iKategoriaZdjec] + tytul[iTytul] + ")")
            oryginal = cv2.imread(sciezkaZrodlowa + kategoriaZdjec[iKategoriaZdjec] + tytul[iTytul], cv2.IMREAD_COLOR)
            minimalnyPromien = int(min(oryginal.shape[:2])/20)
            minimalnyBok = int(min(oryginal.shape[:2]) / 4)


            print("\tTrwa progowanie obrazu...")
            image = progowanie(oryginal)


            print("\tTrwa morfologia konturów...")
            maska = np.ones((5, 5), np.uint8)
            image = cv2.dilate(image, maska, iterations=1)


            print("\tTrwa obliczanie konturów..")
            im2, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            imageWszystkieKontury = cv2.drawContours(oryginal.copy(), contours, -1, (0, 255, 0), 3)
            wynikImage = oryginal.copy()

            print("\tTrwa identyfikacja konturów okręgów...")
            konturyOkregow = []
            for contour in contours:

                x = contour[:, 0, 0]
                y = contour[:, 0, 1]

                xmin = min(x)
                xmax = max(x)
                ymin = min(y)
                ymax = max(y)
                dx = xmax - xmin
                dy = ymax - ymin

                if(dx >= minimalnyPromien) and (dy >= minimalnyPromien):
                    xmean = np.mean(x)
                    ymean = np.mean(y)
                    odlegloscOdSrodka = ((xmean - x) ** 2 + (ymean - y) ** 2) ** (0.5)
                    odlegloscOdSrodkaMean = np.mean(odlegloscOdSrodka)
                    odlegloscOdSrodkaStd = np.std(odlegloscOdSrodka)

                    if(odlegloscOdSrodkaStd < 0.2 * odlegloscOdSrodkaMean):
                        konturyOkregow.append(contour)

                        print("\t\tTrwa identyfikacja nominału monety...")
                        wycinek = np.copy(oryginal[int(ymean - odlegloscOdSrodkaMean): int(ymean + odlegloscOdSrodkaMean),int(xmean - odlegloscOdSrodkaMean): int(xmean + odlegloscOdSrodkaMean)])

                        xlen = wycinek.shape[0]
                        ylen = wycinek.shape[1]
                        xWspolrzedna = np.zeros((xlen, ylen))
                        yWspolrzedna = np.zeros((xlen, ylen))

                        for ix in range(xlen):
                            xWspolrzedna[ix] = np.ones(ylen) * ix
                            yWspolrzedna[ix] = np.arange(ylen)

                        promien = xlen / 2
                        wycinek[(xWspolrzedna - promien) ** 2 + (yWspolrzedna - promien) ** 2 > promien ** 2] = [255, 255, 255]


                        histoWycinek = [0, 0, 0]
                        for iter in range(3):
                            dane = wycinek[:, :, iter]
                            histoWycinek[iter], xSmieci = np.histogram(dane[dane != 255], range(0, 256), density=True)


                        wynikiStatystyk = []
                        for [histoWzorzec, nominalMonety, kompletnaSciezkaWzorca] in histogramyMonet:
                            roznicaHistogramow = 0
                            for iter in range(3):
                                roznicaHistogramow += np.sum(np.abs(histoWycinek[iter] - histoWzorzec[iter]))
                            wynikiStatystyk.append([roznicaHistogramow, nominalMonety, kompletnaSciezkaWzorca])


                        wynikiStatystyk.sort(key=lambda x: x[0])

                        npWyniki = np.array(wynikiStatystyk)
                        jakosc = npWyniki[:, 0]
                        jakosc = jakosc.astype(np.float)
                        ileKandydatow = sum(jakosc < jakosc[0] + 0.15)

                        nazwaNominalu = npWyniki[:ileKandydatow, 1]

                        npUnique = np.unique(nazwaNominalu, return_counts=True)

                        zestawienie = []
                        for iter in range(len(npUnique[0])):
                            zestawienie.append([npUnique[0][iter], npUnique[1][iter]])

                        zestawienie.sort(key=lambda x: x[1], reverse=True)

                        if(int(ymean - 25) >= 0):
                            cv2.putText(wynikImage, zestawienie[0][0], (int(xmean), int(ymean - 25)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
                        if(int(ymean + 25) < oryginal.shape[0]):
                            cv2.putText(wynikImage, zestawienie[0][0], (int(xmean), int(ymean + 25)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)



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

                xmean = np.mean(x)
                ymean = np.mean(y)
                odlegloscOdSrodka = ((xmean - x) ** 2 + (ymean - y) ** 2) ** (0.5)
                odlegloscOdSrodkaMean = np.mean(odlegloscOdSrodka)
                odlegloscOdSrodkaStd = np.std(odlegloscOdSrodka)

                if (odlegloscOdSrodkaStd > 0.2 * odlegloscOdSrodkaMean) and (dx >= minimalnyBok) and (dy >= minimalnyBok):
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

                        print("\t\tTrwa identyfikacja nominału banknotu...")
                        wycinek = np.copy(oryginal)

                        xlen = wycinek.shape[0]
                        ylen = wycinek.shape[1]
                        xWspolrzedna = np.zeros((xlen, ylen))
                        yWspolrzedna = np.zeros((xlen, ylen))

                        for ix in range(xlen):
                            xWspolrzedna[ix] = np.arange(ylen)
                            yWspolrzedna[ix] = np.ones(ylen) * ix

                        wycinek[yWspolrzedna < xWspolrzedna * funkcjaLiniowa[0][0] + funkcjaLiniowa[0][1]] = [255, 255, 255]
                        wycinek[yWspolrzedna > xWspolrzedna * funkcjaLiniowa[1][0] + funkcjaLiniowa[1][1]] = [255, 255, 255]
                        wycinek[yWspolrzedna > xWspolrzedna * funkcjaLiniowa[2][0] + funkcjaLiniowa[2][1]] = [255, 255, 255]
                        wycinek[yWspolrzedna < xWspolrzedna * funkcjaLiniowa[3][0] + funkcjaLiniowa[3][1]] = [255, 255, 255]

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

                        npWyniki = np.array(wynikiStatystyk)
                        jakosc = npWyniki[:, 0]
                        jakosc = jakosc.astype(np.float)
                        ileKandydatow = sum(jakosc < jakosc[0] + 0.04)

                        nazwaNominalu = npWyniki[:ileKandydatow, 1]

                        npUnique = np.unique(nazwaNominalu, return_counts=True)

                        zestawienie = []
                        for iter in range(len(npUnique[0])):
                            zestawienie.append([npUnique[0][iter], npUnique[1][iter]])

                        zestawienie.sort(key=lambda x: x[1], reverse=True)

                        cv2.putText(wynikImage, zestawienie[0][0], (int(xmean), int(ymean - 25)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 255), 3, cv2.LINE_AA)
                        cv2.putText(wynikImage, zestawienie[0][0], (int(xmean), int(ymean + 25)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 0, 0), 3, cv2.LINE_AA)

            print("\tIdentyfikacja konturów okręgów zakończona.")


            imageKonturyOkregow = cv2.drawContours(wynikImage.copy(), konturyOkregow, -1, (0, 255, 0), 3)
            imageKonturyProstokatow = cv2.drawContours(imageKonturyOkregow.copy(), konturyProstokatow, -1, (255, 0, 0), 3)

            print("\tTrwa rysowanie wykresow...")
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 5), subplot_kw={'xticks': [], 'yticks': []})
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)
            ax[0].imshow(cv2.cvtColor(oryginal, cv2.COLOR_BGR2RGB), aspect='equal', interpolation='bilinear')
            ax[1].set_title("Ilosc konturów: " + str(len(contours)))
            ax[1].imshow(cv2.cvtColor(imageWszystkieKontury, cv2.COLOR_BGR2RGB), aspect='equal', interpolation='bilinear')
            ax[2].set_title("Ilosc nominałów: " + str(len(konturyOkregow)))
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