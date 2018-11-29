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



def main():
    sciezkaZrodlowa = "zdjecia/"
    sciezkaDocelowa = "Rozpoznawanie nominalu monet/"
    sciezkaWzorcow = "wzorce/"
    nominalyMonet = ["5gr", "50gr", "1zl", "2zl", "5zl"]

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
            print("\t\tTrwa szykowanie wzorca... (" + str(iTytulWzorca) + "/" + str(len(tytulWzorca)) + ") " + sciezkaWzorca + tytulWzorca[iTytulWzorca])

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


    for iKategoriaZdjec in range(len(kategoriaZdjec)):
        tytul = os.listdir(sciezkaZrodlowa + kategoriaZdjec[iKategoriaZdjec])

        for iTytul in range(len(tytul)):
            print("\nTrwa przetwarzanie obrazu " + str(iTytul) + "/" + str(len(tytul)) + " (" + kategoriaZdjec[iKategoriaZdjec] + tytul[iTytul] + ")")
            oryginal = cv2.imread(sciezkaZrodlowa + kategoriaZdjec[iKategoriaZdjec] + tytul[iTytul], cv2.IMREAD_COLOR)
            minimalnyPromien = int(min(oryginal.shape[:2])/20)


            print("\tTrwa wykrywanie konturów...")
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



            print("\tIdentyfikacja konturów okręgów zakończona.")


            imageKonturyOkregow = cv2.drawContours(wynikImage.copy(), konturyOkregow, -1, (0, 255, 0), 3)

            print("\tTrwa rysowanie wykresow...")
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 5), subplot_kw={'xticks': [], 'yticks': []})
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)
            ax[0].imshow(cv2.cvtColor(oryginal, cv2.COLOR_BGR2RGB), aspect='equal', interpolation='bilinear')
            ax[1].set_title("Ilosc konturów: " + str(len(contours)))
            ax[1].imshow(cv2.cvtColor(imageWszystkieKontury, cv2.COLOR_BGR2RGB), aspect='equal', interpolation='bilinear')
            ax[2].set_title("Ilosc okręgów: " + str(len(konturyOkregow)))
            ax[2].imshow(cv2.cvtColor(imageKonturyOkregow, cv2.COLOR_BGR2RGB), aspect='equal', interpolation='bilinear')

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