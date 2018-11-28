import cv2
from matplotlib import pyplot as plt
import numpy as np
import os



def progowanie(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
	
    return thresh



def main():
    sciezkaZrodlowa = "zdjecia/monety - latwe/"

    tytul = os.listdir(sciezkaZrodlowa)


    #for i in range(len(tytul)):
    #for i in [0, 10, 20, 30, 40, 50, 60]:
    for iTytul in [22]:
        print("\n\nTrwa przetwarzanie obrazu o indeksie i = " + str(iTytul) + "/" + str(len(tytul)) + " (" + tytul[iTytul] + ")")
        oryginal = cv2.imread(sciezkaZrodlowa + tytul[iTytul], cv2.IMREAD_COLOR)


        print("\tTrwa progowanie obrazu...")
        image = progowanie(oryginal)


        print("\tTrwa rysowanie wykresów...")
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.5), subplot_kw={'xticks': [], 'yticks': []})
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)
        ax[0].imshow(cv2.cvtColor(oryginal, cv2.COLOR_BGR2RGB), aspect='equal', interpolation='bilinear')
        ax[1].imshow(image, cmap='gray', aspect='equal', interpolation='bilinear')

        print("\tPrzygotowywanie wyświetlenia obrazów w panelu bocznym...")
        plt.show()


        print("\n\tTrwa wykonywanie cv2.HoughCircles()...")
        minimalnyPromien = int(min(image.shape)/20)
        maksymalnyPromien = int(min(image.shape)/3)
        minimalnyDystans = int(1.5 * minimalnyPromien)
        parametr2 = 40
        while(parametr2 > 0):
            print("Trwa próba parametr2 = " + str(parametr2) + "...")
            circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, minDist=minimalnyDystans, param1=50, param2=parametr2, minRadius=minimalnyPromien, maxRadius=maksymalnyPromien)
            if(not(circles is None) and (len(circles[0]) >= 10)):
                print("funkcja cv2.HoughCircles() zatrzymana dla parametr2 = " + str(parametr2))
                break
            parametr2 -= 1

        wynik = np.copy(oryginal)
        circles = np.uint16(np.around(circles))
        print("\tLiczba znalezionych okregow: " + str(len(circles[0])))
        print("\tTrwa sprawdzanie okregow...")
        for j in circles[0, :]:
            # rysowanie okregu
            cv2.circle(wynik, (j[0], j[1]), j[2], (0, 255, 0), 2)
            # rysowanie srodka okregu
            cv2.circle(wynik, (j[0], j[1]), 2, (0, 0, 255), 3)

            ktoryOkrag = np.copy(oryginal)
            # rysowanie okregu
            cv2.circle(ktoryOkrag, (j[0], j[1]), j[2], (0, 255, 0), 2)
            # rysowanie srodka okregu
            cv2.circle(ktoryOkrag, (j[0], j[1]), 2, (0, 0, 255), 3)


            wycinek = np.copy(oryginal[j[1] - j[2] : j[1] + j[2], j[0] - j[2] : j[0] + j[2]])

            print("\tTrwa rysowanie wykresów...")
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.5), subplot_kw={'xticks': [], 'yticks': []})
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)
            ax[0].imshow(cv2.cvtColor(ktoryOkrag, cv2.COLOR_BGR2RGB), aspect='equal', interpolation='bilinear')
            ax[1].imshow(cv2.cvtColor(wycinek, cv2.COLOR_BGR2RGB), cmap='gray', aspect='equal', interpolation='bilinear')

            print("\tPrzygotowywanie wyświetlenia obrazów w panelu bocznym...")
            plt.show()


            print("\tTrwa porównywanie obrazu ze wzorcem...")
            wzorzec = cv2.imread("wzorce/5zl/1.jpg", cv2.IMREAD_COLOR)
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(wzorzec, None)
            kp2, des2 = orb.detectAndCompute(wycinek, None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key = lambda x:x.distance)
            zestawienie = cv2.drawMatches(wzorzec, kp1, wycinek, kp2, matches[:10], None, flags=2)
            plt.imshow(cv2.cvtColor(zestawienie, cv2.COLOR_BGR2RGB))
            plt.show()



        print("\tTrwa rysowanie wykresów...")
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.5), subplot_kw={'xticks': [], 'yticks': []})
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.1, wspace=0.1)
        ax[0].imshow(cv2.cvtColor(oryginal, cv2.COLOR_BGR2RGB), aspect='equal', interpolation='bilinear')
        ax[1].imshow(cv2.cvtColor(wynik, cv2.COLOR_BGR2RGB), cmap='gray', aspect='equal', interpolation='bilinear')

        print("\tPrzygotowywanie wyświetlenia obrazów w panelu bocznym...")
        plt.show()


        """
        print("\tTrwa zapisywanie do pliku jpg...")
        fig.savefig(sciezkaDocelowa + tytul[i])
        print("\tPomyslnie zapisano do pliku jpg.\n\n")
        """



main()