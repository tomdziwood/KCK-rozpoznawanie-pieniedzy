
import numpy as np
import cv2
from matplotlib import pyplot as plt

for i in range(320,373+1):
    for j in range(1,2+1):
        img1 = cv2.imread('wzorce/' +str(j)+ '0zl/1.jpg',0)          # queryImage
        img2 = cv2.imread('zdjecia/1280 x 960/CAM01'+str(i)+'.jpg',0) # trainImage

        # Initiate SIFT detector
        sift = cv2.ORB_create()   #SIFT()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)
        # Apply ratio test
        good = []
        for m,n in matches:
            #good.append([m])
            if m.distance < 0.75*n.distance:
                good.append([m])
        # cv2.drawMatchesKnn expects list of lists as matches.
        print(good.__sizeof__())
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,outImg=None,flags=2)
        plt.imshow(img3),plt.show()


"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy._lib.six import xrange

img1 = cv2.imread('wzorce/10zl/1.jpg',0)          # queryImage
img2 = cv2.imread('zdjecia/1280 x 960/CAM01371.jpg',0) # trainImage
# Initiate SIFT detector
sift = cv2.ORB_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
#flann = cv2.FlannBasedMatcher(index_params,search_params)
#matches = flann.knnMatch(des1,des2, k=2)#flann.knnMatch(des1,des2,k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in xrange(len(matches))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
plt.imshow(img3,),plt.show()

"""