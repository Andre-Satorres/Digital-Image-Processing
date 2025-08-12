import cv2 as cv

img_path = "C:/Users/USER/Documents/IMAGENS/"
img = cv.imread(img_path + "IGB2/celio.jpg")

cv.imshow("Display window", img)
k = cv.waitKey(0) # Wait for a keystroke in the window

cv.bilateralFilter()