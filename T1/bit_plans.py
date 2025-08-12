import cv2 as cv

def to_binary(im):
    return cv.threshold(cv.cvtColor(im, cv.COLOR_RGB2GRAY), 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

img = cv.imread('out.png')

img_bit_plan_0 = to_binary(img & 1)
img_bit_plan_1 = to_binary(img & 2)
img_bit_plan_2 = to_binary(img & 4)
img_bit_plan_7 = to_binary(img & 128)

cv.imwrite('bit_plan_0.png', img_bit_plan_0)
cv.imwrite('bit_plan_1.png', img_bit_plan_1)
cv.imwrite('bit_plan_2.png', img_bit_plan_2)
cv.imwrite('bit_plan_7.png', img_bit_plan_7)