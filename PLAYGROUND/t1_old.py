import cv2 as cv
import numpy as np

# 1) Dividir uma imagem em 16 blocos e retornar uma reorganizacao desses blocos

img = cv.imread("baboon.ppm", cv.IMREAD_GRAYSCALE)

reorg = np.array([1, 13, 3, 4, 
         5, 8, 6, 7, 
         9, 8, 11, 12,
         2, 14, 15, 16]).reshape((4, 4))

len_block = int(img.shape[0] / 4)

img2 = np.array(img)
for i in range(len(reorg)):
    start_i = i * len_block
    for j in range(len(reorg[0])):
        start_j = j * len_block

        new_pos = int(reorg[i][j] - 1)
        img_i = (new_pos // 4) * len_block
        img_j = (new_pos % 4) * len_block

        img2[start_i:start_i+len_block, start_j:start_j+len_block] = img[img_i:img_i+len_block, img_j:img_j+len_block]


img2 = np.zeros_like(img)

new_positions = (reorg - 1).reshape(-1)

img_i = ((new_positions // 4) * len_block)
img_j = ((new_positions % 4) * len_block)

start_i = np.repeat(np.arange(4) * len_block, 4)
start_j = np.tile(np.arange(4) * len_block, 4)


plus = np.arange(len_block)

a = np.arange(16 * len_block * len_block).reshape(len_block * len_block, 16)
a.T.reshape(4, 4, len_block, len_block).transpose(0, 2, 1, 3).reshape(4*len_block, 4*len_block)

print(a)


img2[start_i[:, None] + plus, start_j[:, None] + plus] = img[img_i[:, None] + plus, img_j[:, None] + plus]

# print(img2)

cv.imshow("a", img2)
cv.waitKey(0)