import cv2 as cv
import numpy as np
import io

img = cv.imread('out.png')

bit_plan = 0x01 # 0x07
amount = '01b' # '03b'

binary_msg = "".join(format(channel & bit_plan, amount) for px in img.reshape(-1, 3) for channel in px)[:574176]

# The second argument, 2, specifies that we are interpreting the string as a binary representation.
msg = ''.join(chr(int(binary_msg[i*8:i*8+8], 2)) for i in range(len(binary_msg)//8))
f = io.open('out.txt', 'w', encoding="utf-8")
f.write(msg)
f.close()

print(f"Mensagem decodificada com sucesso no arquivo out.txt")