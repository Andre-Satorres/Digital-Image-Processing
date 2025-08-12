import cv2 as cv
import numpy as np
import io
import sys

def num_bits(n):
    if n == 0:
        return 0
    return num_bits(n // 2) + (n & 1)


def hide_text_in_image(image: cv.typing.MatLike, message: str, bit_plans_to_use: int):
    bit_plan = 255 - bit_plans_to_use
    bit_amount = num_bits(bit_plans_to_use)
    message_index = 0
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            for i in range(3):
                if message_index >= len(message):
                    return
                
                channel = image[x, y, i]

                mask = np.array(list(format(bit_plans_to_use, '03b')), dtype=np.unicode_)
                mask = np.where(mask == '1', message[message_index:message_index+bit_amount], mask)
                message_index += bit_amount

                image[x, y, i] = (channel & bit_plan) | int(''.join(mask))

# Leitura da entrada
args = sys.argv
in_filename = args[1]
in_text = args[2]
bit_plans = int(args[3]) # int de 0 a 7
out_filename = args[4]

img = cv.imread(in_filename)
text_content = io.open(in_text, mode="r", encoding="utf-8").read()
message = np.array(list(text_content), dtype=np.unicode_)

# O '08b' significa que queremos uma representação binária com zero à esquerda para preencher até 8 dígitos. 
# Por exemplo, o número 65 seria formatado como '01000001'
binary_message = ''.join(format(ord(char), '08b') for char in message)

# Verifique se a imagem é grande o suficiente para a mensagem
max_message_length = (img.shape[0] * img.shape[1] * 3 * num_bits(bit_plans))
if len(binary_message) > max_message_length:
    raise ValueError("A mensagem é muito longa para a imagem.")

print(len(binary_message))

hide_text_in_image(img, binary_message, bit_plans)

# Salve a imagem resultante
cv.imwrite(out_filename, img)
print(f"Mensagem ocultada com sucesso na imagem {out_filename}")
