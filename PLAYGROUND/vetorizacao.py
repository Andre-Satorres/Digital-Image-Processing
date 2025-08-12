import numpy as np;
import cv2 as cv;
import matplotlib.pyplot as plt;


# 1) Contar o número de transições de False para True em uma sequência.

# x = np.array ([False, True, False, False, True])
# print((x[:-1] < x[1:]).sum())

# ---

# 2) Selecionar apenas os números ímpares e elevá-los ao quadrado

# x = np.arange(10)
# y = (x[x & 1 == True]) ** 2
# print(y)

# ---

# 3) Somar todos os elementos de um vetor que são divisíveis por 5.

# x = np.arange(20)
# print(np.sum(x[x % 5 == 0]))

# ---

# 4) Somar subsequências de n valores em um vetor

# X = np.arange(12) + 1
# n = 4
# X = X.reshape((-1, n))
# print(X.sum(axis=1))

# ---

# 5) Trocar todos os valores negativos de um vetor por zeros

# X = np.array([-3, -2, -1, 0, 1, 2, 3])
# X[X < 0] = 0
# print(X)

# ---

# 6) Contar o número de valores pares em um vetor

# X = np.arange(10)
# print(np.sum(X % 2 == 0))

# ---

# 7) Encontrar os máximos (picos) locais em um vetor

# x = np.array ([1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 2, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1])
# y = np.diff(np.sign(np.diff(x)))
# ----- z = np.where(y != 0)[0] + 1 -- minimos e maximos
# print(np.where(y < 0)[0] + 1) # so maximos

# ---

# 8) Calcular a média móvel com uma janela de tamanho n
# x = np.array ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# n = 4

# janela = np.lib.stride_tricks.sliding_window_view(x, window_shape=n)
# print(np.mean(janela, axis=1))

# ---

# 9) Normalizar um vetor para que seus elementos somem 1.

# x = np.arange(10)
# y = x / np.sum(x)
# print(y)

# 10) Calcular a distância euclidiana entre dois vetores.

# a = np.array ([1, 2, 3])
# b = np.array ([4, 5, 6])
# print(np.sqrt(np.square(b - a).sum()))

# ---

# 11) Verificar se um vetor está ordenado crescentemente (retorno: True ou False).

# x = np.arange(5)
# y = np.diff(x)
# print(np.sum(y < 0) == 0)

# 12) Calcular as diferenças consecutivas de um vetor.

# x = np.array([10, 7, 4, 3, 2])
# print(np.diff(x))

# 13) Somar os elementos positivos de um vetor.

# x = np.array ([1, -2, 3, 4, -5, 6, -7, 8, -9])
# print(np.sum(x[x > 0]))

# 14) Verificar se um vetor é um palíndromo.

# x = np.array ([1, 3, 3, 4, 5, 4, 3, 3, 1])
# y = x[::-1]
# z = x - y
# print(np.sum(z != 0) == 0)

# 15) Construir o histograma de uma imagem monocromática.

img_path = "C:/Users/USER/Documents/IMAGENS/"
img = cv.imread(img_path + "IGB2/celio.jpg", cv.IMREAD_GRAYSCALE)

values, counts = np.unique(img, return_counts=True)
plt.plot(values,counts)
plt.xlim(0, 255)
plt.xlabel('Valor de cinza')
plt.ylabel('Aparicoes [#]')
plt.show()