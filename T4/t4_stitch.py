import cv2
import numpy as np
from scipy.spatial.distance import cdist

def SIFT(a, b):
  sift = cv2.SIFT_create(contrastThreshold=0.01)
  return sift.detectAndCompute(a, None), sift.detectAndCompute(b, None)


def BRIEF(a, b):
  star = cv2.xfeatures2d.StarDetector_create()
  brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
  return brief.compute(a, star.detect(a, None)), brief.compute(b, star.detect(b, None))


def ORB(a, b):
  orb = cv2.ORB_create()
  return orb.detectAndCompute(a, None), orb.detectAndCompute(b, None)


def ask_method():
  print('Qual método deseja utilizar?')
  print('1) SIFT')
  print('2) BRIEF')
  print('3) ORB')
  option = int(input())

  if option == 1:
    method = SIFT
  elif option == 2:
    method = BRIEF
  elif option == 3:
    method = ORB
  else:
    raise SystemExit('Unrecognized option:', option)
  
  return method


# -----------------------
# MAIN
# -----------------------

name = input('Digite o nome da imagem desejada (deve estar na pasta input): ')

# Ler as imagens de entrada
# IMAGEM DA ESQUERDA
a_color = cv2.imread(f'input/{name}_left.jpg')

# IMAGEM DA DIREITA
b_color = cv2.imread(f'input/{name}_right.jpg')

# 1) Converter as imagens coloridas de entrada em imagens de nıveis de cinza.
a = cv2.cvtColor(a_color, cv2.COLOR_BGR2GRAY)
b = cv2.cvtColor(b_color, cv2.COLOR_BGR2GRAY)

# 2) Encontrar pontos de interesse e descritores invariantes locais para o par de imagens.
method = ask_method()
(keypoints_1, descritores_1), (keypoints_2, descritores_2) = method(a, b)

descritores_1 = np.array(descritores_1)
descritores_2 = np.array(descritores_2)

# 3) Computar distancias (similaridades) entre cada descritor das duas imagens
distancias = cdist(descritores_1, descritores_2, metric='euclidean')

# 4) Selecionar as melhores correspondencias para cada descritor de imagem.
indices_correspondencias = np.argmin(distancias, axis=1)
bf = cv2.BFMatcher()

# Realizar a correspondência de k-vizinhos mais próximos
matches = bf.knnMatch(descritores_1, descritores_2, k=2)

distancias_1 = np.array([m[0].distance for m in matches])
distancias_2 = np.array([m[1].distance for m in matches])

# Aplicar a condição de Lowe's
razao = distancias_1 / distancias_2
threshold = input('Digite o limiar desejado [Deixe vazio para usar o default]: ')
if threshold == '':
  threshold = 0.75

mascara_lowe = razao < float(threshold)

# Filtrar as melhores correspondências
melhores_correspondencias = [matches[i][0] for i in range(len(mascara_lowe)) if mascara_lowe[i]]

if len(melhores_correspondencias) < 4:
  print(threshold)
  exit('No homography could be computed!')

# Extrair os pontos correspondentes
pts_imagem_1 = np.float32([keypoints_1[m.queryIdx].pt for m in melhores_correspondencias])
pts_imagem_2 = np.float32([keypoints_2[m.trainIdx].pt for m in melhores_correspondencias])

# 5) Estimar a matriz de homografia usando RANSAC
matriz_homografia, mascara = cv2.findHomography(pts_imagem_1, pts_imagem_2, cv2.RANSAC, 5.0)

# 8) Desenhar linhas entre pontos correspondentes
altura1, largura1, _ = a_color.shape
altura2, largura2, _ = b_color.shape

largura_combined = largura1 + largura2
altura_combined = max(altura1, altura2)

imagem_combined = np.zeros((altura_combined, largura_combined, 3), dtype=np.uint8)
imagem_combined[:altura1, :largura1, :] = a_color
imagem_combined[:altura2, largura1:largura1 + largura2, :] = b_color

# Obter pontos correspondentes apenas dos matches válidos pela máscara
pts_imagem_1_validos = pts_imagem_1[mascara.ravel() == 1]
pts_imagem_2_validos = pts_imagem_2[mascara.ravel() == 1]

for pt1, pt2 in zip(pts_imagem_1_validos, pts_imagem_2_validos):
    pt1 = (int(pt1[0]), int(pt1[1]))
    pt2 = (int(pt2[0]) + largura1, int(pt2[1]))
    cv2.line(imagem_combined, pt1, pt2, (0, 255, 0), 1)

# 6) Aplicar uma projecao de perspectiva (cv2.warpPerspective) para alinhar as imagens
# 7) Unir as imagens alinhadas e criar a imagem panoramica.
# Dimensões das imagens
h1, w1 = a_color.shape[:2]
h2, w2 = b_color.shape[:2]

# Definir os pontos das bordas das imagens
list_of_points_1 = np.float32([[0,0], [0, h1],[w1, h1], [w1, 0]]).reshape(-1, 1, 2)
temp_points = np.float32([[0,0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1,1,2)

# Transformar os pontos da imagem b para o espaço da imagem a
list_of_points_2 = cv2.perspectiveTransform(temp_points, matriz_homografia)
list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

# Encontrar os limites da nova imagem
[x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

# Criar a matriz de translacao
H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

# Aplicar a transformação e a translacao à imagem b
output_img = cv2.warpPerspective(a_color, H_translation.dot(matriz_homografia), (w2 + abs(x_min), h2 + abs(y_min)))
output_img[-y_min:h2-y_min, -x_min:w2-x_min] = b_color

cv2.imwrite(f'output/{name}_{method.__name__}_matches.jpg', imagem_combined)
cv2.imwrite(f'output/{name}_{method.__name__}.jpg', output_img)