import cv2
import numpy as np


def scale_image_inverse_mapping(img, scale, interpol_func):
    sx, sy = scale
    h, w = img.shape[:2]
    new_w, new_h = int(w * sx), int(h * sy)
    y, x = np.meshgrid(np.arange(new_h), np.arange(new_w), indexing='ij')
    orig_x = x / sx
    orig_y = y / sy
    return np.round(interpol_func(img, orig_x, orig_y)).astype(int)


def rotate_image(img, angle, interpol_func):
    (h, w) = img.shape[:2]
    angle_rad = np.deg2rad(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    new_h = int(abs(h * cos_a) + abs(w * sin_a))
    new_w = int(abs(h * sin_a) + abs(w * cos_a))
    Y, X = np.meshgrid(np.arange(new_h), np.arange(new_w), indexing='ij')

    old_center = np.array([w // 2, h // 2])
    new_center = np.array([new_w // 2, new_h // 2])

    dx = new_center[0] - old_center[0] * cos_a + old_center[1] * sin_a
    dy = new_center[1] - old_center[0] * sin_a - old_center[1] * cos_a

    orig_x = cos_a*X + sin_a*Y + (-dx*cos_a - dy*sin_a)
    orig_y = -sin_a*X + cos_a*Y + (sin_a*dx - cos_a*dy)

    # Pixels that should remain black
    x_out_of_bounds = (orig_x < 0) | (orig_x >= img.shape[1])
    y_out_of_bounds = (orig_y < 0) | (orig_y >= img.shape[0])

    # Apply the mask to assign black (zero) values to out-of-bounds indices
    new_img = np.round(interpol_func(img, orig_x, orig_y)).astype(int)
    new_img[x_out_of_bounds | y_out_of_bounds] = 0
    return new_img


# x and y are 2D arrays with same shape = img.shape
# they store the original index of the pixel ij in the image
# x[0, 0] = 3 => pixel 0,0 in new image has coord x=3 in old image
def closest_neighbour_interpolation(img, x, y):
  x = np.round(x).astype(int)
  y = np.round(y).astype(int)
  x_clipped = np.clip(x, 0, img.shape[1] - 1)
  y_clipped = np.clip(y, 0, img.shape[0] - 1)

  return img[y_clipped, x_clipped]


def bilinear_interpolation(img, x, y):
  dx = x - np.floor(x)
  dy = y - np.floor(y)
  x = np.floor(x).astype(int)
  y = np.floor(y).astype(int)
  x = np.clip(x, 0, img.shape[1] - 2)
  y = np.clip(y, 0, img.shape[0] - 2)

  # 4 neighbours
  Qi = [img[y+j, x+i] * ((-1)**(i-j) * (dx-i) * (dy-j))[..., np.newaxis] for i in range(0, 2) for j in range(0, 2)]
  return np.sum(Qi, axis=0)


def R(S):
  return (P(S+2) - 4*P(S+1) + 6*P(S) - 4*P(S-1)) / 6


def P(X):
    X[X < 0] = 0
    return X ** 3


def bicubic_interpolation(img, x, y):
    dx = x - np.floor(x)
    dy = y - np.floor(y)
    x = np.floor(x).astype(int)
    y = np.floor(y).astype(int)
    x = np.clip(x, 1, img.shape[1] - 3)
    y = np.clip(y, 1, img.shape[0] - 3)

    # 16 neighbours
    Qi = [img[y+n, x+m] * R(m-dx)[..., np.newaxis] * R(dy-n)[..., np.newaxis] for m in range(-1, 3) for n in range(-1, 3)]
    return np.sum(Qi, axis=0)


def L(n, x, y, img):
  dx = x - np.floor(x)
  x = np.floor(x).astype(int)

  temp = (dx-1) * (dx-2)
  temp2 = (dx * temp)

  T1 = -temp2[..., np.newaxis] * img[y+n-2, x-1] / 3
  T2 = (temp + temp2)[..., np.newaxis] * img[y+n-2, x]

  temp += 4*dx - 2
  temp2 = temp * (dx-2)

  T3 = -temp2[..., np.newaxis] * img[y+n-2, x+1]
  T4 = (temp + temp2)[..., np.newaxis] * img[y+n-2, x+2] / 3

  return (T1 + T2 + T3 + T4) / 2


def lagrange_interpolation(img, x, y):
    dy = y - np.floor(y)
    y = np.floor(y).astype(int)

    x = np.clip(x, 0, img.shape[1] - 3)
    y = np.clip(y, 0, img.shape[0] - 3)

    l = [L(n, x, y, img) for n in range(1, 5)]

    temp = (dy-1) * (dy-2)
    temp2 = dy * temp

    T1 = -temp2[..., np.newaxis] * l[0] / 3
    T2 = (temp + temp2)[..., np.newaxis] * l[1]

    temp += 4*dy - 2
    temp2 = temp * (dy-2)

    T3 = -temp2[..., np.newaxis] * l[2]
    T4 = (temp + temp2)[..., np.newaxis] * l[3] / 3

    return (T1 + T2 + T3 + T4) / 2


def choose_interpol_method():
  print('Escolha um método de interpolação:')
  print('1) Vizinho mais próximo')
  print('2) Bilinear')
  print('3) Bicúbica')
  print('4) Polinômios de Lagrange')
  method = int(input())

  if method == 1:
    func = closest_neighbour_interpolation
  elif method == 2:
    func = bilinear_interpolation
  elif method == 3:
    func = bicubic_interpolation
  elif method == 4:
    func = lagrange_interpolation
  else:
      raise SystemExit('Unrecognized option:', method)
  
  return func


def choose_scale_factors(h, w):
  option = input('Digite 1 se deseja usar um fator de escala: ')
  if option == '1':
    sx = float(input('Digite o fator de escala em x: '))
    sy = float(input('Digite o fator de escala em y: '))
  else:
    new_x = int(input('Digite a nova largura da imagem: '))
    new_y = int(input('Digite a nova altura da imagem: '))
    sx = new_x / w
    sy = new_y / h
  
  return (sx, sy)


def main():
  print('---- Bem vindo ao Image Transform! ----')
  
  inFilename = input('Digite o nome da imagem de entrada (deve estar na pasta input): ')
  outFilename = input('Digite o nome da imagem de saída (deixe em branco para usar nomeação default): ')

  print('O que você deseja fazer?')
  print('1) Girar imagem')
  print('2) Escalar imagem')
  
  img = cv2.imread(f'input/{inFilename}.png', cv2.IMREAD_COLOR)

  option = int(input())
  out_suffix = ''

  if option == 1:
    angle = float(input('Digite o ângulo de rotação desejado em graus: '))
    out_suffix = f'_rotated_{angle}'
    new_img = rotate_image(img, angle, choose_interpol_method())
  elif option == 2:
    (h, w) = img.shape[:2]
    method = choose_interpol_method()
    new_img = scale_image_inverse_mapping(img, choose_scale_factors(h, w), method)
    (h, w) = new_img.shape[:2]
    out_suffix = f'_scaled_{w}_{h}_{method.__name__}'
  else:
    raise SystemExit('Unrecognized option:', option)

  if outFilename == "":
       outFilename = inFilename + out_suffix

  cv2.imwrite(f'output/{outFilename}.png', new_img)

if __name__ == '__main__':
    main()