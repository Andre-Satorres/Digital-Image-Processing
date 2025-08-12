import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt

def create_rectangle_filter(image, size_w, size_h, pad, dx):
    height, width = image.shape[:2]

    center_h = height // 2
    center_w = width // 2
    mask = np.ones((height, width), dtype=np.uint8)

    mask[(center_h-pad-size_h):(center_h-pad), (center_w-size_w-dx//2):(center_w+size_w-dx//2)] = 0
    mask[center_h+pad:center_h+pad+size_h, center_w-size_w+dx//2:center_w+size_w+dx//2] = 0
    return mask


def create_filter(image, inner_radius, outer_radius):
    rows, cols = image.shape
    center = [rows // 2, cols // 2]

    row_indices, col_indices = np.ogrid[:rows, :cols]
    distance_from_center = np.sqrt((row_indices - center[0])**2 + (col_indices - center[1])**2)

    mask = np.logical_and(distance_from_center >= inner_radius, distance_from_center <= outer_radius)
    new = np.zeros_like(image, dtype=np.uint8)
    new[mask] = 1
    return new


def from_freq_to_space(image):
    return np.uint8(np.abs(np.fft.ifft2(np.fft.ifftshift(image))))


def save_image(path, name, img):
    cv.imwrite(os.path.join(path, f'{name}.png'), img)


def plot_histograms(path, plots: list):
    plt.figure(figsize=(10, 5))
    i = 1
    for name, plot in plots:
        plt.subplot(1, len(plots), i)
        unique_values, value_counts = np.unique(plot, return_counts=True)
        plt.bar(unique_values, value_counts, edgecolor='blue', align='edge')
        plt.title(f'{name} Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.ylim(0)
        plt.xlim(-5, 260)
        i += 1

    plt.tight_layout(pad=2.0)
    plt.savefig(path + '6_histogram_comparison.png')


def main():
    # (i) abrir uma imagem de entrada convertida para escala de cinza
    filename = 'snake' # input('Imagem:')
    original = cv.imread(filename + '.png', cv.IMREAD_GRAYSCALE)

    # (ii) aplicar a transformada rapida de Fourier e (iii) centralizar o espectro de frequencia
    original_after_fourier = np.fft.fftshift(np.fft.fft2(original))
    original_after_fourier_inverse = from_freq_to_space(original_after_fourier)

    # (iv) criar os nucleos (mascaras) para os diferentes filtros com as mesmas dimensoes das imagens
    # tal que que os raios dos cırculos definem as frequencias que serao atenuadas/preservadas
    low_pass_filter = create_filter(original_after_fourier, 0, 30)
    high_pass_filter = 1 - low_pass_filter
    band_pass_filter = create_filter(original_after_fourier, 7, 8)
    band_reject_filter = 1 - band_pass_filter

    rectangle_filter = create_rectangle_filter(original_after_fourier, size_h=2, size_w=1, pad=4, dx=10)

    # (v) aplicar cada filtro por meio da multiplicacao entre o espectro de frequencia e a mascara do filtro
    # (vi) aplicar a transformada inversa de Fourier para converter o espectro de frequencia filtrada de volta para o domınio espacial, 
    # gerando a imagem filtrada
    original_with_low_pass_filter = from_freq_to_space(original_after_fourier * low_pass_filter)
    original_with_high_pass_filter = from_freq_to_space(original_after_fourier * high_pass_filter)
    original_with_band_pass_filter = from_freq_to_space(original_after_fourier * band_pass_filter)
    original_with_band_reject_filter = from_freq_to_space(original_after_fourier * band_reject_filter)
    
    # Noise Removal (if necessary)
    original_with_rectangle_filter = from_freq_to_space(original_after_fourier * rectangle_filter)

    ### COMPRESSION
    magnitude_spectrum = np.log(np.abs(original_after_fourier) + 1)
    threshold_frequency = 0.5 * np.max(magnitude_spectrum)
    compression_filter = np.where(magnitude_spectrum >= threshold_frequency, 1, 0)
    original_compressed = from_freq_to_space(original_after_fourier * compression_filter)

    save = True
    if save:
        original_after_fourier = np.uint8(cv.normalize(magnitude_spectrum, None, 0, 255, cv.NORM_MINMAX))
        low_pass_filter = np.where(low_pass_filter == 1, original_after_fourier, 0)
        high_pass_filter = np.where(high_pass_filter == 1, original_after_fourier, 0)
        band_pass_filter = np.where(band_pass_filter == 1, original_after_fourier, 0)
        band_reject_filter = np.where(band_reject_filter == 1, original_after_fourier, 0)
        rectangle_filter = np.where(rectangle_filter == 1, original_after_fourier, 0)
        compression_filter = np.where(compression_filter == 1, original_after_fourier, 0)

        # (vii) visualizar e analisar os resultados
        path = f'out/{filename}/'
        os.makedirs(path, exist_ok=True)

        save_image(path, '1_original', original)

        save_image(path, '2_1_fourier', original_after_fourier)
        save_image(path, '2_2_original_after_fourier_inverse', original_after_fourier_inverse)
        save_image(path, '2_3_binary_diff', np.where(original == original_after_fourier_inverse, 0, 255))
        save_image(path, '2_4_abs_diff', np.abs(np.subtract(original, original_after_fourier_inverse)))

        save_image(path, '3_1_low_pass_filter', low_pass_filter)
        save_image(path, '3_2_high_pass_filter', high_pass_filter)
        save_image(path, '3_3_band_pass_filter', band_pass_filter)
        save_image(path, '3_4_band_reject_filter', band_reject_filter)
        save_image(path, '3_5_rectangle_filter', rectangle_filter)

        save_image(path, '4_1_original_with_low_pass_filter', original_with_low_pass_filter)
        save_image(path, '4_2_original_with_high_pass_filter', original_with_high_pass_filter)
        save_image(path, '4_3_original_with_band_pass_filter', original_with_band_pass_filter)
        save_image(path, '4_4_original_with_band_reject_filter', original_with_band_reject_filter)
        save_image(path, '4_5_original_with_rectangle_filter', original_with_rectangle_filter)

        save_image(path, '5_1_original_compressed', original_compressed)
        save_image(path, '5_2_compression_filter', compression_filter)

        # Plot the histograms
        plot_histograms(path, [
            ('Original', original.flatten()), 
            ('After inverse', original_after_fourier_inverse.flatten()),
            ('Compressed', original_compressed.flatten()),
            ])

main()