import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def calculate_haralick_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    distances = [1, 2, 3]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(gray_image, distances, angles, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')
    correlation = graycoprops(glcm, 'correlation')
    energy = graycoprops(glcm, 'energy')
    std_dev = graycoprops(glcm, 'homogeneity')
    return contrast, correlation, energy, std_dev


def apply_gabor_filter(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ksize = 5
    theta = np.pi / 4
    sigma = 1.0
    lambd = 5.0
    gamma = 0.5
    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma)
    filtered_image = cv2.filter2D(gray_image, cv2.CV_8UC3, gabor_kernel)
    return filtered_image


if __name__ == '__main__':
    image_paths = ["obraz.jpg", "obraz2.jpg", "obraz3.jpg"]
    images = [cv2.imread(image_path) for image_path in image_paths]
    for i, image in enumerate(images):
        region_of_interest = image[100:300, 100:300]
        contrast, correlation, energy, std_dev = calculate_haralick_features(region_of_interest)
        gabor_result = apply_gabor_filter(region_of_interest)
        print(f"Analiza tekstur dla obrazu {i + 1}")
        print(f"Kontrast: {contrast}, Korelacja: {correlation}, Energia: {energy}, Odchylenie standardowe: {std_dev}")
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Obraz Oryginalny")
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2RGB))
        plt.title("Obszar Analizy Tekstur")
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.imshow(gabor_result, cmap='gray')
        plt.title("Wynik Filtracji Gabora")
        plt.axis("off")
        plt.show()
