# import cv2
# import matplotlib.pyplot as plt
#
# image1 = cv2.imread('obraz1.jpg', cv2.IMREAD_GRAYSCALE)
# image2 = cv2.imread('obraz2.jpg', cv2.IMREAD_GRAYSCALE)
# image3 = cv2.imread('obraz3.jpg', cv2.IMREAD_GRAYSCALE)
# image4 = cv2.imread('obraz4.jpg', cv2.IMREAD_GRAYSCALE)
#
# histogram1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
# histogram2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
# histogram3 = cv2.calcHist([image3], [0], None, [256], [0, 256])
# histogram4 = cv2.calcHist([image4], [0], None, [256], [0, 256])
#
# plt.figure(figsize=(12, 4))
# plt.subplot(131), plt.imshow(image1, cmap='gray'), plt.title('Obraz 1')
# plt.subplot(132), plt.plot(histogram1), plt.title('Histogram Obrazu 1')
# plt.subplot(133), plt.hist(image1.ravel(), 256, [0, 256]), plt.title('Histogram Obrazu 1 (inna metoda)')
# plt.show()
#
# plt.figure(figsize=(12, 4))
# plt.subplot(131), plt.imshow(image2, cmap='gray'), plt.title('Obraz 2')
# plt.subplot(132), plt.plot(histogram2), plt.title('Histogram Obrazu 2')
# plt.subplot(133), plt.hist(image2.ravel(), 256, [0, 256]), plt.title('Histogram Obrazu 2 (inna metoda)')
# plt.show()
#
# plt.figure(figsize=(12, 4))
# plt.subplot(131), plt.imshow(image3, cmap='gray'), plt.title('Obraz 3')
# plt.subplot(132), plt.plot(histogram3), plt.title('Histogram Obrazu 3')
# plt.subplot(133), plt.hist(image3.ravel(), 256, [0, 256]), plt.title('Histogram Obrazu 3 (inna metoda)')
# plt.show()
#
# plt.figure(figsize=(12, 4))
# plt.subplot(131), plt.imshow(image4, cmap='gray'), plt.title('Obraz 4')
# plt.subplot(132), plt.plot(histogram4), plt.title('Histogram Obrazu 4')
# plt.subplot(133), plt.hist(image4.ravel(), 256, [0, 256]), plt.title('Histogram Obrazu 4 (inna metoda)')
# plt.show()

# import cv2
# import matplotlib.pyplot as plt
#
# image1 = cv2.imread('obraz1.jpg', cv2.IMREAD_GRAYSCALE)
# image2 = cv2.imread('obraz2.jpg', cv2.IMREAD_GRAYSCALE)
# image3 = cv2.imread('obraz3.jpg', cv2.IMREAD_GRAYSCALE)
# image4 = cv2.imread('obraz4.jpg', cv2.IMREAD_GRAYSCALE)
#
# equalized_image1 = cv2.equalizeHist(image1)
# equalized_image2 = cv2.equalizeHist(image2)
# equalized_image3 = cv2.equalizeHist(image3)
# equalized_image4 = cv2.equalizeHist(image4)
#
# min_pixel_value = 50
# max_pixel_value = 200
# stretched_image1 = cv2.normalize(image1, None, min_pixel_value, max_pixel_value, cv2.NORM_MINMAX)
# stretched_image2 = cv2.normalize(image2, None, min_pixel_value, max_pixel_value, cv2.NORM_MINMAX)
# stretched_image3 = cv2.normalize(image3, None, min_pixel_value, max_pixel_value, cv2.NORM_MINMAX)
# stretched_image4 = cv2.normalize(image4, None, min_pixel_value, max_pixel_value, cv2.NORM_MINMAX)
#
# alpha = 1.5
# scaled_image1 = cv2.multiply(image1, alpha)
# scaled_image2 = cv2.multiply(image2, alpha)
# scaled_image3 = cv2.multiply(image3, alpha)
# scaled_image4 = cv2.multiply(image4, alpha)
#
# kernel_size = (5, 5)
# smoothed_image1 = cv2.GaussianBlur(image1, kernel_size, 0)
# smoothed_image2 = cv2.GaussianBlur(image2, kernel_size, 0)
# smoothed_image3 = cv2.GaussianBlur(image3, kernel_size, 0)
# smoothed_image4 = cv2.GaussianBlur(image4, kernel_size, 0)
#
# sharpened_image3 = cv2.addWeighted(image3, 1.5, smoothed_image3, -0.5, 0)
# sharpened_image4 = cv2.addWeighted(image4, 1.5, smoothed_image4, -0.5, 0)
#
# plt.figure(figsize=(12, 4))
# plt.subplot(231), plt.imshow(image1, cmap='gray'), plt.title('Obraz 1')
# plt.subplot(232), plt.imshow(equalized_image1, cmap='gray'), plt.title('Wyrównany Obraz 1')
# plt.subplot(233), plt.imshow(stretched_image1, cmap='gray'), plt.title('Rozciągnięty Obraz 1')
# plt.subplot(234), plt.imshow(scaled_image1, cmap='gray'), plt.title('Skalowany Obraz 1')
# plt.subplot(235), plt.imshow(smoothed_image1, cmap='gray'), plt.title('Wygładzony Obraz 1')
# plt.show()
#
# plt.figure(figsize=(12, 4))
# plt.subplot(231), plt.imshow(image2, cmap='gray'), plt.title('Obraz 2')
# plt.subplot(232), plt.imshow(equalized_image2, cmap='gray'), plt.title('Wyrównany Obraz 2')
# plt.subplot(233), plt.imshow(stretched_image2, cmap='gray'), plt.title('Rozciągnięty Obraz 2')
# plt.subplot(234), plt.imshow(scaled_image2, cmap='gray'), plt.title('Skalowany Obraz 2')
# plt.subplot(235), plt.imshow(smoothed_image2, cmap='gray'), plt.title('Wygładzony Obraz 2')
# plt.show()
#
# plt.figure(figsize=(12, 4))
# plt.subplot(231), plt.imshow(image3, cmap='gray'), plt.title('Obraz 3')
# plt.subplot(232), plt.imshow(equalized_image3, cmap='gray'), plt.title('Wyrównany Obraz 3')
# plt.subplot(233), plt.imshow(stretched_image3, cmap='gray'), plt.title('Rozciągnięty Obraz 3')
# plt.subplot(234), plt.imshow(scaled_image3, cmap='gray'), plt.title('Skalowany Obraz 3')
# plt.subplot(235), plt.imshow(sharpened_image3, cmap='gray'), plt.title('Wyostrzony Obraz 3')
# plt.show()
#
# plt.figure(figsize=(12, 4))
# plt.subplot(231), plt.imshow(image4, cmap='gray'), plt.title('Obraz 3')
# plt.subplot(232), plt.imshow(equalized_image4, cmap='gray'), plt.title('Wyrównany Obraz 4')
# plt.subplot(233), plt.imshow(stretched_image4, cmap='gray'), plt.title('Rozciągnięty Obraz 4')
# plt.subplot(234), plt.imshow(scaled_image4, cmap='gray'), plt.title('Skalowany Obraz 4')
# plt.subplot(235), plt.imshow(sharpened_image4, cmap='gray'), plt.title('Wyostrzony Obraz 4')
# plt.show()

from skimage import io
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

original_image = io.imread('original_image.jpg', as_gray=True)
equalized_image = io.imread('equalized_image.jpg', as_gray=True)
stretched_image = io.imread('stretched_image.jpg', as_gray=True)
scaled_image = io.imread('scaled_image.jpg', as_gray=True)
smoothed_image = io.imread('smoothed_image.jpg', as_gray=True)

psnr_equalized = psnr(original_image, equalized_image)
psnr_stretched = psnr(original_image, stretched_image)
psnr_scaled = psnr(original_image, scaled_image)
psnr_smoothed = psnr(original_image, smoothed_image)

ssim_equalized = ssim(original_image, equalized_image, data_range=equalized_image.max() - equalized_image.min())
ssim_stretched = ssim(original_image, stretched_image, data_range=stretched_image.max() - stretched_image.min())
ssim_scaled = ssim(original_image, scaled_image, data_range=scaled_image.max() - scaled_image.min())
ssim_smoothed = ssim(original_image, smoothed_image, data_range=smoothed_image.max() - smoothed_image.min())

print(f'PSNR po wyrównaniu histogramu: {psnr_equalized:.2f}')
print(f'PSNR po rozciągnięciu histogramu: {psnr_stretched:.2f}')
print(f'PSNR po skalowaniu histogramu: {psnr_scaled:.2f}')
print(f'PSNR po wygładzeniu histogramu: {psnr_smoothed:.2f}')

print(f'SSIM po wyrównaniu histogramu: {ssim_equalized:.2f}')
print(f'SSIM po rozciągnięciu histogramu: {ssim_stretched:.2f}')
print(f'SSIM po skalowaniu histogramu: {ssim_scaled:.2f}')
print(f'SSIM po wygładzeniu histogramu: {ssim_smoothed:.2f}')
