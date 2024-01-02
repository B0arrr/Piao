# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# image = cv2.imread('obraz.jpg', cv2.IMREAD_GRAYSCALE)
# sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
# sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
# sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
# canny_edges = cv2.Canny(image, threshold1=100, threshold2=200)
# laplacian = cv2.Laplacian(image, cv2.CV_64F)
# plt.figure(figsize=(12, 6))
# plt.subplot(231), plt.imshow(image, cmap='gray'), plt.title('Obraz źródłowy')
# plt.subplot(232), plt.imshow(sobel_magnitude, cmap='gray'), plt.title('Operator Sobela')
# plt.subplot(233), plt.imshow(canny_edges, cmap='gray'), plt.title('Operator Canny')
# plt.subplot(234), plt.imshow(laplacian, cmap='gray'), plt.title('Operator Laplace\'a')
# plt.subplot(235), plt.imshow(sobel_x, cmap='gray'), plt.title('Operator Sobela (X)')
# plt.subplot(236), plt.imshow(sobel_y, cmap='gray'), plt.title('Operator Sobela (Y)')
# plt.show()

# import cv2
# import matplotlib.pyplot as plt
# from skimage import metrics
#
# image_class = cv2.imread('obraz.jpg', cv2.IMREAD_GRAYSCALE)
#
# if image_class is None:
#     print("Nie udało się wczytać obrazu.")
# else:
#     sobel_x = cv2.Sobel(image_class, cv2.CV_64F, 1, 0, ksize=5)
#     sobel_y = cv2.Sobel(image_class, cv2.CV_64F, 0, 1, ksize=5)
#     sobel_combined = cv2.addWeighted(cv2.convertScaleAbs(sobel_x), 0.5, cv2.convertScaleAbs(sobel_y), 0.5, 0)
#     canny = cv2.Canny(image_class, 100, 200)
#     laplacian = cv2.Laplacian(image_class, cv2.CV_64F)
#     psnr_sobel = metrics.peak_signal_noise_ratio(image_class, sobel_combined)
#     ssim_sobel = metrics.structural_similarity(image_class, sobel_combined)
#     psnr_canny = metrics.peak_signal_noise_ratio(image_class, canny)
#     ssim_canny = metrics.structural_similarity(image_class, canny)
#     psnr_laplacian = metrics.peak_signal_noise_ratio(image_class, laplacian, data_range=laplacian.max() - laplacian.min())
#     ssim_laplacian = metrics.structural_similarity(image_class, laplacian, data_range=laplacian.max() - laplacian.min())
#     fig = plt.figure(figsize=(14, 6))
#     plt.subplot(2, 4, 1), plt.imshow(image_class, cmap='gray')
#     plt.title('Obraz oryginalny'), plt.xticks([]), plt.yticks([])
#     plt.subplot(2, 4, 2), plt.imshow(sobel_combined, cmap='gray')
#     plt.title(f'Sobela\nPSNR: {psnr_sobel:.2f}\nSSIM: {ssim_sobel:.2f}'), plt.xticks([]), plt.yticks([])
#     plt.subplot(2, 4, 3), plt.imshow(canny, cmap='gray')
#     plt.title(f'Canny\nPSNR: {psnr_canny:.2f}\nSSIM: {ssim_canny:.2f}'), plt.xticks([]), plt.yticks([])
#     plt.subplot(2, 4, 4), plt.imshow(laplacian, cmap='gray')
#     plt.title(f'Laplace\'a\nPSNR: {psnr_laplacian:.2f}\nSSIM: {ssim_laplacian:.2f}'), plt.xticks([]),
#     plt.yticks([])
#     plt.show()
