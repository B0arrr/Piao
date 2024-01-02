# import cv2
# import matplotlib.pyplot as plt
#
# image = cv2.imread('obraz.jpg', cv2.IMREAD_GRAYSCALE)
# _, global_thresholded = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
# threshold_values = [50, 100, 150]
# fig, axes = plt.subplots(2, len(threshold_values) + 1, figsize=(12, 4))
# axes[0, 0].imshow(image, cmap='gray')
# axes[0, 0].set_title('Oryginał')
# axes[1, 0].imshow(global_thresholded, cmap='gray')
# axes[1, 0].set_title('Prog. globalne')
# for i, threshold_value in enumerate(threshold_values):
#     _, thresholded = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
#     axes[0, i + 1].imshow(image, cmap='gray')
#     axes[0, i + 1].set_title(f'Prog. {threshold_value}')
#     axes[1, i + 1].imshow(thresholded, cmap='gray')
#     axes[1, i + 1].set_title(f'Prog. {threshold_value}')
# for ax in axes.ravel():
#     ax.axis('off')
# plt.tight_layout()
# plt.show()

# import cv2
# import matplotlib.pyplot as plt
#
# image = cv2.imread('obraz.jpg', cv2.IMREAD_COLOR)
# image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(image_gray, threshold1=30, threshold2=100)
# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# axes[0].set_title('Oryginał')
# axes[1].imshow(edges, cmap='gray')
# axes[1].set_title('Krawędzie')
# shapes = ['Okręgi', 'Prostokąty', 'Elipsy']
# for shape_type in shapes:
#     detected_objects = []
#     if shape_type == 'Okręgi':
#         detected_objects = [c for c in contours if len(cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)) > 8]
#     elif shape_type == 'Prostokąty':
#         detected_objects = [c for c in contours if len(cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)) == 4]
#     elif shape_type == 'Elipsy':
#         detected_objects = [c for c in contours if len(c) >= 5]
#     image_with_contours = image.copy()
#     cv2.drawContours(image_with_contours, detected_objects, -1, (0, 0, 255), 2)
#     axes[2].imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
#     axes[2].set_title(f'{shape_type}')
# for ax in axes:
#     ax.axis('off')
# plt.tight_layout()
# plt.show()

# import cv2
# import matplotlib.pyplot as plt
#A
# image = cv2.imread('obraz.jpg')
#
# _, segmented_mask = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 150, 255, cv2.THRESH_BINARY)
#
# cv2.imwrite('maska_segmentacji.jpg', segmented_mask)
#
# plt.imshow(segmented_mask, cmap='gray')
# plt.title('Maska segmentacji')
# plt.show()

# import cv2
# import numpy as np
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# import matplotlib.pyplot as plt
#
# image = cv2.imread('obraz.jpg', cv2.IMREAD_GRAYSCALE)
# segmentation_mask = cv2.imread('maska_segmentacji.jpg', cv2.IMREAD_GRAYSCALE)
# height, width = image.shape
# X = np.column_stack((np.arange(height), np.arange(width)))
# y = segmentation_mask.ravel()
# model = make_pipeline(StandardScaler(), SVC(kernel='linear'))
# model.fit(X, y)
# segmented_image = model.predict(X).reshape(height, width)
# fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# axes[0].imshow(image, cmap='gray')
# axes[0].set_title('Oryginał')
# axes[1].imshow(segmentation_mask, cmap='gray')
# axes[1].set_title('Maska segmentacji')
# axes[2].imshow(segmented_image, cmap='gray')
# axes[2].set_title('Segmentacja')
# for ax in axes:
#     ax.axis('off')
# plt.tight_layout()
# plt.show()

# import cv2
# import matplotlib.pyplot as plt
#
# image = cv2.imread('obraz.jpg', cv2.IMREAD_GRAYSCALE)
# _, global_thresholded = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
# threshold_values = [50, 100, 150]
# fig, axes = plt.subplots(2, len(threshold_values) + 1, figsize=(12, 4))
# axes[0, 0].imshow(image, cmap='gray')
# axes[0, 0].set_title('Oryginał')
# axes[1, 0].imshow(global_thresholded, cmap='gray')
# axes[1, 0].set_title('Prog. globalne')
# for i, threshold_value in enumerate(threshold_values):
#     _, thresholded = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
#     axes[0, i+1].imshow(image, cmap='gray')
#     axes[0, i+1].set_title(f'Prog. {threshold_value}')
#     axes[1, i+1].imshow(thresholded, cmap='gray')
#     axes[1, i+1].set_title(f'Prog. {threshold_value}')
# for ax in axes.ravel():
#     ax.axis('off')
# plt.tight_layout()
# plt.show()

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# image = cv2.imread('obraz.jpg', cv2.IMREAD_COLOR)
# image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(image_gray, threshold1=30, threshold2=100)
# contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# axes[0].set_title('Oryginał')
# axes[1].imshow(edges, cmap='gray')
# axes[1].set_title('Krawędzie')
# detected_circles = cv2.HoughCircles(image_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)
# if detected_circles is not None:
#     detected_circles = np.uint16(np.around(detected_circles))
# for circle in detected_circles[0, :]:
#     cv2.circle(image, (circle[0], circle[1]), circle[2], (0, 0, 255), 2)
#     axes[2].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     axes[2].set_title('Segmentacja na podstawie kształtów (okręgi)')
# for ax in axes:
#     ax.axis('off')
# plt.tight_layout()
# plt.show()

# import cv2
# import numpy as np
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# import matplotlib.pyplot as plt
#
# image = cv2.imread('obraz.jpg', cv2.IMREAD_GRAYSCALE)
# segmentation_mask = cv2.imread('maska_segmentacji.jpg', cv2.IMREAD_GRAYSCALE)
# height, width = image.shape
# X = np.column_stack((np.arange(height), np.arange(width)))
# y = segmentation_mask.ravel()
# model = make_pipeline(StandardScaler(), SVC(kernel='linear'))
# model.fit(X, y)
# segmented_image = model.predict(X).reshape(height, width)
# fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# axes[0].imshow(image, cmap='gray')
# axes[0].set_title('Oryginał')
# axes[1].imshow(segmentation_mask, cmap='gray')
# axes[1].set_title('Maska segmentacji')
# axes[2].imshow(segmented_image, cmap='gray')
# axes[2].set_title('Segmentacja przy użyciu uczenia maszynowego (SVM)')
# for ax in axes:
#     ax.axis('off')
# plt.tight_layout()
# plt.show()
