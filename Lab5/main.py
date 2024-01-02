# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# def convert_color_spaces(image):
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     return rgb_image, hsv_image, lab_image
#
#
# def white_balance_correction(image):
#     lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     l_channel, a_channel, b_channel = cv2.split(lab_image)
#     l_channel_avg = np.mean(l_channel)
#     a_channel = a_channel - ((l_channel_avg - 128) * (a_channel.std() / 128.0))
#     b_channel = b_channel - ((l_channel_avg - 128) * (b_channel.std() / 128.0))
#     corrected_lab_image = cv2.merge([l_channel, l_channel, l_channel])
#     corrected_image = cv2.cvtColor(corrected_lab_image, cv2.COLOR_LAB2BGR)
#     return corrected_image
#
#
# def remove_red_eye_effect(image):
#     lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     l_channel, a_channel, b_channel = cv2.split(lab_image)
#     red_mask = cv2.inRange(b_channel, 150, 255)
#     eye_mask = red_mask > 0
#     l_channel[eye_mask] = np.mean(l_channel)
#     corrected_lab_image = cv2.merge([l_channel, a_channel, b_channel])
#     corrected_image = cv2.cvtColor(corrected_lab_image, cv2.COLOR_LAB2BGR)
#     return corrected_image
#
#
# image_color_space = cv2.imread("obraz.jpg")
# image_white_balance = cv2.imread("obraz.jpg")
# image_red_eye = cv2.imread("obraz.jpg")
# rgb_image, hsv_image, lab_image = convert_color_spaces(image_color_space)
# corrected_white_balance_image = white_balance_correction(image_white_balance)
# corrected_red_eye_image = remove_red_eye_effect(image_red_eye)
# plt.figure(figsize=(15, 5))
# plt.subplot(1, 4, 1)
# plt.imshow(cv2.cvtColor(image_color_space, cv2.COLOR_BGR2RGB))
# plt.title("Original Image")
# plt.axis("off")
# plt.subplot(1, 4, 2)
# plt.imshow(rgb_image)
# plt.title("RGB Image")
# plt.axis("off")
# plt.subplot(1, 4, 3)
# plt.imshow(corrected_white_balance_image)
# plt.title("White Balance Correction")
# plt.axis("off")
# plt.subplot(1, 4, 4)
# plt.imshow(corrected_red_eye_image)
# plt.title("Red Eye Correction")
# plt.axis("off")
# plt.show()


import cv2
import matplotlib.pyplot as plt

original_image = cv2.imread("obraz.jpg")
rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title("Original Image (BGR)")
plt.axis("off")
plt.subplot(1, 3, 2)
plt.imshow(rgb_image)
plt.title("RGB Image")
plt.axis("off")
plt.subplot(1, 3, 3)
plt.imshow(hsv_image)
plt.title("HSV Image")
plt.axis("off")
plt.show()
