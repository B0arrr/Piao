# import cv2
# import numpy as np
#
#
# def wczytaj_obraz(sciezka):
#     obraz = cv2.imread(sciezka)
#     return obraz
#
#
# def eliminacja_zarysowan(obraz):
#     obraz_wyjsciowy = cv2.medianBlur(obraz, 5)
#     return obraz_wyjsciowy
#
#
# def wypelnij_brakujace_fragmenty(obraz):
#     maska = cv2.inRange(obraz, np.array([0, 0, 0]), np.array([10, 10, 10]))
#     obraz_wyjsciowy = cv2.inpaint(obraz, maska, 3, cv2.INPAINT_TELEA)
#     return obraz_wyjsciowy
#
#
# def testowanie_i_ocena_skutecznosci(obraz_wejsciowy):
#     obraz_po_elim_zarysowan = eliminacja_zarysowan(obraz_wejsciowy)
#     obraz_po_wypelnieniu = wypelnij_brakujace_fragmenty(obraz_po_elim_zarysowan)
#     cv2.imshow('Obraz przed', obraz_wejsciowy)
#     cv2.imshow('Obraz po eliminacji zarysowań', obraz_po_elim_zarysowan)
#     cv2.imshow('Obraz po wypełnieniu brakujących fragmentów', obraz_po_wypelnieniu)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
#
#
# if __name__ == "__main__":
#     stara_fotografia = wczytaj_obraz("obraz.jpg")
#     testowanie_i_ocena_skutecznosci(stara_fotografia)

# import cv2
# import numpy as np
#
#
# def create_mask(image):
#     mask = cv2.inRange(image, np.array([0, 0, 0]), np.array([50, 50, 50]))
#     return mask
#
#
# if __name__ == "__main__":
#     stara_fotografia = wczytaj_obraz("obraz.jpg")
#     maska = create_mask(stara_fotografia)
#     cv2.imwrite("mask.jpg", maska)
#     testowanie_i_ocena_skutecznosci(stara_fotografia)

# import cv2
# import matplotlib.pyplot as plt
#
#
# def restore_old_photo(image_path, mask_path):
#     old_photo = cv2.imread(image_path)
#     damaged_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#     restored_photo = cv2.inpaint(old_photo, damaged_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.imshow(cv2.cvtColor(old_photo, cv2.COLOR_BGR2RGB))
#     plt.title("Old Photo")
#     plt.axis("off")
#     plt.subplot(1, 2, 2)
#     plt.imshow(cv2.cvtColor(restored_photo, cv2.COLOR_BGR2RGB))
#     plt.title("Restored Photo")
#     plt.axis("off")
#     plt.show()
#
#
# if __name__ == '__main__':
#     restore_old_photo("obraz.jpg", "mask.jpg")

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_image(file_path):
    """
    Wczytuje obraz z podanej ścieżki.
    Parameters:
    - file_path (str): Ścieżka do pliku obrazu.
    Returns:
    - image (numpy.ndarray): Wczytany obraz.
    """
    image = cv2.imread(file_path)
    return image


def create_damage_mask(image_shape):
    """
    Tworzy pustą maskę o wymiarach zgodnych z obrazem.
    Parameters:
    - image_shape (tuple): Kształt obrazu (wysokość, szerokość, liczba kanałów).
    Returns:
    - damage_mask (numpy.ndarray): Pusta maska uszkodzeń.
    """
    damage_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    return damage_mask


def mark_damaged_areas(image, damage_mask):
    """
    Oznacza obszary uszkodzeń na obrazie na podstawie podanej maski.
    Parameters:
    - image (numpy.ndarray): Obraz do oznaczenia.
    - damage_mask (numpy.ndarray): Maska uszkodzeń.
    Returns:
    - marked_image (numpy.ndarray): Obraz z oznaczonymi uszkodzonymi obszarami.
    """
    marked_image = image.copy()
    marked_image[damage_mask > 0] = [0, 0, 255]  # Oznaczenie na czerwono
    return marked_image


def restore_damaged_areas(image, damage_mask):
    """
    Przywraca uszkodzone obszary na obrazie przy użyciu inpaintingu.
    Parameters:
    - image (numpy.ndarray): Obraz z uszkodzeniami.
    - damage_mask (numpy.ndarray): Maska uszkodzeń.
    Returns:
    - restored_image (numpy.ndarray): Obraz z przywróconymi obszarami.
    """
    restored_image = cv2.inpaint(image, damage_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return restored_image


def display_images(original_image, marked_image, restored_image):
    """
    Wyświetla obraz przed i po procesie restauracji.
    Parameters:
    - original_image (numpy.ndarray): Oryginalny obraz.
    - marked_image (numpy.ndarray): Obraz z oznaczonymi uszkodzonymi obszarami.
    - restored_image (numpy.ndarray): Obraz z przywróconymi obszarami.
    """
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB))
    plt.title("Damaged Areas Marked")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(restored_image, cv2.COLOR_BGR2RGB))
    plt.title("Restored Image")
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    old_photo = load_image("obraz.jpg")
    damage_mask = create_damage_mask(old_photo.shape)
    restored_photo = restore_damaged_areas(old_photo, damage_mask)
    display_images(old_photo, mark_damaged_areas(old_photo, damage_mask), restored_photo)
