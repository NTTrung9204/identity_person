import cv2
import numpy as np
import os

folder_path = 'dataset/dataset_original'
target_folder = 'dataset/dataset_processed/train/images'
target_size = 640

def add_padding(path_name, target_size, target_path):
    # Đọc ảnh từ file
    image = cv2.imread(path_name)

    # Kích thước ảnh gốc
    h, w, _ = image.shape

    # Tính toán padding
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    # Thay đổi kích thước ảnh nhưng vẫn giữ nguyên tỉ lệ
    resized_image = cv2.resize(image, (new_w, new_h))

    # Tính toán padding để đưa ảnh về kích thước vuông 640x640
    top_pad = (target_size - new_h) // 2
    bottom_pad = target_size - new_h - top_pad
    left_pad = (target_size - new_w) // 2
    right_pad = target_size - new_w - left_pad

    # Thêm padding
    padded_image = cv2.copyMakeBorder(resized_image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # Lưu ảnh đã padding
    target_path = target_path.replace('.png', '.jpg')
    cv2.imwrite(target_path, padded_image)

for filename in os.listdir(folder_path):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        image_path = os.path.join(folder_path, filename)
        target_path = os.path.join(target_folder, filename)
        add_padding(image_path, target_size, target_path)
