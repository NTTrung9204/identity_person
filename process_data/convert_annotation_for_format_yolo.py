import json
import os
import cv2

# Đường dẫn tới thư mục chứa cả tệp JSON và ảnh gốc
data_folder = 'dataset/dataset_original'
# Thư mục để lưu ảnh đã padding
target_folder = 'dataset/dataset_processed/train/images'
# Kích thước mục tiêu
target_size = 640

# Hàm để thêm padding và trả về kích thước mới cùng thông tin padding
def add_padding_and_get_new_size(image_path):
    # Đọc ảnh từ file
    image = cv2.imread(image_path)

    # Kích thước ảnh gốc
    h, w, _ = image.shape

    # Tính toán padding
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    # Tính toán padding
    top_pad = (target_size - new_h) // 2
    bottom_pad = target_size - new_h - top_pad
    left_pad = (target_size - new_w) // 2
    right_pad = target_size - new_w - left_pad

    return new_w, new_h, top_pad, left_pad

# Duyệt qua tất cả các tệp trong thư mục dữ liệu
for file in os.listdir(data_folder):
    if file.endswith('.json'):
        json_path = os.path.join(data_folder, file)
        
        # Đọc tệp JSON
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Lấy chiều cao và chiều rộng của ảnh gốc
        image_width = data['imageWidth']
        image_height = data['imageHeight']

        # Khởi tạo danh sách để lưu các nhãn
        annotations = []

        # Lấy tên ảnh tương ứng với tệp JSON
        image_name = file.replace('.json', '.png')
        image_path = os.path.join(data_folder, image_name)

        # Kiểm tra xem ảnh có tồn tại không
        if not os.path.exists(image_path):
            print(f"Image {image_name} not found for {file}. Trying with jpg...")
        
        image_name = file.replace('.json', '.jpg')
        image_path = os.path.join(data_folder, image_name)

        # Kiểm tra xem ảnh có tồn tại không
        if not os.path.exists(image_path):
            print(f"Image {image_name} not found for {file}. Skipping...")
            continue

        # Lấy thông tin kích thước và padding
        new_w, new_h, top_pad, left_pad = add_padding_and_get_new_size(image_path)

        # Chuyển đổi từng đối tượng trong JSON sang định dạng YOLO
        for shape in data['shapes']:
            label = shape['label']
            points = shape['points']

            # Tính toán tọa độ gốc
            x_min = min(points[0][0], points[1][0])
            x_max = max(points[0][0], points[1][0])
            y_min = min(points[0][1], points[1][1])
            y_max = max(points[0][1], points[1][1])

            # Tính toán giá trị theo định dạng YOLO sau khi đã padding
            x_center = ((x_min + x_max) / 2) * (new_w / image_width) + left_pad
            y_center = ((y_min + y_max) / 2) * (new_h / image_height) + top_pad
            width = (x_max - x_min) * (new_w / image_width)
            height = (y_max - y_min) * (new_h / image_height)

            # Chuyển đổi sang tỷ lệ
            x_center /= target_size
            y_center /= target_size
            width /= target_size
            height /= target_size

            # Thêm vào danh sách nhãn
            annotations.append(f"{label} {x_center} {y_center} {width} {height}")

        # Xuất ra tệp văn bản với định dạng YOLO
        output_file_path = os.path.join("dataset/dataset_processed/train/labels/", f'{file.replace(".json", ".txt")}')
        with open(output_file_path, 'w') as out_file:
            for annotation in annotations:
                out_file.write(annotation + '\n')

        print(f'Annotations for {file} have been saved to {output_file_path}')
