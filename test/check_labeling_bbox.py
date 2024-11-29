import cv2
import os
from matplotlib import pyplot as plt

# Đường dẫn tới thư mục chứa ảnh và nhãn
# img_dir = 'dataset/images/training'
# label_dir = 'dataset/labels/training'
img_dir = 'dataset/dataset_processed/train/images'
label_dir = 'dataset/dataset_processed/train/labels'

# Đọc tất cả các file ảnh
img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

for img_file in img_files:
    print(img_file)
    img_path = os.path.join(img_dir, img_file)
    label_path = os.path.join(label_dir, img_file.replace('.jpg', '.txt'))
    
    # Đọc ảnh
    img = cv2.imread(img_path)
    
    # Lấy kích thước ảnh
    height, width, _ = img.shape

    try:
        # Đọc file nhãn
        with open(label_path, 'r') as f:
            labels = f.readlines()
    except:
        continue
    
    for label in labels:
        label = label.strip().split()

        # Duyệt qua từng bounding box (mỗi bounding box có 5 giá trị)
        class_id = int(label[0])
        for i in range(1, len(label), 4):
            x_center, y_center, w, h = map(float, label[i:i+4])
            
            # Chuyển đổi tọa độ từ tỷ lệ sang pixel
            x_center *= width
            y_center *= height
            w *= width
            h *= height
            
            # Tính toán góc trên bên trái và dưới bên phải của bounding box
            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)
            
            # Vẽ bounding box lên ảnh
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Hiển thị class_id trên bounding box
            cv2.putText(img, str(class_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Hiển thị ảnh
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

