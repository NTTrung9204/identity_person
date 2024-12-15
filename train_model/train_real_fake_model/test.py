import torch
from torchvision import transforms, models
from PIL import Image
import os
from imutils.video import VideoStream
import imutils
import time
import cv2

# Đặt đường dẫn tới mô hình đã lưu
model_path = 'real_fake_model.pth'  # Đường dẫn tới mô hình đã huấn luyện

# Hàm load lại mô hình
def load_model(model_path):
    # Sử dụng mô hình ResNet18 đã huấn luyện sẵn
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # Chỉnh lại layer cuối cho 2 lớp
    model.load_state_dict(torch.load(model_path))  # Tải trọng số của mô hình
    model.eval()  # Đặt mô hình vào chế độ đánh giá (evaluation)
    return model

# Hàm nhận ảnh đầu vào và dự đoán
def predict_image(model, image_array):
    # Định nghĩa các phép biến đổi ảnh
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Thay đổi kích thước ảnh về 128x128
        transforms.ToTensor(),  # Chuyển ảnh thành Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa
    ])
    
    # Mở ảnh từ đường dẫn
    # image = Image.open(image_path)
    image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    
    # Áp dụng các phép biến đổi vào ảnh
    image = transform(image).unsqueeze(0)  # Thêm batch dimension (1, 3, 128, 128)
    
    # Dự đoán với mô hình
    with torch.no_grad():  # Tắt tính toán gradient để tăng tốc
        outputs = model(image)  # Thực hiện dự đoán
        _, predicted = torch.max(outputs, 1)  # Lấy nhãn có xác suất cao nhất
    
    # Lớp dự đoán
    class_names = ['fake', 'real']  # Danh sách tên lớp
    return class_names[predicted.item()]  # Trả về tên lớp dự đoán

# Example usage:
# Load mô hình đã huấn luyện
model = load_model(model_path)

# Nhận ảnh đầu vào và dự đoán
# image_path = "dataset_real_face/face_24878.jpg"  # Đường dẫn tới ảnh cần dự đoán
# image_test = cv2.imread(image_path)
# prediction = predict_image(model, image_test)
# print(f"Predicted class: {prediction}")

# cv2.imshow("Image", image_test)



vs = VideoStream(src=0).start()

time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # Vẽ hình vuông ở giữa frame
    cv2.rectangle(frame, (150, 50), (250, 150), (0, 255, 0), 2)

    # Cắt và lưu ảnh
    roi = frame[50:150, 150:250]
    prediction = predict_image(model, roi)
    print(f"Predicted class: {prediction}")
    cv2.imshow("ROI", roi)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()