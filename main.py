from ultralytics import YOLO
import torch
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time

detection_model = YOLO("model/detection_model.pt")

# Tải mô hình InceptionResNetV1 từ facenet-pytorch
extract_model = InceptionResnetV1(pretrained='vggface2').eval()

# Hàm tiền xử lý ảnh khuôn mặt
def preprocess_face(face_image):
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    face_image = Image.fromarray(face_image)
    return transform(face_image).unsqueeze(0)  # Thêm chiều batch

# Hàm trích xuất đặc trưng khuôn mặt
def extract_face_embedding(face_image):
    face_tensor = preprocess_face(face_image)
    with torch.no_grad():
        embedding = extract_model(face_tensor)
    return embedding[0].cpu().numpy()  # Trả về vector đặc trưng

# Hàm nhận diện khuôn mặt
def detect_faces(image_path):
    results = detection_model(image_path)
    faces = results[0].boxes.xyxy.cpu().numpy()
    return faces

def extract_identity_embedding(folder_path):
    embeddings = {}
    for label in os.listdir(folder_path):
        embeddings[label] = []
        label_path = os.path.join(folder_path, label)
        for image in os.listdir(label_path):
            image_path = os.path.join(label_path, image)
            embedding = extract_face_embedding(cv2.imread(image_path))
            embeddings[label].append(embedding)
    return embeddings

def nearest_face(face_embedding, identity_embedding):
    distances = {}
    for label, embeddings in identity_embedding.items():
        distances[label] = []
        for embedding in embeddings:
            distance = np.linalg.norm(face_embedding - embedding)
            distances[label].append(distance)
    min_distance = np.inf
    min_label = None
    for label, label_distances in distances.items():
        if min(label_distances) < min_distance:
            min_distance = min(label_distances)
            min_label = label
    return min_label, min_distance

def identify_faces(faces, identity_embedding, image):
    identified_faces = []
    for face in faces:
        x1, y1, x2, y2 = face
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        face_image = image[y1:y2, x1:x2]
        face_embedding = extract_face_embedding(face_image)
        label, distance = nearest_face(face_embedding, identity_embedding)
        identified_faces.append((label, face, distance))

    return identified_faces


# # image_path = "test/actual_test/test_identify_1.jpg"
# # image_path = "test/actual_test/test_identify_2.png"
# # image_path = "test/actual_test/test_identify_3.jpeg"
# # image_path = "test/actual_test/test_identify_4.jpg"
# # image_path = "test/actual_test/image_1.jpg"
# # image_path = "test/actual_test/trung_phong_2.jpg"
# # image_path = "test/actual_test/image_16.jpg"
# folder_test = "test/actual_test"
# folder_path = "identity"

# if __name__ == '__main__':
#     identity_embedding = extract_identity_embedding(folder_path)

#     for image in os.listdir(folder_test):
#         image_path = os.path.join(folder_test, image)

#         faces = detect_faces(image_path)

#         image = cv2.imread(image_path)

#         identified_faces = identify_faces(faces, identity_embedding, image)

#         print(identified_faces)

#         for label, face, distance in identified_faces:
#             x1, y1, x2, y2 = face
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
#             cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             cv2.putText(image, label, (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
#             cv2.putText(image, str(distance), (x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#         plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         plt.show()


# Đường dẫn thư mục chứa các ảnh
# folder_test = "test/actual_test"
# folder_path = "identity"

# # Mở camera và chụp ảnh liên tục
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Không thể mở camera.")
#     exit()

# identity_embedding = extract_identity_embedding(folder_path)

# while True:
#     # Đọc khung hình từ camera
#     ret, frame = cap.read()
#     if not ret:
#         print("Không thể đọc khung hình từ camera.")
#         break
    
#     # Phát hiện khuôn mặt trong ảnh
#     faces = detect_faces(frame)

#     # Nhận diện khuôn mặt
#     identified_faces = identify_faces(faces, identity_embedding, frame)

#     # Vẽ hình chữ nhật và thêm thông tin nhận diện
#     for label, face, distance in identified_faces:
#         x1, y1, x2, y2 = face
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

#         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
#         cv2.putText(frame, f"Distance: {distance:.2f}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#     # Hiển thị khung hình cho người dùng
#     # cv2.imshow("Camera Feed", frame)
#     plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     plt.show(block=False)
#     plt.pause(0.001)

#     # Nhấn 'q' để thoát khỏi vòng lặp
#     if 0xFF == ord('q'):
#         break

# # Giải phóng camera và đóng cửa sổ
# cap.release()
# cv2.destroyAllWindows()

# print("Hoàn thành.")
