import torch
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
from PIL import Image
import cv2

# Tải mô hình InceptionResNetV1 từ facenet-pytorch
model = InceptionResnetV1(pretrained='vggface2').eval()

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
        embedding = model(face_tensor)
    return embedding[0].cpu().numpy()  # Trả về vector đặc trưng


image_path = "identity/son_tung/son_tung_1.png"

# img = Image.open(image_path)
img = cv2.imread(image_path)

print(extract_face_embedding(img))
