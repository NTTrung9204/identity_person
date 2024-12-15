import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

# Thiết lập các tham số
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# Định nghĩa các phép biến đổi cho ảnh (Augmentation)
transform = transforms.Compose([
    transforms.Resize((128, 128)),    # Resize ảnh về kích thước 128x128
    transforms.ToTensor(),            # Chuyển ảnh thành Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa
])

# Tạo dataset từ thư mục 'real' và 'fake'
data_dir = './'  # Đặt đường dẫn đến thư mục chứa "real" và "fake"
train_dataset = datasets.ImageFolder(
    os.path.join(data_dir),
    transform=transform
)

# Tạo DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Xây dựng mô hình CNN (Dùng pre-trained ResNet18)
model = models.resnet18(pretrained=True)  # Dùng mô hình ResNet18 đã được huấn luyện sẵn
model.fc = nn.Linear(model.fc.in_features, 2)  # Chỉnh lại layer cuối để phân loại 2 lớp

# Di chuyển mô hình vào GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Định nghĩa loss function và optimizer
criterion = nn.CrossEntropyLoss()  # Loss function cho bài toán phân loại
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# List lưu loss và accuracy sau mỗi epoch
losses = []
accuracies = []

# Huấn luyện mô hình
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_preds / total_preds

        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}')
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)

# Bắt đầu huấn luyện
train_model(model, train_loader, criterion, optimizer, num_epochs)

# Lưu mô hình sau khi huấn luyện
torch.save(model.state_dict(), 'real_fake_model.pth')

# Vẽ biểu đồ Loss và Accuracy nếu cần
plt.plot(losses, label='Training Loss')
plt.plot(accuracies, label='Training Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.show()

print("Hoàn thành.")
