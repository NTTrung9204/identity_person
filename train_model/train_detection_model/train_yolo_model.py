from ultralytics import YOLO
import torch

# Kiểm tra xem GPU có khả dụng không
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device used:", device)   

# Tạo mô hình YOLOv5
model = YOLO('yolov5su.pt')

# Khởi tạo optimizer (có thể sử dụng optimizer mặc định của YOLO hoặc tùy chỉnh)
optimizer = torch.optim.Adam(model.model.parameters(), lr=0.01)

if __name__ == '__main__':

    # Huấn luyện mô hình
    results = model.train(
        data='config.yaml',        # File cấu hình dữ liệu
        epochs=500,                 # Số epoch
        batch=8,                    # Giảm batch size để tránh lỗi out of memory
        imgsz=640,                  # Giảm kích thước ảnh
        device=device,              # Sử dụng GPU nếu khả dụng
        workers=4,                  # Số lượng luồng xử lý dữ liệu
        augment=True,               # Sử dụng data augmentation
        patience=0,                 # Số epoch không cải thiện trước khi dừng
    )

    try:
        # Lưu checkpoint (mô hình, optimizer và các thông tin khác)
        checkpoint = {
            'model_state_dict': model.model.state_dict(),  # Trọng số của mô hình
            'optimizer_state_dict': optimizer.state_dict(), # Trạng thái của optimizer
            'epoch': 500,  # Số epoch hiện tại
            'results': results,  # Kết quả huấn luyện
        }

        torch.save(checkpoint, 'detection_model_torch_v1.pth')
    except:
        model.save('detection_model_yolo_v1.pt')
