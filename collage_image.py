import cv2
import time
import os
from matplotlib import pyplot as plt

# Kiểm tra và tạo thư mục "result_collage" nếu chưa tồn tại
if not os.path.exists('result_collage'):
    os.makedirs('result_collage')

# Mở camera (0 là chỉ định camera mặc định của máy tính)
cap = cv2.VideoCapture(0)

# Kiểm tra xem camera có mở thành công không
if not cap.isOpened():
    print("Không thể mở camera")
    exit()

# Lấy thời gian bắt đầu
start_time = time.time()

# Số lượng ảnh muốn chụp
num_images = 100
counter = 0

# Giảm độ phân giải của ảnh (640x480) để xử lý nhanh hơn
frame_width, frame_height = 640, 480

rect_x, rect_y, rect_w, rect_h = 200, 100, 200, 250

# Lấy ảnh trong vòng 5 giây
while counter < num_images:
    # Đọc hình ảnh từ camera
    ret, frame = cap.read()
    
    if not ret:
        print("Không thể đọc hình ảnh từ camera")
        break
    
    # Giảm độ phân giải ảnh (640x480)
    frame_resized = cv2.resize(frame, (frame_width, frame_height))

    # Vẽ hình chữ nhật lên ảnh để người dùng biết nơi cần đặt mặt
    cv2.rectangle(frame_resized, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 255, 0), 2)

    # Hiển thị ảnh cho người dùng
    plt.imshow(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
    plt.show(block=False)
    plt.pause(0.01)



    if time.time() - start_time > 5:
        # Cắt ảnh trong vùng hình chữ nhật và lưu
        cropped_face = frame_resized[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w]
        if cropped_face.size > 0:
            image_filename = f"result_collage/image_{counter+1}.jpg"
            cv2.imwrite(image_filename, cropped_face)
            print(f"Đã lưu {image_filename}")
            counter += 1

    # Kiểm tra thời gian (5 giây)
    if time.time() - start_time > 65:
        break

# Giải phóng camera và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()

print("Đã chụp xong 100 ảnh hoặc hết 5 giây.")
