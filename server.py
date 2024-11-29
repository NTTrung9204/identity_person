import random
from flask import Flask, request, send_file
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
from main import detect_faces, identify_faces, extract_identity_embedding

app = Flask(__name__)
CORS(app)

folder_path = "identity"

identity_embedding = extract_identity_embedding(folder_path)

@app.route('/process_images', methods=['POST'])
def process_images():
    # Kiểm tra nếu không có file
    if 'images' not in request.files:
        return "No images provided", 400

    images = request.files.getlist('images')  # Nhận danh sách các ảnh
    print(f"Received {len(images)} images")
    processed_images = []

    result_label = None
    min_distance = np.inf
    result_face = None

    # Chuyển từng ảnh thành đen trắng và lưu vào danh sách
    for file in images:
        img = Image.open(file.stream)
        img = np.array(img)
        faces = detect_faces(img)
        identified_faces = identify_faces(faces, identity_embedding, img)

        for label, face, distance in identified_faces:
            if distance < min_distance:
                min_distance = distance
                result_label = label
                result_face = face
        
        # return {"result": result_label, "distance": min_distance}, 200

    if result_label:
        # gửi lại min_distance, result_label, result_face cho client
        # result_face = io.BytesIO(result_face)
        # result_face.seek(0)
        print(result_label, min_distance)
        if min_distance < 0.5:
            return {
                "result": result_label,
                "distance": float(min_distance)  # Chuyển đổi thành kiểu float chuẩn
            }
        else:
            return {
                "result": "Unknown",
                "distance": float(min_distance)
            }





    #     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #     # Chuyển ảnh về định dạng gửi lại client
    #     _, buffer = cv2.imencode('.jpg', gray_img)
    #     processed_images.append(buffer)

    # # Chọn một ảnh ngẫu nhiên từ danh sách
    # if processed_images:
    #     random_image = random.choice(processed_images)
    #     random_image_io = io.BytesIO(random_image)
    #     random_image_io.seek(0)
    #     return send_file(random_image_io, mimetype='image/jpeg')

    return "No processed images", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
