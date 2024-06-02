import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from make_landmarks import draw_landmarks_on_image

import numpy as np

model_path = 'pose_landmarker_full.task'

# Tạo một đối tượng PoseLandmarker với chế độ video
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=2)

# Sử dụng context manager để đảm bảo giải phóng tài nguyên sau khi sử dụng
with PoseLandmarker.create_from_options(options) as landmarker:
    # Sử dụng OpenCV để tải video đầu vào
    cap = cv2.VideoCapture('D:\\demo-mediapipe\\image\\Durazzo.mp4')
    
    # Kiểm tra xem video có được mở thành công hay không
    if not cap.isOpened():
        print("Không thể mở video.")
        exit()
    
    # Lấy tốc độ khung hình (fps) của video
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps * 1000)
    # Lặp qua từng khung hình trong video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Lấy timestamp của khung hình hiện tại
        frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        
        # Chuyển đổi khung hình từ BGR sang RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Chuyển đổi khung hình sang đối tượng Image của MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Thực hiện phát hiện landmarks
        pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        
        print(pose_landmarker_result)
        
        # Vẽ các landmarks lên khung hình
        annotated_image = draw_landmarks_on_image(rgb_frame, pose_landmarker_result)
        
        # Chuyển đổi khung hình từ RGB sang BGR để hiển thị với OpenCV
        bgr_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        
        # Hiển thị khung hình đã được vẽ landmarks
        cv2.imshow('Pose Landmarks', bgr_annotated_image)

        # Đợi 1ms và thoát nếu nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()
