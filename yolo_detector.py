from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt") 
cap = cv2.VideoCapture(0)

def something_to_code():
    ret, frame = cap.read()
    if not ret:
        return frame, 0

    results = model(frame)
    face_count = 0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0: 
                face_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Anonymize (blur)
                roi = frame[y1:y2, x1:x2]
                roi = cv2.GaussianBlur(roi, (35, 35), 30)
                frame[y1:y2, x1:x2] = roi

    return frame, face_count
