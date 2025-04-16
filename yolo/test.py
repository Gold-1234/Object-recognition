import cv2
from ultralytics import YOLO
import time

model = YOLO('yolov8n.pt')  # Downloads automatically if not present

# Open the video file
cap = cv2.VideoCapture('../media/video3.mp4')
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
detection_interval = int(fps/5)

frame_index = 0
last_detections = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video reached.")
        break

    frame_index += 1

    if frame_index%detection_interval == 0:
        start_yolo = time.time()
        results = model(frame)  
        inference_time = (time.time() - start_yolo) * 1000
        mp_count = 0
        yolo_count = 0

        last_detections = []
        for result in results:
            
            boxes = result.boxes
            for box in boxes: 
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = model.names[cls]
                last_detections.append((x1, y1, x2, y2, conf, label))
        print(f"Frame {frame_index} | Inference Time: {inference_time:.2f} ms")
           

    for x1, y1, x2, y2, conf, label in last_detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Red box
        text_position = (max(0, x1), max(20, y1 - 10))
        print(f"Detected: {label} | Score: {conf:.2f} | | Bounding Box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        cv2.putText(
            frame, f"{label} ({conf:.2f})", text_position,
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1
        )

    cv2.imshow("YOLOv8 Object Detection", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()