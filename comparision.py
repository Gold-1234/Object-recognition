import cv2
from ultralytics import YOLO
import time
import mediapipe as mp
import csv

model = YOLO('yolov8n.pt')  # Downloads automatically if not present

BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

csv_file = open('detection_metrics.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame', 'Timestamp_sec', 'YOLO_Inference_ms', 'YOLO_Objects', 'YOLO_Speed_objs_per_ms', 
                     'MP_Inference_ms', 'MP_Objects', 'MP_Speed_objs_per_ms'])

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path='./efficientdet_lite2.tflite'),
    score_threshold=0.25,
    running_mode=VisionRunningMode.VIDEO)

# Open the video file
cap = cv2.VideoCapture('./media/video3.mp4')
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
detection_interval = int(fps/5)

with ObjectDetector.create_from_options(options) as detector:
	frame_index = 0
	yolo_detections = []
	mp_detections = []

	while True:
		ret, frame = cap.read()
		if not ret:
			print("End of video reached.")
			break

		number = 0
		num = 0

		frame_index += 1
		mp_start = time.time()
		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
		mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame) 

		if frame_index%detection_interval == 0:
			start_yolo = time.time()
			yolo_results = model(frame)  
			timestamp_sec = frame_index/fps

			
			mp_result = detector.detect_for_video(mp_image, int(timestamp))
			

			yolo_detections = []
			mp_detections = []
			for result in yolo_results:
				boxes = result.boxes
				for box in boxes: 
					x1, y1, x2, y2 = map(int, box.xyxy[0])
					conf = box.conf[0]
					cls = int(box.cls[0])
					label = model.names[cls]
					yolo_detections.append((x1, y1, x2, y2, conf, label, num))
					num = len(yolo_detections)
			# print(f"Frame {frame_index} | Inference Time: {yolo_inference_time:.2f} ms")
			yolo_inference_time = (time.time() - start_yolo) * 1000
			yolo_speed = num/yolo_inference_time

			for detection in mp_result.detections:
				bbox = detection.bounding_box
				category = detection.categories[0].category_name
				x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
				score = detection.categories[0].score
				mp_detections.append((x,y,w,h,category,score, number))
				number = len(mp_detections)
			mp_inference_time = (time.time() - mp_start) * 1000
			mp_speed = number/mp_inference_time
			
			print(f"Frame {frame_index}:")
			print(f"  YOLOV8 inference time: {yolo_inference_time} || number of objects: {num} || speed: {yolo_speed}")
			print(f"  Mediapipe inference time: {mp_inference_time} || number of objects: {number} || speed: {mp_speed}")
			csv_writer.writerow([frame_index, timestamp_sec, yolo_inference_time, num, yolo_speed, 
                                 mp_inference_time, number, mp_speed])
		for x1, y1, x2, y2, conf, label, num in yolo_detections:
			cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Red box
			text_position = (max(0, x1), max(20, y1 - 10))
			# print(f"Detected: {label} | Score: {conf:.2f} | | Bounding Box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
			cv2.putText(
				frame, f"{label} ({conf:.2f})", text_position,
				cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1
			)
			

		for x, y, w, h, category, score, number  in mp_detections:
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
			text_position = (max(0, x), max(20, y - 10))  
			cv2.putText(
                frame,
                f"{category} ({score:.2f})",
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 255, 0),
                1
            )

			
			
			
		cv2.imshow("YOLOv8 Object Detection", frame)

		if cv2.waitKey(30) & 0xFF == ord('q'):
			break

cap.release()
cv2.destroyAllWindows()