import mediapipe as mp
import cv2
import time

BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path='../models/efficientdet_lite2.tflite'),
    score_threshold=0.25,
    running_mode=VisionRunningMode.VIDEO)

cap = cv2.VideoCapture('../media/video.mp4')  
fps = cap.get(cv2.CAP_PROP_FPS)
detection_interval = int(fps/5)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_index = 0
last_detections = []
with ObjectDetector.create_from_options(options) as detector:

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)   

        if frame_index%detection_interval==0:
            mp_start = time.time()
            result = detector.detect_for_video(mp_image, int(timestamp))
            inference_time = (time.time() - mp_start) * 1000

            last_detections = []
            for detection in result.detections:
                bbox = detection.bounding_box
                x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
                category = detection.categories[0].category_name
                score = detection.categories[0].score
                last_detections.append((x,y,w,h,category,score))
            
            print(f"Detected: {category} | Score: {score:.2f} | time: {inference_time}")
            print(f"Bounding Box: x={x}, y={y}, w={w}, h={h}")
       
        for x, y, w, h, category, score in last_detections:
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

        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()