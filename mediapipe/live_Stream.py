import mediapipe as mp
import cv2
import time

BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def print_result(result: DetectionResult, output_image: mp.Image, timestamp_ms: int):
    print('detection result: {}'.format(result))
    
options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path='./efficientdet_lite2.tflite'),
    max_results=5,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

cap = cv2.VideoCapture('video.mp4')  
fps = cap.get(cv2.CAP_PROP_FPS)
e = cap.get(cv2.CAP_PROP_FRAME_COUNT)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

with ObjectDetector.create_from_options(options) as detector:
    frame_index = 0
    detection_interval = 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_index = frame_index + 1
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        frame_timestamp_ms = 1000 * frame_index / fps
       
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)   

        if frame_index%detection_interval==0:
            result = detector.detect_async(mp_image, int(timestamp))

        # Draw bounding boxes
       
            for detection in result.detections:
                bbox = detection.bounding_box
                x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
                category = detection.categories[0].category_name
                score = detection.categories[0].score

                print(f"Detected: {category} | Score: {score:.2f}")
                print(f"Bounding Box: x={x}, y={y}, w={w}, h={h}")

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                text_position = (max(0, x), max(20, y - 10))  
                cv2.putText(
                    frame,
                    f"{category} ({score:.2f})",
                    text_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()