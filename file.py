from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
from text_to_speech.main import speak
from poseDetection.pose import classify_from_frame
import time

app = Flask(__name__)

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Model path
model_path = './models/efficientdet_lite2.tflite'
score_threshold = 0.4

# Variables to persist across frames
last_detections = []
last_spoken = None
last_spoken_time = 0
cooldown_seconds = 3

def initialize_detector():
    options = ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        score_threshold=score_threshold,
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=process_result
    )
    return ObjectDetector.create_from_options(options)

def process_result(result, output_image, timestamp_ms):
    global last_detections, last_spoken, last_spoken_time
    last_detections = []

    for detection in result.detections:
        bbox = detection.bounding_box
        x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
        category = detection.categories[0].category_name
        score = detection.categories[0].score
        last_detections.append((x, y, w, h, category, score))

    if last_detections:
        x, y, w, h, category, score = last_detections[-1]
        print(f"Detected: {category} | Score: {score:.2f}")

        now = time.time()
        if category != last_spoken or (now - last_spoken_time) > cooldown_seconds:
            speak(f"Detected {category} with score {score:.2f}")
            last_spoken = category
            last_spoken_time = now

def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    detection_interval = int(fps / 2)
    frame_index = 0
    start_time = time.time() * 1000

    detector = initialize_detector()

    with detector:
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame_index += 1

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.time() * 1000 - start_time)

            if frame_index % detection_interval == 0:
                detector.detect_async(mp_image, timestamp_ms)

            for x, y, w, h, category, score in last_detections:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"{category} ({score:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)