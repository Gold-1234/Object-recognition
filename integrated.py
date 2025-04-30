from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import time
from text_to_speech.main import speak
import threading
from poseDetection.pose import classify_from_frame

app = Flask(__name__)

# Add shared detection state
detection_running = False
detection_lock = threading.Lock()
# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Model config
model_path = './models/efficientdet_lite2.tflite'
score_threshold = 0.4

# Globals
last_detections = []
last_spoken = None
last_spoken_time = 0
cooldown_seconds = 3
pose_label = 'Unknown'

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

        if category == "person":
            now = time.time()
            if category != last_spoken or (now - last_spoken_time) > cooldown_seconds:
                speak(f"Detected {category} with score {score:.2f}")
                last_spoken = category
                last_spoken_time = now

def generate_frames():
    global detection_running

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = initialize_detector()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    detection_interval = int(fps / 2)
    frame_index = 0
    start_time = time.time() * 1000

    with detector:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_index += 1

            # Only process if running
            with detection_lock:
                if detection_running and frame_index % detection_interval == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    timestamp_ms = int(time.time() * 1000 - start_time)
                    detector.detect_async(mp_image, timestamp_ms)

            # Draw boxes
            for x, y, w, h, category, score in last_detections:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"{category} ({score:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                print(category)
                if category == 'person':
                        frame, pose_label = classify_from_frame(frame)
                        if pose_label:
                              cv2.putText(frame, f"Pose: {pose_label}", (10, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)



            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    global detection_running
    with detection_lock:
        detection_running = not detection_running
    return jsonify({'running': detection_running})

@app.route('/get_detections')
def get_detections():
    return jsonify({'objects': [d[4] for d in last_detections]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)