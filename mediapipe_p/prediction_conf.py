import mediapipe as mp
import cv2
import time
import csv
import os

# Model setup
BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

video_files = [
    "./media/light_far.mp4", 
    "./media/dark_near.mp4",  
    "./media/dark_far.mp4",  
    "./media/light_near.mp4"
    ] 
conf_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for video_path in video_files:
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    for conf in conf_thresholds:
        print(f"\nProcessing {video_name} with conf={conf}")

        options = ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path='./models/efficientdet_lite2.tflite'),
            score_threshold=conf,
            running_mode=VisionRunningMode.VIDEO
        )

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        detection_interval = int(fps / 2)

        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            continue

        frame_index = 0
        unique_objects = set()
        detection_matrix = []

        with ObjectDetector.create_from_options(options) as detector:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_index += 1

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp = (frame_index / fps) * 1000  # ms
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                if frame_index % detection_interval == 0:
                    result = detector.detect_for_video(mp_image, int(timestamp))

                    detected_labels = {}

                    for detection in result.detections:
                        category = detection.categories[0].category_name
                        score = detection.categories[0].score
                        if score >= conf:
                            unique_objects.add(category)
                            detected_labels[category] = 1  # Mark as detected

                    # Fill in row data
                    row = {
                        "timestamp": round(timestamp, 1),
                        "frame": frame_index
                    }
                    for obj in unique_objects:
                        row[obj] = detected_labels.get(obj, 0)
                    detection_matrix.append(row)

        cap.release()

        # Save to CSV after processing
        try:
            os.makedirs("video", exist_ok=True)
            unique_objects = sorted(unique_objects)
            csv_path = f"video/pred_conf_{conf:.1f}_{video_name}.csv"
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                header = ["Timestamp", "Frame Number"] + unique_objects
                writer.writerow(header)
                for row in detection_matrix:
                    csv_row = [row["timestamp"], row["frame"]] + [row.get(obj, 0) for obj in unique_objects]
                    writer.writerow(csv_row)
            print(f"Saved CSV: {csv_path}")
        except Exception as e:
            print(f"Error writing CSV for {video_name} with conf {conf}: {e}")