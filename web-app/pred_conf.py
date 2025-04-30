import cv2
import time
import csv
import os
from ultralytics import YOLO

# Create predictions folder
os.makedirs("predictions", exist_ok=True)

# Load the YOLOv8 model
model = YOLO('./models/yolo11n.pt')  # or your trained model path

# List of videos (placeholder paths; replace with actual videos)
videos = [
    "./media/light_near.mp4",  
    "./media/light_far.mp4",  
    "./media/dark_near.mp4", 
    "./media/dark_far.mp4"
]

# Confidence thresholds
conf_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for video_path in videos:
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    for conf in conf_thresholds:
        print(f"\nProcessing {video_name} with conf={conf}")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        detection_interval = int(fps / 2)

        frame_index = 0
        unique_objects = set()
        detection_matrix = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1

            if frame_index % detection_interval == 0:
                results = model.predict(frame, conf=conf, verbose=False)
                row = {
                    "timestamp": round((frame_index / fps) * 1000, 1),
                    "frame": frame_index
                }

                detected_labels = {}
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0].item())
                        category = r.names[cls_id]
                        unique_objects.add(category)
                        detected_labels[category] = 1
                for obj in unique_objects:
                    row[obj] = detected_labels.get(obj, 0)
                detection_matrix.append(row)

        cap.release()

        # Save CSV
        # Save CSV
csv_path = f"Yolo/{video_name}/pred_conf_{conf:.1f}_{video_name}.csv"
os.makedirs(os.path.dirname(csv_path), exist_ok=True)  # ✅ Create directory if it doesn't exist

unique_objects = sorted(unique_objects)
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    header = ["Timestamp", "Frame Number"] + unique_objects
    writer.writerow(header)
    for row in detection_matrix:
        csv_row = [row["timestamp"], row["frame"]] + [row.get(obj, 0) for obj in unique_objects]
        writer.writerow(csv_row)

print(f"Saved CSV: {csv_path}")

print("Prediction folder 'predictions' with 6 subfolders and 9 CSVs each generated.")