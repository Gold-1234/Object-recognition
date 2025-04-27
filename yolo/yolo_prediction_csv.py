import cv2
from ultralytics import YOLO
import time
import pandas as pd
import os

def process_videos_yolo(
    video_paths,
    target_objects,
    model_path,
    conf_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    output_base_dir='./yolo/predictions',
    frame_interval=None
):
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading YOLO model '{model_path}': {e}")
        return

    # label_mapping = {
    # 'handbag': 'bag',
    # 'backpack': 'bag',
    # 'book': 'notebook',
    # 'cell phone': 'phone'
    # }
    
    for video_path in video_paths:
        if not video_path or not os.path.isfile(video_path):
            print(f"Skipping invalid or missing video: {video_path}")
            continue
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video '{video_path}'.")
            continue

        output_dir = os.path.join(output_base_dir, f'yolo_predictions_{video_name}')
        os.makedirs(output_dir, exist_ok=True)

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps/2) if frame_interval is None else frame_interval

        frame_index = 0
        all_predictions = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1
            if frame_index % frame_interval == 0:
                timestamp_ms = int(round(cap.get(cv2.CAP_PROP_POS_MSEC)))
                row = {'Timestamp (ms)': timestamp_ms}
                for obj in target_objects:
                    row[obj] = 0
                start_time = time.perf_counter()
                results = model(frame)
                end_time = time.perf_counter()
                row['Inference Time (ms)'] = (end_time - start_time) * 1000
                row['Detections'] = [(box.conf[0].item(), model.names[int(box.cls[0])].lower()) 
                                   for result in results for box in result.boxes]
                all_predictions.append(row)

        cap.release()

        # Generate CSVs for each confidence threshold
        for conf_score in conf_thresholds:
            prediction_log = []
            for row in all_predictions:
                pred_row = row.copy()
                del pred_row['Detections']
                for obj in target_objects:
                    pred_row[obj] = 0
                for conf, label in row['Detections']:
                    if conf >= conf_score and label in target_objects:
                        pred_row[label] = 1
                prediction_log.append(pred_row)
            
            df = pd.DataFrame(prediction_log)
            filename = os.path.join(output_dir, f'yolo_predictions_conf_{conf_score}.csv')
            try:
                df.to_csv(filename, index=False)
                print(f"Saved predictions for video '{video_name}', conf {conf_score} to '{filename}'")
            except Exception as e:
                print(f"Error saving CSV for video '{video_name}', conf {conf_score}: {e}")
       
print('All videos processed')