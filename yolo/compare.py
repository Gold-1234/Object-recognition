import cv2
from ultralytics import YOLO
import time
import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_yolo_predictions(
    video_paths,
    target_objects,
    ground_truth_dir='./ground_truth',
    prediction_base_dir='./yolo/predictions',
    output_base_dir='./yolo/compared_ground',
    conf_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
):

    for video in video_paths:
        
        if not video or not os.path.isfile(video):
            print(f"Skipping invalid or missing video: {video}")
            continue

        video = os.path.splitext(os.path.basename(video))[0]

        timestamp_df = pd.read_csv(f'{ground_truth_dir}/{video}_ground.csv')
        timestamp_df['Rounded Timestamp (ms)'] = (timestamp_df['Timestamp (ms)'] / 500).round() * 500

        metrics_summary = []

        output_dir = os.path.join(output_base_dir, video)
        os.makedirs(output_dir, exist_ok=True)

    # Process each confidence threshold
        for conf_score in conf_thresholds:

            yolo_filename = os.path.join(
                    prediction_base_dir,
                    f'yolo_predictions_{video}',
                    f'yolo_predictions_conf_{conf_score}.csv'
                )

            try:
                yolo_df = pd.read_csv(yolo_filename)
            except FileNotFoundError:
                print(f"Error: {yolo_filename} not found.")
                continue

            # Round timestamps in YOLO predictions
            yolo_df['Rounded Timestamp (ms)'] = (yolo_df['Timestamp (ms)'] / 500).round() * 500 

            # Merge with ground truth on rounded timestamps
            matched_df = pd.merge(
                yolo_df,
                timestamp_df,
                how='inner',
                left_on='Rounded Timestamp (ms)',
                right_on='Rounded Timestamp (ms)',
                suffixes=('_pred', '_true')
            )

            matched_df['Confidence'] = conf_score 

            for obj in target_objects:
                matched_df[f'{obj}_pred'] = matched_df.get(f'{obj}_pred', 0)
                matched_df[f'{obj}_true'] = matched_df.get(f'{obj}_true', 0)

            # Select output columns
            output_columns = [
                'Timestamp (ms)_pred', 'Rounded Timestamp (ms)', 'Confidence', 'Inference Time (ms)'
            ] + [f'{obj}_pred' for obj in target_objects] + [f'{obj}_true' for obj in target_objects]
            matched_df = matched_df[output_columns]

            # Save matched CSV
            matched_filename = os.path.join(output_dir, f'matched_yolo_conf_{conf_score}.csv')
            matched_df.to_csv(matched_filename, index=False)
            print(f"Saved matched data for conf {conf_score:.1f} to '{matched_filename}'")

            # Compute metrics for each object
            metrics_row = {'Confidence': conf_score}
            for obj in target_objects:
                y_true = matched_df[f'{obj}_true']
                y_pred = matched_df[f'{obj}_pred']
                
                # Compute metrics
                metrics_row[f'{obj}_accuracy'] = accuracy_score(y_true, y_pred)
                metrics_row[f'{obj}_precision'] = precision_score(y_true, y_pred, zero_division=0)
                metrics_row[f'{obj}_recall'] = recall_score(y_true, y_pred, zero_division=0)
                metrics_row[f'{obj}_f1'] = f1_score(y_true, y_pred, zero_division=0)

            # Compute average metrics across objects
            metrics_row['avg_accuracy'] = np.mean([metrics_row[f'{obj}_accuracy'] for obj in target_objects])
            metrics_row['avg_precision'] = np.mean([metrics_row[f'{obj}_precision'] for obj in target_objects])
            metrics_row['avg_recall'] = np.mean([metrics_row[f'{obj}_recall'] for obj in target_objects])
            metrics_row['avg_f1'] = np.mean([metrics_row[f'{obj}_f1'] for obj in target_objects])

            # Add average inference time
            metrics_row['avg_inference_time_ms'] = matched_df['Inference Time (ms)'].mean()

            metrics_summary.append(metrics_row)

        if metrics_summary:
                metrics_df = pd.DataFrame(metrics_summary)
                metrics_filename = os.path.join(output_base_dir, f'metrics_summary_{video}.csv')
                try:
                    metrics_df.to_csv(metrics_filename, index=False)
                    print(f"Saved metrics summary for video '{video}' to '{metrics_filename}'")
                except Exception as e:
                    print(f"Error saving metrics summary for video '{video}': {e}")
        else:
            print(f"No metrics computed for video '{video}'.")