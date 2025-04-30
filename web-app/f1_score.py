import pandas as pd
import numpy as np
import os

# Folders
ground_truth_folder = "ground_truth"
prediction_folder = "YOLO_predictions"  # Updated to YOLO predictions

# Confidence thresholds
conf_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Initialize results
results = []

# Video paths
video_paths = ['./media/light_near.mp4', "./media/light_far.mp4", 
    "./media/dark_near.mp4", "./media/dark_far.mp4"] 

for video_path in video_paths:

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Ground truth path
    gt_path = f"{ground_truth_folder}/{video_name}_ground.csv"
    print(f"Loading ground truth: {gt_path}")
    if not os.path.exists(gt_path):
        print(f"Ground truth for {video_name} not found.")
        continue
    
    try:
        gt_df = pd.read_csv(gt_path)
    except Exception as e:
        print(f"Error loading {gt_path}: {e}")
        continue

    print(f"Processing ground truth file: {gt_path}")
    print(f"Columns in {gt_path}: {gt_df.columns.tolist()}")
    print(f"First few rows of {gt_path}:\n{gt_df.head()}")

    gt_df.columns = gt_df.columns.str.strip().str.lower()

    frame_col = next((col for col in gt_df.columns if "frame" in col.lower()), None)
    if not frame_col:
        print(f"Error: No 'Frame Number' column found in {gt_path}. Columns: {gt_df.columns.tolist()}")
        continue

    try:
        gt_df = gt_df.set_index(frame_col)
    except KeyError as e:
        print(f"Error setting index on {gt_path}: {e}")
        print(f"Available columns: {gt_df.columns.tolist()}")
        continue

    for conf in conf_thresholds:
        video_results_df = pd.DataFrame([r for r in results if r['video'] == video_name])
        video_csv_path = f"yolo/results/{video_name}_results.csv"
        os.makedirs("yolo/results", exist_ok=True)
        video_results_df.to_csv(video_csv_path, index=False)
        print(f"Saved results for {video_name} to '{video_csv_path}'.")
        confidence = int(conf * 10)
        pred_path = f"{prediction_folder}/{video_name}/prediction_conf_{confidence}.csv"
        print(f"Loading prediction: {pred_path}")
        if not os.path.exists(pred_path):
            print(f"Prediction for {video_name}, conf={conf:.1f} not found.")
            continue

        try:
            pred_df = pd.read_csv(pred_path)
        except Exception as e:
            print(f"Error loading {pred_path}: {e}")
            continue

        print(f"Processing prediction file: {pred_path}")
        print(f"Columns in {pred_path}: {pred_df.columns.tolist()}")
        print(f"First few rows of {pred_path}:\n{pred_df.head()}")

        pred_df.columns = pred_df.columns.str.strip().str.lower()

        pred_frame_col = next((col for col in pred_df.columns if "frame" in col.lower()), None)
        if not pred_frame_col:
            print(f"Error: No 'Frame Number' column found in {pred_path}. Columns: {pred_df.columns.tolist()}")
            continue

        try:
            pred_df = pred_df.set_index(pred_frame_col)
        except KeyError as e:
            print(f"Error setting index on {pred_path}: {e}")
            print(f"Available columns: {pred_df.columns.tolist()}")
            continue

        meta_cols = ["timestamp", "frame number", "frame"]
        objects = sorted(set(gt_df.columns) - set(meta_cols) | set(pred_df.columns) - set(meta_cols))

        for obj in objects:
            if obj not in pred_df.columns:
                pred_df[obj] = 0
            if obj not in gt_df.columns:
                gt_df[obj] = 0

        all_frames = sorted(set(pred_df.index))
        aligned_pred = pd.DataFrame(0, index=all_frames, columns=objects)
        aligned_gt = pd.DataFrame(0, index=all_frames, columns=objects)

        for frame in all_frames:
            if frame in pred_df.index:
                aligned_pred.loc[frame, objects] = pred_df.loc[frame, objects]
            if frame in gt_df.index:
                aligned_gt.loc[frame, objects] = gt_df.loc[frame, objects]
            else:
                print(f"Warning: Frame {frame} missing in ground truth for {video_name}")

        tp = ((aligned_pred == 1) & (aligned_gt == 1)).sum()
        fp = ((aligned_pred == 1) & (aligned_gt == 0)).sum()
        fn = ((aligned_pred == 0) & (aligned_gt == 1)).sum()

        total_tp = tp.sum()
        total_fp = fp.sum()
        total_fn = fn.sum()
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            "video": video_name,
            "conf_threshold": conf,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn
        })

        print(f"\nVideo {video_name}, Conf={conf:.1f}:")
        for obj in objects:
            precision_obj = tp[obj] / (tp[obj] + fp[obj]) if (tp[obj] + fp[obj]) > 0 else 0
            recall_obj = tp[obj] / (tp[obj] + fn[obj]) if (tp[obj] + fn[obj]) > 0 else 0
            f1_obj = 2 * (precision_obj * recall_obj) / (precision_obj + recall_obj) if (precision_obj + recall_obj) > 0 else 0
            print(f"Object: {obj}")
            print(f"  TP: {tp[obj]}, FP: {fp[obj]}, FN: {fn[obj]}")
            print(f"  Precision: {precision_obj:.4f}, Recall: {recall_obj:.4f}, F1: {f1_obj:.4f}")

# Aggregate results
results_df = pd.DataFrame(results)
print("\nAggregate Results:")
for conf in conf_thresholds:
    conf_results = results_df[results_df["conf_threshold"] == conf]
    if not conf_results.empty:
        avg_f1 = conf_results["f1"].mean()
        avg_precision = conf_results["precision"].mean()
        avg_recall = conf_results["recall"].mean()
        print(f"Conf={conf:.1f}: Avg Precision={avg_precision:.4f}, Avg Recall={avg_recall:.4f}, Avg F1={avg_f1:.4f}")
    else:
        print(f"Conf={conf:.1f}: No results available")

# Save results
results_df.to_csv("yolo_f1_results.csv", index=False)
print("F1 results saved to 'yolo_f1_results.csv'.")