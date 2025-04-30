import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define your videos and CSV names (match them properly)
video_paths = ['./media/light_near.mp4', './media/light_far.mp4', './media/dark_near.mp4', './media/dark_far.mp4']

# Get just the video base names to use in filenames
video_names = [os.path.splitext(os.path.basename(v))[0] for v in video_paths]

sns.set(style="whitegrid")
metrics = ['precision', 'recall', 'f1']

# Iterate over each video's results file
for video in video_names:
    csv_path = f"mediapipe_p/results/f1_results_{video}.csv"
    
    if not os.path.exists(csv_path):
        print(f"Warning: CSV not found for {video}, skipping...")
        continue

    df = pd.read_csv(csv_path)

    # Convert confidence thresholds to strings for clearer x-axis
    df["conf_threshold"] = df["conf_threshold"].astype(str)

    # Plot
    plt.figure(figsize=(10, 6))
    for metric in metrics:
        plt.plot(df["conf_threshold"], df[metric], marker='o', label=metric.capitalize())

    plt.title(f"Metrics vs Confidence Threshold for '{video}'")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()