import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define your videos and CSV names
video_paths = ['./media/light_near.mp4', './media/light_far.mp4', './media/dark_near.mp4', './media/dark_far.mp4']
video_names = [os.path.splitext(os.path.basename(v))[0] for v in video_paths]

sns.set(style="whitegrid")

# Collect CSV file paths
csv_files = []
for video in video_names:
    csv_path = f"mediapipe_p/results/f1_results_{video}.csv"
    if os.path.exists(csv_path):
        csv_files.append(csv_path)
    else:
        print(f"Warning: CSV not found for {video}, skipping...")

# Read and combine all CSVs
all_dfs = []
for file in csv_files:
    df = pd.read_csv(file)
    all_dfs.append(df)

combined_df = pd.concat(all_dfs, ignore_index=True)

# Ensure 'conf_threshold' is numeric
combined_df['conf_threshold'] = combined_df['conf_threshold'].astype(float)

# Group by video and confidence threshold to average F1
grouped = combined_df.groupby(['video', 'conf_threshold']).agg({'f1': 'mean'}).reset_index()

# Plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=grouped, x='conf_threshold', y='f1', hue='video', marker='o')

plt.title("F1 Score vs Confidence Threshold")
plt.xlabel("Confidence Threshold")
plt.ylabel("F1 Score")
plt.grid(True)
plt.legend(title='Video')
plt.tight_layout()
plt.savefig("f1_combined_plot.png")
plt.show()