import os
from models.download import download_yolo_model
from yolo.yolo_prediction_csv import process_videos_yolo
from yolo.compare import evaluate_yolo_predictions

target_objects = ['bench', 'plant', 'bottle', 'watch', 'phone', 'notebook', 'bag', 'shoes', 'chair']

# to add light far
video_name = ['light_near', 'dark_far', 'dark_near', 'light_far']

video_paths = []
for name in video_name:
    path = f'./media/{name}.mp4'
    if not os.path.isfile(path):
        print(f"Warning: Video '{path}' not found.")
        continue
    video_paths.append(path)
	
		
model_path = download_yolo_model(
    file_id='19BhXvQLiF12sMs28ElDBjFnckmiqlG-C',
    model_name='yolo11n.pt',
)
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Failed to download YOLO model: {model_path}")

process_videos_yolo(
	video_paths,
    target_objects,
    model_path,
)

evaluate_yolo_predictions(
	video_paths,
    target_objects,
)


