import os
import gdown

def download_yolo_model(file_id: str, model_name: str, model_dir: str = "./models"):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)
    if not os.path.exists(model_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {model_name} from Google Drive...")
        gdown.download(url, model_path, quiet=False)
    else:
        print(f"{model_name} already exists at {model_path}.")

    return model_path
