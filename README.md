
# Initial Experimental Setup

## For Object Recognition
1. Record **1-minute** video(s) in _controlled settings_ using strictly the **same phone** => Fix one of the orientations - **Portrait** / Landscape.
2. Controlled Settings => You define the **objects**, the **scale of the objects**, the color, brightness, and all the lighting conditions in which the video(s) are shot.
3. The holistic detection model must be fine-tuned for individual parameters => You have to iterate over the detection script using a for loop with different values of each relevant parameter such as _threshold_score_.
4. The target for which we are fine-tuning the model is to achieve higher performance metrics (accuracy, precision, recall, f1-score) => True Positive, True Negative, False Positive, and False Negative.
5. Record True Labels on the recorded video. If your video has been recorded at 30 FPS, take 2 representative frames per second (1 representative frame per 15 frames).
6. Manual Annotation - Create a CSV / Excel file, which has the following columns - Timestamp of the video for the Representative Frame (uptil milliseconds), Object_1(Cup), Object_2(Pen),...... For each Object column, you will have a Binary outcome as the value. 1 implies that the object is present in the frame and 0 implies that the object is absent from the frame.

# Instructions to setup the project

## Instructions for Part 1 - Object Recognition

This project tests object recognition using two different models: **YOLO** and **MediaPipe**.  
Each model has its own setup and running instructions as outlined below.

First, install the required libraries by running:
```bash
pip install -r requirements.txt
```
```bash
python3 integrated.py
```

# System Architecture
![UML class (6)](https://github.com/user-attachments/assets/a0c90b72-fd23-4cd9-8081-3300dd010d8e)

# Working model



![WhatsApp Image 2025-04-30 at 22 14 44](https://github.com/user-attachments/assets/1f5d5534-ad14-4589-9454-a088cfb9f9d5)
![WhatsApp Image 2025-04-30 at 22 16 41](https://github.com/user-attachments/assets/0444c3b6-c6c6-452a-b36d-de46ab41b93f)
![WhatsApp Image 2025-04-30 at 22 21 50](https://github.com/user-attachments/assets/ddd26510-c037-4854-b555-b330acfabd6f)
![WhatsApp Image 2025-04-30 at 22 17 03](https://github.com/user-attachments/assets/4813fd7f-9538-46bc-afa0-031620baf90b)

![WhatsApp Image 2025-04-30 at 22 53 37 (1)](https://github.com/user-attachments/assets/0b8e7c17-f14f-4c05-9b40-3299ba3cf29c)
![WhatsApp Image 2025-04-30 at 22 53 24 (1)](https://github.com/user-attachments/assets/b2cf4281-b5c5-4e57-a2a1-2513ee9db7ef)
