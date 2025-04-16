# To-do

1. Add instructions to this README.md file to setup the project and run it.
2. Multimedia files such as _.mp4_ files must **NOT** be kept in the Github repository. They must always be referenced through an external source such as an External Google Drive Link to each file.
3. Add a **.gitignore** file to exclude bulky files such as **.mp4** to get automatically excluded from the Git repository when the commit is pushed to the remote Github repository.
4. Rename the repository to small case "object-recognition".
5. Remove temporary files such as **.DS_Store** from the repository by placing their name in the **.gitignore** file.
6. Add **mediapipe_env** to _.gitignore_.


# Initial Experimental Setup
1. Record **1-minute** video(s) in _controlled settings_ using strictly the **same phone**.
2. Controlled Settings => You define the **objects**, the **scale of the objects**, the color, brightness, and all the lighting conditions in which the video(s) are shot.
3. The holistic detection model must be fine-tuned for individual parameters => You have to iterate over the detection script using a for loop with different values of each relevant parameter such as _threshold_score_.
4. The target for which we are fine-tuning the model is to achieve higher performance metrics (accuracy, precision, recall, f1-score) => True Positive, True Negative, False Positive, and False Negative.
5. Record True Labels on the recorded video. If your video has been recorded at 30 FPS, take 2 representative frames per second (1 representative frame per 15 frames).
6. Manual Annotation - Create a CSV / Excel file, which has the following columns - Timestamp of the video for the Representative Frame (uptil milliseconds), Object_1(Cup), Object_2(Pen),...... For each Object column, you will have a Binary outcome as the value. 1 implies that the object is present in the frame and 0 implies that the object is absent from the frame.

# Instructions to setup the project

## Instructions for Part 1 - Object Recognition


## Instructions for Part 2 - Holistic Recognition using MediaPipe


