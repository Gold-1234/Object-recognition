#detecting t,tree and warrior II pose using webcam
import math
import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle

def classifyPose(landmarks, output_image):
    label = 'Unknown Pose'
    safety_label = 'Needs Adjustment'
    color = (0, 0, 255)

    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                           landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

    # Warrior II Pose
    if (165 < left_elbow_angle < 195 and 165 < right_elbow_angle < 195):
        if (80 < left_shoulder_angle < 110 and 80 < right_shoulder_angle < 110):
            if ((165 < left_knee_angle < 195 or 165 < right_knee_angle < 195) or
                (90 < right_knee_angle < 120 or 90 < left_knee_angle < 120)):
                label = 'warrior II pose'
                safety_label = 'Safe Pose'

    # T Pose
    if (160 < left_knee_angle < 195 and 160 < right_knee_angle < 195):
        if (80 < left_shoulder_angle < 110 and 80 < right_shoulder_angle < 110):
            label = 'T pose'
            safety_label = 'Safe Pose'
            
    # Tree Pose
    if (165 < left_knee_angle < 195 or 165 < right_knee_angle < 195):
        left_foot_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value][1]
        right_foot_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value][1]
        left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value][1]
        right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value][1]

        if (left_foot_y < right_knee_y) or (right_foot_y < left_knee_y):
            label = 'tree pose'
            safety_label = 'Unsafe Pose'

    # Hands on Hips Pose
    left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value][1]
    right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value][1]
    left_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][1]
    right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][1]

    if (45 < left_elbow_angle < 100 and 45 < right_elbow_angle < 100):
        if (abs(left_wrist_y - left_hip_y) < 60 and abs(right_wrist_y - right_hip_y) < 60):
            label = 'hands on hips'
            safety_label = 'Safe Pose'

    if label != 'Unknown Pose':
        color = (0, 255, 0)

    cv2.putText(output_image, f'{label} ({safety_label})', (10, 30),
                cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

    return output_image, label

def detectPose(frame, pose):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = [(lmk.x * frame.shape[1], lmk.y * frame.shape[0], lmk.z * frame.shape[1])
                     for lmk in results.pose_landmarks.landmark]
        return frame, landmarks
    return frame, None

#it runs whole process
def main():
    pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
    camera_video = cv2.VideoCapture(0)
    camera_video.set(3, 1280)
    camera_video.set(4, 960)

    # Buffer for pose classification (store the last 3 poses)
    pose_buffer = []

    cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

    while camera_video.isOpened():
        ok, frame = camera_video.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

        frame, landmarks = detectPose(frame, pose_video)
        if landmarks:
            frame, label = classifyPose(landmarks, frame)

            # Add label to the buffer
            pose_buffer.append(label)

            # Keep the buffer size to 3
            if len(pose_buffer) > 3:
                pose_buffer.pop(0)

            if len(set(pose_buffer)) == 1:
                final_label = pose_buffer[0]
            else:
                final_label = 'Unknown Pose'

            cv2.putText(frame, final_label, (10, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        cv2.imshow('Pose Classification', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    camera_video.release()
    cv2.destroyAllWindows()
    
def classify_from_frame(frame):
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
    frame, landmarks = detectPose(frame, pose)
    if landmarks:
        frame, label = classifyPose(landmarks, frame)
        return frame, label
    return frame, "Unknown Pose"

if __name__ == "__main__":
    main()
