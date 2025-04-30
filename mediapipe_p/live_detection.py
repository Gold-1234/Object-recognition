import mediapipe as mp
import cv2
import time

# MediaPipe task imports
BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def main():
    video_source = 0  # Webcam (default camera)
    model_path = './models/efficientdet_lite2.tflite'
    score_threshold = 0.4
    
    try:
        # Initialize video capture (webcam)
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 30  
        detection_interval = int(fps / 2)  
        
        last_detections = []
        last_inference_time = 0
        
        def process_result(result, output_image, timestamp_ms): #async function
            nonlocal last_detections, last_inference_time 		#non-local used to declare non local, while writing
            last_detections = []
            
            for detection in result.detections:
                bbox = detection.bounding_box
                x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
                category = detection.categories[0].category_name
                score = detection.categories[0].score
                last_detections.append((x, y, w, h, category, score))
                
            if last_detections:
                x, y, w, h, category, score = last_detections[-1]
                print(f"Detected: {category} | Score: {score:.2f} | time: {last_inference_time:.2f} ms")
                print(f"Bounding Box: x={x}, y={y}, w={w}, h={h}")
        
        # Initialize detector
        options = ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            score_threshold=score_threshold,
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=process_result
        )
        detector = ObjectDetector.create_from_options(options)
        
        frame_index = 0
        start_time = time.time() * 1000  # Reference time in ms
        
        # Process frames with detector context
        with detector:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_index += 1
                
                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(time.time() * 1000 - start_time)  # Relative timestamp
                
                # Trigger detection if frame index matches interval
                if frame_index % detection_interval == 0:
                    detector.detect_async(mp_image, timestamp_ms)
                
                # Draw detections
                for x, y, w, h, category, score in last_detections:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    text_position = (max(0, x), max(20, y - 10))
                    cv2.putText(
                        frame,
                        f"{category} ({score:.2f})",
                        text_position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2
                    )
                
                # Display frame
                cv2.imshow("Object Detection", frame)
                
                # Break on 'q'
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()