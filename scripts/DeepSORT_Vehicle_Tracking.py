import sys
import os
import cv2
from ultralytics import YOLO

from deep_sort_realtime.deepsort_tracker import DeepSort




# Initialize Model, Tracker, Output Video and Video Capture

video_path = os.path.join('.', 'data/base_video.mp4')
model_path = os.path.join('.', 'runs/detect/train3/weights', 'best.pt')
output_video_name = 'final_video'
output_video_path = os.path.join('.', 'output')
fps=10

model = YOLO(model_path)
capture = cv2.VideoCapture(video_path)

# max_age: Max bound track can go before updated; nn_budget: set resource bound on embedder
tracker = DeepSort(max_age=30, nn_budget=100)

b, frame = capture.read()

height, width, layers = frame.shape
video = cv2.VideoWriter(os.path.join(output_video_path, output_video_name + '.mp4'), 
                        cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# For each frame, process, draw box, and output to video
while b:
    if frame is None:
        print('Frame Could Not Be Loaded')
        continue
    frame = frame.copy()

    # Use Model to Predict Objects
    detections = []
    results = model.predict(source=frame, conf=0.5, verbose=False)
    for result in results:
        
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            detections.append([[x1, y1, x2 - x1, y2 - y1], score, int(class_id)])

    # Update Tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw Tracked Bounding Boxes
    for track in tracks:        
        # If track hasn't been updated recently, we do not care about it for this frame
        if not track.is_confirmed() or track.time_since_update > 1:
                continue        
        
        # Get Information From Track and Bounding Box Coordinates
        id = track.track_id
        xmin, ymin, xmax, ymax = track.to_tlbr()

        # Draw Bounding Box
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
        cv2.putText(frame, f"ID: {id}", (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
    
    # cv2.imshow('image', frame)
    # cv2.waitKey(30)
    video.write(frame)
    b, frame = capture.read()


capture.release()
video.release()
cv2.destroyAllWindows()
