import sys
import os
import cv2
from ultralytics import YOLO

# Importing DeepSORT
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.detection import Detection



# Initialize Model, Tracker, Output Video and Video Capture

video_path = os.path.join('.', 'base_video.mp4')
model_path = os.path.join('.', 'runs/detect/train3/weights', 'best.pt')
output_video_name = 'final_video'
fps=2

model = YOLO(model_path)
capture = cv2.VideoCapture(video_path)

# max_age: Maximum number of frames a track can go without being updated
tracker = Tracker(metric=NearestNeighborDistanceMetric('cosine', 0.4) ,max_age=30)

b, frame = capture.read()

height, width, layers = frame.shape
video = cv2.VideoWriter(output_video_name + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))


#### detection threshhold
# For each frame, process, draw box, and output to video
while b:
    if frame is None:
        print('Frame Could Not Be Loaded')
        continue

    results = model.predict(source=frame, conf=0.5, verbose=False)
    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            detections.append(Detection([x1, y1, x2 - x1, y2 - y1], score, None))
    
    
    # Update Tracker
    tracker.update(detections)

    # Draw Tracked Bounding Boxes
    for track in tracker.tracks:
        
        # If track hasn't been updated recently, we do not care about it for this frame
        if not track.is_confirmed() or track.time_since_update > 1:
                continue
        
        # Get Information From Track and Bounding Box Coordinates
        id = track.track_id
        xmin, ymin, xmax, ymax = track.to_tlbr()

        # Draw Bounding Box
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
        cv2.putText(frame, f"ID: {id}", (int((xmin + xmax) / 2), int(ymax) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)



    # Change!!
    video.write(frame)
    b, frame = capture.read()

capture.release()
video.release()
cv2.destroyAllWindows()