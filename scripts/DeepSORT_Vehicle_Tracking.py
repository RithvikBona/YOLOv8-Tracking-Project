import sys
import os
import cv2
import argparse
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def main(live, fps, input_video, model_path, output_video_name):
    # Initialize Model, Tracker, Output Video and Video Capture
    output_video_path = os.path.join('.', 'output')
    if not os.path.exists(output_video_path):
            os.makedirs(output_video_path)

    model = YOLO(model_path)
    capture = cv2.VideoCapture(input_video)

    # max_age: Max bound track can go before updated; nn_budget: set resource bound on embedder
    tracker = DeepSort(max_age=30, nn_budget=100)

    b, frame = capture.read()
    height, width, layers = frame.shape
    video = cv2.VideoWriter(os.path.join(output_video_path, output_video_name), 
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
        
        if live:
            cv2.imshow('image', frame)
            cv2.waitKey(30)

        video.write(frame)
        b, frame = capture.read()


    capture.release()
    video.release()
    cv2.destroyAllWindows()


def validate_file(f):
    if not os.path.exists(f):
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

def validate_mp4(f):
    if not f.endswith('.mp4'):
        raise argparse.ArgumentTypeError("{0} does not end with .mp4".format(f))
    return f

def parse_args():
  parser = argparse.ArgumentParser(description="DeepSORT Script")
  parser.add_argument("-l", "--live", action='store_true',
    help="Shows a Live Feed of Video With Annotations and Tracking")
  parser.add_argument("--fps", required=False, default=10,
    help="Frames Per Second of Output Video")
  parser.add_argument("-i", "--input_video", required=True, 
    type=validate_file, metavar="FILE",
    help="Input Video Path")
  parser.add_argument("-m", "--model", required=True, 
    type=validate_file, metavar="FILE",
    help="Model Path")
  parser.add_argument("-o", "--output_name", required=False, type=validate_mp4, 
    default='final_video.mp4',
    help="Output Video Name in MP4 Format")
  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  main(args.live, int(args.fps), args.input_video, args.model, args.output_name)