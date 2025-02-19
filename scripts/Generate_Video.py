import os
import cv2
import argparse

def main(fps, video_images_path, output_name):
    # Create List of Images
    images = [filename for filename in os.listdir(video_images_path) 
            if filename.endswith(('.jpg', '.jpeg', '.png'))]

    images.sort()

    # Get Image Dimensions and Shapes For Video Dimensions
    frame = cv2.imread(os.path.join(video_images_path, images[0]))
    height, width, layers = frame.shape

    # Create Video and Add Image Frames to Video
    video = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(video_images_path, image)))

    # Publish Video and Shut Down CV
    video.release()
    cv2.destroyAllWindows()
    print("Video generated successfully!")

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
  parser.add_argument("--fps", required=False, default=10,
    help="Frames Per Second of Output Video")
  parser.add_argument("-i", "--input_images", required=True, 
    type=validate_file, metavar="FILE",
    help="Input Video Path")
  parser.add_argument("-o", "--output_name", required=False, type=validate_mp4, 
    default='base2_video.mp4',
    help="Output Video Name in MP4 Format")
  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  main(int(args.fps), args.input_images, args.output_name)