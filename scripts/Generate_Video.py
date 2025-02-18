import os
import cv2

## Edit to Make More Command Line Friendly / Not Hardcoded

video_images_path = 'data/video_images'
output_video_name = 'base_video'
fps=10


# Create List of Images
images = [filename for filename in os.listdir(video_images_path) 
          if filename.endswith(('.jpg', '.jpeg', '.png'))]

images.sort()

# Get Image Dimensions and Shapes For Video Dimensions
frame = cv2.imread(os.path.join(video_images_path, images[0]))
height, width, layers = frame.shape

# Create Video and Add Image Frames to Video
video = cv2.VideoWriter(output_video_name + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(video_images_path, image)))

# Publish Video and Shut Down CV
video.release()
cv2.destroyAllWindows()
print("Video generated successfully!")