import os
from PIL import Image 
import argparse

def main(data_path):
    ## Define folder paths
    train_image_path = os.path.join(data_path, 'train/images')
    train_label_path = os.path.join(data_path, 'train/labels')
    val_image_path = os.path.join(data_path, 'val/images')
    val_label_path = os.path.join(data_path, 'val/labels')

    ## Do relabelling for both train and val
    for (curr_image_path, curr_label_path) in [(train_image_path, train_label_path), 
                                            (val_image_path, val_label_path)]:
        label_list = [filename for filename in os.listdir(curr_label_path)]
        for label in label_list:

            # Get image dimensions for normalization
            image_filename = label[:-3] + 'png'
            image_path = os.path.join(curr_image_path, image_filename)
            img = Image.open(image_path)
            full_width = img.width 
            full_height = img.height 

            overwrite_content = ''
            label_path = os.path.join(curr_label_path, label)
            with open(label_path, 'r') as file:
            # Read each line in the file
                for line in file:
                    tokens = line.strip().split()

                    if not tokens:
                        continue
                    
                    # First get object class, check if relevant, and assign class_id
                    class_name = tokens[0]
                    if class_name == 'Car':
                        class_id = 0
                    elif class_name == 'Pedestrian':
                        class_id = 1
                    elif class_name == 'Cyclist':
                        class_id = 2
                    else:
                        # If irrelevant, skip to next object
                        continue
                    
                    # Get object information for YOLO format
                    x_min = float(tokens[4])
                    x_max = float(tokens[6])
                    y_min = float(tokens[5])
                    y_max = float(tokens[7])

                    object_center_x = (x_min + x_max) / 2
                    object_center_y = (y_min + y_max) / 2
                    object_width = x_max - x_min
                    object_height = y_max - y_min

                    # Normalize
                    norm_obj_center_x = object_center_x / full_width
                    norm_obj_center_y = object_center_y / full_height
                    norm_obj_width = object_width / full_width
                    norm_obj_height = object_height / full_height

                    # Add to overwrite content to eventually overwrite original label file
                    overwrite_content += f'{class_id} {norm_obj_center_x} {norm_obj_center_y} {norm_obj_width} {norm_obj_height}\n'
                        
            # Overwrite file contents with YOLOv8 format
            with open(label_path, 'w') as file:
                file.write(overwrite_content)

def validate_file(f):
    if not os.path.exists(f):
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

def parse_args():
  parser = argparse.ArgumentParser(description="DeepSORT Script")
  parser.add_argument("-d", "--data_dir", required=True, 
    type=validate_file, metavar="FILE",
    help="Data Directory Containing Training And Validation Data")
  
if __name__ == "__main__":
  args = parse_args()
  main(args.data_dir)