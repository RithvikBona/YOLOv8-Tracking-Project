import os
import random
import shutil
import argparse

def main(train_ratio, data_path):

    ## Define folder paths
    unsplit_img_path = os.path.join(data_path, 'images')
    unsplit_label_path = os.path.join(data_path, 'labels')
    train_path = os.path.join(data_path, 'train')
    val_path = os.path.join(data_path, 'val')


    img_list = [filename for filename in os.listdir(unsplit_img_path)]

    ## Shuffle the image filename list
    random.shuffle(img_list)

    ## Number of image-label pairs in each set
    train_size = int(len(img_list) * train_ratio)
    val_size = int(len(img_list) * (1 - train_ratio))

    ## Create destination folders and sub-folders if they don't exist
    for curr_path in [train_path, val_path]:
        if not os.path.exists(curr_path):
            os.makedirs(curr_path)
            os.makedirs(os.path.join(curr_path, 'images'))
            os.makedirs(os.path.join(curr_path, 'labels'))

    ## Copy image files and corresponding labels to destination folders
    for i, f in enumerate(img_list):
        if i < train_size:
            dest_folder = train_path
        else:
            dest_folder = val_path
        
        shutil.copy(os.path.join(unsplit_img_path, f), os.path.join(dest_folder, 'images/' + f))
        # Get label filename and copy that over as well
        label_filename = f[:-3] + 'txt'
        shutil.copy(os.path.join(unsplit_label_path, label_filename), os.path.join(dest_folder, 'labels/' + label_filename))

def validate_file(f):
    if not os.path.exists(f):
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

def parse_args():
  parser = argparse.ArgumentParser(description="Dataset divider")
  parser.add_argument("--train_ratio", required=False, default=0.7,
    help="Train ratio - Ex 0.7 means splitting data in 70% train and 30% validation")
  parser.add_argument("-d", "--data_dir", required=True, 
    type=validate_file, metavar="FILE",
    help="Data Directory Containing Images and Labels")
  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  main(float(args.train_ratio), args.data_dir)