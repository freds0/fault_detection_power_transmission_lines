import os
import argparse
import yaml
import pandas as pd
import tqdm
from utils.data_augmentation import aug_image, save_augmentations
import time
from shutil import copyfile

def select_data(data):
    selected_data = data[data['class'] != 'isolador_ok']
    selected_data = list(selected_data.sort_values('filename').drop_duplicates('filename', keep='last')['filename'])
    return selected_data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./')
    parser.add_argument('--config_file', default='./configs/parameters.yaml')
    parser.add_argument('--input_folder', help='input images folder.')
    parser.add_argument('--output_folder', help='Folder to save data augmented images.')
    parser.add_argument('--input_csv', help='Input csv filepath.')
    parser.add_argument('--output_csv', help='Output csv filepath')
    parser.add_argument('--image_extension', help='Image file extension. Ex. .jpg or .jpeg')
    parser.add_argument('--aug_per_file',  help='Total number of augmentations per file.')
    parser.add_argument('--seed', default=42, help='seed: int or -1 for random')
    parser.add_argument('--resize', action='store_true')
    parser.add_argument('--new_shape', default='(1280, 720)', help='Ex. (1280, 720)')
    args = parser.parse_args()

    try:
        with open(args.config_file, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file {}'.format(args.config_file))
        exit()
    '''
    if args.seed == -1:
        ia.seed(int(time.time() * 10**7))
    else:
        ia.seed(int(args.seed))
    '''
    # total number of augmentations per image file
    aug_per_file = args.aug_per_file if args.aug_per_file else config['preprocess']['aug_per_file']
    # define input folder
    input_folder = os.path.join(args.base_dir, args.input_folder) if args.input_folder else config['pipeline_config']['input_train_img_folder']
    # define and create output_folder
    output_folder = os.path.join(args.base_dir, args.output_folder) if args.output_folder else config['preprocess']['output_data_aug_imgs_folder']
    # define csv filepath
    input_csv_filepath = os.path.join(args.base_dir, args.input_csv) if args.input_csv else config['pipeline_config']['input_train_csv']
    # define csv output filepath
    output_csv_filepath = os.path.join(args.base_dir, args.output_csv) if args.output_csv else config['preprocess']['output_data_aug_csv']

    # create output folder
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # 1. load DataFrame with annotations
    data = pd.read_csv(input_csv_filepath)
    # 2. select data
    img_list = select_data(data)

    print('Number of images found: {}'.format(len(img_list)))

    # create a new pandas table for the augmented images' bounding boxes
    aug_data= pd.DataFrame(columns=data.columns.tolist())

    # 2. iterate over the images and augmentate them
    print('Generating data augmented files...')
    for filename in tqdm.tqdm(img_list):
        # augment image
        aug_images, aug_bbs = aug_image(filename, data, config['preprocess']['augmentations'], input_folder, aug_per_file)
        # store augmentations in new DataFrame and save image
        aug_data = save_augmentations(aug_images, aug_bbs, aug_data, filename, output_folder, args.resize, args.new_shape)

    # Merge csv datasets
    new_data =  pd.concat([data,aug_data], ignore_index=True)
    # save new DataFrame
    new_data.to_csv(output_csv_filepath, index=False)

    # copy images files to same folder
    print('Copying image files...')
    for filename in tqdm.tqdm(os.listdir(config['pipeline_config']['input_train_img_folder'])):
        copyfile(os.path.join(config['pipeline_config']['input_train_img_folder'], filename), os.path.join(config['preprocess']['output_data_aug_imgs_folder'], filename))