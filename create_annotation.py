import argparse
import yaml
from os.path import join
from os import makedirs
from utils.annotation_helper import generate_annotation
from utils.model_loader import load_custom_model
from glob import glob
from tqdm import tqdm


def execute_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--yaml', default='config/parameters.yaml', help='Config file YAML format')
    parser.add_argument('-c', '--checkpoint_dir', help='Folder to load exported checkpoint.')
    parser.add_argument('-i', '--image_path', help='Input jpg images foder.')
    parser.add_argument('-e', '--image_extension', help='Image file extension: jpg or jpeg')
    parser.add_argument('-l', '--label_map',  help='Path to pbtxt file')
    parser.add_argument('-o', '--output_dir', default='./output_annotate', help='Output folder')
    args = parser.parse_args()

    try:
        with open(args.yaml, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file {}'.format(args.yaml))
        print(e)
        exit()

    model_path = args.checkpoint_dir if args.checkpoint_dir else join(config['pipeline_config']['checkpoint_save_path'], 'exported')
    label_map = args.label_map if args.label_map else config['pipeline_config']['labelmap_path']
    image_path = args.image_path if args.image_path else config['pipeline_config']['input_test_img_folder']
    img_extension = args.image_extension if args.image_extension else config['pipeline_config']['image_extension']

    print("Loading model...")
    detection_model = load_custom_model(model_path)
    image_files = join(image_path, f'*.{img_extension}')
    makedirs(args.output_dir, exist_ok=True)

    print("Executing auto-annotation...")
    for image_path in tqdm(glob(image_files)):
        generate_annotation(detection_model, label_map, image_path, args.output_dir)


if __name__ == "__main__":
    execute_inference()
