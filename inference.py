import argparse
import os
from utils.inference_helper import generate_inference
from utils.loader import load_custom_model
from glob import glob
from tqdm import tqdm
import yaml

def execute_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='configs/parameters.yaml')
    parser.add_argument('--checkpoint_dir', help='Folder to load exported checkpoint.')
    parser.add_argument('--image_path', help='Input jpg images folder')
    parser.add_argument('--label_map', help='Path to pbtxt file')
    parser.add_argument('--output_dir', default='./output_inference', help='Output folder')
    args = parser.parse_args()

    try:
        with open(args.config_file, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file {}'.format(args.config_file))
        exit()

    model_path = args.checkpoint_dir if args.checkpoint_dir else os.path.join(config['pipeline_config']['checkpoint_save_path'], 'exported')
    label_map = args.label_map if args.label_map else config['pipeline_config']['labelmap_path']
    image_path = args.image_path if args.image_path else config['pipeline_config']['input_test_img_folder']

    print("Loading model...")
    detection_model = load_custom_model(model_path)
    
    image_files = os.path.join(image_path, '*.jpg')
    os.makedirs(args.output_dir, exist_ok=True)

    print("Executing inference...")
    for image_path in tqdm(glob(image_files)):
        generate_inference(detection_model, label_map, image_path, args.output_dir)


if __name__ == "__main__":
    execute_inference()
