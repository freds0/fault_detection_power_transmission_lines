import argparse
import os
from utils.annotation_helper import generate_annotation
from utils.loader import load_custom_model
from glob import glob
from tqdm import tqdm


def execute_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./saved_models/mobilenet_v2_capybara_dataset',
                        help='Folder to saved_model')
    parser.add_argument('--image_path', default='./examples/test/', help='Input jpg images folder')
    parser.add_argument('--label_map', default='./capybara_dataset/data/object-detection.pbtxt',
                        help='Path to pbtxt file')
    parser.add_argument('--output_path', default='./output', help='Output folder')
    args = parser.parse_args()

    print("Loading model...")
    detection_model = load_custom_model(args.model_path)

    image_files = os.path.join(args.image_path, '*.jpg')

    os.makedirs(args.output_path, exist_ok=True)

    print("Executing auto-annotation...")
    for image_path in tqdm(glob(image_files)):
        generate_annotation(detection_model, args.label_map, image_path, args.output_path)


if __name__ == "__main__":
    execute_inference()
