import argparse
import subprocess
import yaml
import os


def execute_model_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='configs/parameters.yaml')
    parser.add_argument('--save_dir', help='Path to save evaluation results.')
    parser.add_argument('--checkpoint_dir', help='Folder to load exported checkpoint.')
    parser.add_argument('--pipeline_config_file', help='Path to your custom pipeline.config file.')
    parser.add_argument('--num_workers', default=64, type=int, help='Number of cores that can be used for the evaluation job.')
    args = parser.parse_args()

    try:
        with open(args.config_file, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file {}'.format(args.config_file))
        print(e)
        exit()

    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else os.path.join(config['pipeline_config']['checkpoint_save_path'], 'exported')
    pipeline_config_file = args.pipeline_config_file if args.pipeline_config_file else config['pipeline_config']['pipeline_config_filepath']
    save_dir  = args.save_dir if args.save_dir else os.path.join(config['pipeline_config']['checkpoint_save_path'], 'exported')

    print("Executing evaluation...")
    subprocess.run(["python3", "/tensorflow/models/research/object_detection/model_main_tf2.py",
        "--pipeline_config_path={}".format(pipeline_config_file),
        "--model_dir={}".format(checkpoint_dir),
        "--checkpoint_dir={}".format(save_dir),
        "--num_workers={}".format(args.num_workers),
        "--alsologtostderr"
    ])


if __name__ == "__main__":
    execute_model_evaluation()
