import argparse
import subprocess
import yaml

def execute_model_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--yaml', default='config/parameters.yaml', help='Config file YAML format')
    parser.add_argument('-c', '--checkpoint_dir', help='Path to directory where your training job writes checkpoints.')
    parser.add_argument('-p', '--pipeline_file', help='Path to your custom pipeline.config file.')
    parser.add_argument('-o', '--output_dir', help='Folder to save evaluation results.')
    parser.add_argument('-n', '--num_workers', default=64, type=int, help='Number of cores that can be used for the evaluation job.')
    args = parser.parse_args()

    try:
        with open(args.yaml, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file {}'.format(args.yaml))
        print(e)
        exit()

    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else config['pipeline_config']['checkpoint_save_path']
    pipeline_file = args.pipeline_file if args.pipeline_file else config['pipeline_config']['pipeline_config_filepath']
    output_dir  = args.output_dir if args.output_dir else config['pipeline_config']['checkpoint_save_path']
    timeout = 3

    print("Executing evaluation...")
    subprocess.run(["python3", "/tensorflow/models/research/object_detection/model_main_tf2.py",
        "--pipeline_config_path={}".format(pipeline_file),
        "--checkpoint_dir={}".format(checkpoint_dir),                    
        "--model_dir={}".format(output_dir),
        "--num_workers={}".format(args.num_workers),
        "--eval_timeout={}".format(timeout),
        "--alsologtostderr"
    ])


if __name__ == "__main__":
    execute_model_evaluation()
