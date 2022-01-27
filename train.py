import argparse
import subprocess
import yaml

def execute_train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='configs/parameters.yaml')
    parser.add_argument('--checkpoint_save_path', help='Path to save your checkpoints')
    parser.add_argument('--pipeline_file', help='Path to your pipeline config file.')
    parser.add_argument('--checkpoint_every_n', type=int, help='The number of steps per checkpoint.')
    parser.add_argument('--num_train_steps'. type=int, help='Total number of trainning steps.')
    parser.add_argument('--num_workers', default=64, type=int, help='Number of cores that can be used for the training job.')
    args = parser.parse_args()

    try:
        with open(args.config_file, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file {}'.format(args.config_file))
        exit()

    checkpoint_save_path = args.checkpoint_save_path if args.checkpoint_save_path else config['pipeline_config']['checkpoint_save_path']
    pipeline_file = args.pipeline_file if args.pipeline_file else config['pipeline_config']['pipeline_config_filepath']
    num_train_steps = args.num_train_steps if args.num_train_steps else config['pipeline_config']['num_train_steps']
    checkpoint_every_n = args.checkpoint_every_n if args.checkpoint_every_n else config['pipeline_config']['checkpoint_every_n']

    print("Executing train...")
    subprocess.run(["python3", "/tensorflow/models/research/object_detection/model_main_tf2.py",
        "--model_dir={}".format(checkpoint_save_path),
        "--pipeline_config_path={}".format(pipeline_file),
        "--checkpoint_every_n={}".format(checkpoint_every_n),
        "--num_train_steps={}".format(num_train_steps),
        "--num_workers={}".format(args.num_workers),
        "--alsologtostderr"
    ])

if __name__ == "__main__":
    execute_train()
