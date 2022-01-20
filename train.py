import argparse
import subprocess


def execute_train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default='./checkpoints_mobilenet_v2_capybara_dataset/',
                        help='Path to save your checkpoints')
    parser.add_argument('--config_file', default='./configs/mobilenet_v2_pipeline.config', help='Path to your config file.')
    parser.add_argument('--checkpoint_every_n', default=1000, type=int, help='The number of steps per checkpoint.')
    parser.add_argument('--num_train_steps', default=100000, type=int, help='Total number of trainning steps.')
    parser.add_argument('--num_workers', default=64, type=int, help='Number of cores that can be used for the training job.')
    args = parser.parse_args()

    print("Executing train...")
    subprocess.run(["python3", "/tensorflow/models/research/object_detection/model_main_tf2.py",
        "--model_dir={}".format(args.checkpoint_path),
        "--pipeline_config_path={}".format(args.config_file),
        "--checkpoint_every_n={}".format(args.checkpoint_every_n),
        "--num_train_steps={}".format(args.num_train_steps),
        "--num_workers={}".format(args.num_workers),
        "--alsologtostderr"
    ])

if __name__ == "__main__":
    execute_train()
