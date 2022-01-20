import argparse
import subprocess


def execute_model_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', default='./checkpoints_mobilenet_v2_capybara_dataset/',
                        help='Path to save your checkpoints')
    parser.add_argument('--model_dir', default='./checkpoints_mobilenet_v2_capybara_dataset/',
                        help='Path to model')
    parser.add_argument('--config_file', default='./configs/mobilenet_v2_pipeline.config', help='Path to your config file.')
    parser.add_argument('--num_workers', default=64, type=int, help='Number of cores that can be used for the evaluation job.')
    args = parser.parse_args()

    print("Executing evaluation...")
    subprocess.run(["python3", "/tensorflow/models/research/object_detection/model_main_tf2.py",
        "--pipeline_config_path={}".format(args.config_file),
        "--model_dir={}".format(args.checkpoint_path),
        "--checkpoint_dir={}".format(args.checkpoint_path),
        "--num_workers={}".format(args.num_workers),
        "--alsologtostderr"
    ])

if __name__ == "__main__":
    execute_model_evaluation()
