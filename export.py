import argparse
import subprocess


def export_to_frozen_graph():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='./configs/mobilenet_v2_pipeline.config', help='Path to your config file.')
    parser.add_argument('--trained_checkpoint_dir', default='./mobilenet_v2_capybara_dataset', help='Path to trained checkpoint directory.')
    parser.add_argument('--output_directory', default='./mobilenet_v2_capybara_dataset/exported/', help='Path to write outputs (.pb frozen graph).')
    args = parser.parse_args()

    print("Exporting checkpoints to frozen graph...")
    subprocess.run(["python3", "/tensorflow/models/research/object_detection/exporter_main_v2.py",
        "--input_type=image_tensor",
        "--pipeline_config_path={}".format(args.config_file),
        "--trained_checkpoint_dir={}".format(args.trained_checkpoint_dir),
        "--output_directory={}".format(args.output_directory)
    ])

if __name__ == "__main__":
    export_to_frozen_graph()
