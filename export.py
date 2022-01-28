import os
import argparse
import subprocess
import yaml
import os


def export_to_frozen_graph():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', default='configs/parameters.yaml')
    parser.add_argument('--pipeline_config_file', help='Path to your custom pipeline.config file.')
    parser.add_argument('--checkpoint_dir', help='Path to trained checkpoint directory.')
    parser.add_argument('--output_export_dir', help='Path to write outputs (.pb frozen graph).')
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
    output_export_dir  = args.output_export_dir if args.output_export_dir else os.path.join(config['pipeline_config']['checkpoint_save_path'], 'exported')

    print("Exporting frozen graph checkpoints to {}...".format(output_export_dir))
    subprocess.run(["python3", "/tensorflow/models/research/object_detection/exporter_main_v2.py",
        "--input_type=image_tensor",
        "--pipeline_config_path={}".format(pipeline_config_file),
        "--trained_checkpoint_dir={}".format(checkpoint_dir),
        "--output_directory={}".format(output_export_dir)
    ])

if __name__ == "__main__":
    export_to_frozen_graph()
