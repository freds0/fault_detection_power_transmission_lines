from os.path import join
import argparse
import subprocess
import yaml

def export_to_frozen_graph():
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--yaml', default='config/parameters.yaml', help='Config file YAML format')
    parser.add_argument('-p', '--pipeline_file', help='Path to your custom pipeline.config file.')
    parser.add_argument('-c', '--checkpoint_dir', help='Path to trained checkpoint directory.')
    parser.add_argument('-o', '--output_dir', help='Path to write outputs (.pb frozen graph).')
    args = parser.parse_args()

    try:
        with open(args.yaml, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file {}'.format(args.yaml))
        print(e)
        exit()

    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else join(config['pipeline_config']['checkpoint_save_path'])
    pipeline_file = args.pipeline_file if args.pipeline_file else config['pipeline_config']['pipeline_config_filepath']
    output_dir  = args.output_dir if args.output_dir else join(config['pipeline_config']['checkpoint_save_path'], 'exported')

    print("Exporting frozen graph checkpoints to {}...".format(output_dir))
    subprocess.run(["python3", "/tensorflow/models/research/object_detection/exporter_main_v2.py",
        "--input_type=image_tensor",
        "--pipeline_config_path={}".format(pipeline_file),
        "--trained_checkpoint_dir={}".format(checkpoint_dir),
        "--output_directory={}".format(output_dir)
    ])

if __name__ == "__main__":
    export_to_frozen_graph()
