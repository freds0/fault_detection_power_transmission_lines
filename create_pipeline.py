import argparse
import yaml
from utils.pipeline_creation import model_config
import ast

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./')
    parser.add_argument('-y', '--yaml', default='config/parameters.yaml', help='Config file YAML format')
    parser.add_argument('-m', '--model_name', help='Choose a model name.')
    parser.add_argument('-f', '--fine_tune_checkpoint', help='Define a checkpoint path to fine tuning.')
    parser.add_argument('-b', '--batch_size', help='Batch size number.')
    parser.add_argument('-o', '--output_filepath', help='Output pipeline config filepath.')
    args = parser.parse_args()

    try:
        with open(args.yaml, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file {}'.format(args.yaml))
        print(e)
        exit()

    model_name = args.model_name if args.model_name else config['pipeline_config']['model_name']
    fine_tune_checkpoint = args.fine_tune_checkpoint if args.fine_tune_checkpoint else None
    batch_size = args.batch_size if args.batch_size else config['pipeline_config']['batch_size']
    labelmap_path = config['pipeline_config']['labelmap_path']
    train_record_path = config['pipeline_config']['train_record_path']
    test_record_path = config['pipeline_config']['test_record_path']
    num_classes = len(ast.literal_eval(config['pipeline_config']['classes_names']))
    output_filepath = args.output_filepath if args.output_filepath else config['pipeline_config']['pipeline_config_filepath']

    pipeline_config = model_config(
        model_name,
        labelmap_path,
        num_classes,
        train_record_path,
        test_record_path,
        batch_size,
        output_filepath,
        fine_tune_checkpoint
    )

    result_config_filepath = pipeline_config.create_pipeline_config()

    print(f'Pipeline config created at {result_config_filepath}')
