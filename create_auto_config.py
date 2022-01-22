import argparse
import os
import tqdm
import yaml
from utils.model_configuration import model_config

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./')
    parser.add_argument('--config_file', default='configs/parameters.yaml')
    parser.add_argument('--model_name', help='Choose a model name.')
    parser.add_argument('--finetune_checkpoint', help='Define a checkpoint path to fine tuning.')
    parser.add_argument('--batch_size', help='Batch size number.')
    args = parser.parse_args()

    try:
        with open(args.config_file, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file {}'.format(args.config_file))
        exit()

    model_name = args.model_name if args.model_name else config['pipeline_config']['model_name']
    finetune_checkpoint = args.finetune_checkpoint if args.finetune_checkpoint else None
    batch_size = args.batch_size if args.batch_size else config['pipeline_config']['batch_size']
    labelmap_path = config['pipeline_config']['labelmap_path']
    train_record_path = config['pipeline_config']['train_record_path']
    test_record_path = config['pipeline_config']['test_record_path']
    num_classes = config['pipeline_config']['num_classes']
    output_filepath = config['pipeline_config']['pipeline_config_filepath']

    pipeline_config = model_config(
        model_name,
        labelmap_path,
        num_classes,
        train_record_path,
        test_record_path,
        batch_size,
        output_filepath,
        finetune_checkpoint
    )
    result_config_filepath = pipeline_config.create_pipeline_config()
    print(f'Pipeline config created at {result_config_filepath}')