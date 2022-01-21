import argparse
import os
import tqdm
import yaml
from shutil import copyfile
from utils.model_prepare import model_config
from tools.generate_tfrecord import create_tf_record


def prepare_config(config):

    my_config = model_config(
        **config
    )

    folder = my_config.download_and_unzip_model()
    config_filepath = my_config.create_pipeline_config()
    print(f'Pipeline config created at {config_filepath}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./')
    parser.add_argument('--config_file', default='configs/parameters.yaml')
    args = parser.parse_args()

    try:
        with open(args.config_file, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file {}'.format(args.config_file))
        exit()

    prepare_config(config['prepare'])

    # copy images files to same folder
    print('Copying image files...')
    for filename in tqdm.tqdm(os.listdir(config['dataset']['input_folder'])):
        copyfile(os.path.join(config['dataset']['input_folder'], filename), os.path.join(config['dataset']['output_folder'], filename))

    # create tfrecords
    print('Generating TFRecords...')
    create_tf_record(config['preprocess']['output_csv'], config['dataset']['output_folder'], config['prepare']['train_record_path'])