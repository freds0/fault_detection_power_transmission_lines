from object_detection.utils import config_util
from shutil import copy
from glob import glob
import argparse
import subprocess
import yaml
import tempfile
from os.path import join, isdir
#from os import rename

def get_last_checkpoint(checkpoints_folder):
    if not isdir(checkpoints_folder):
        print('Error: {} is not a folder.'.format(checkpoints_folder))
        return False

    checkpoints_list = glob(join(checkpoints_folder, '*ckpt*'))
    checkpoints_list = sorted(checkpoints_list, reverse=True)
    if len(checkpoints_list) == 0:
        print('Error: folder {} not contains ckpt-### file.'.format(checkpoints_folder))
        return False

    checkpoint_filename = checkpoints_list[0].split('.')[0] # get ckpt-## filename
    return checkpoint_filename

def set_checkpoint_at_pipeline(pipeline_filepath, checkpoint_filepath):
    '''
    Source: https://stackoverflow.com/questions/55323907/dynamically-editing-pipeline-config-for-tensorflow-object-detection
    '''
    # Load the pipeline config as a dictionary
    pipeline_config_dict = config_util.get_configs_from_pipeline_file(pipeline_filepath)
    print(pipeline_config_dict['train_config'].fine_tune_checkpoint)
    # Override the fine_tune_checkpoint filepath
    pipeline_config_dict['train_config'].fine_tune_checkpoint = checkpoint_filepath

    pipeline_config = config_util.create_pipeline_proto_from_configs(pipeline_config_dict)

    with tempfile.TemporaryDirectory() as temp_dir:
        print('Created temporary directory', temp_dir)
        # Save the pipeline config to disk
        config_util.save_pipeline_config(pipeline_config, temp_dir)
        # Create a backup of original pipeline config
        #rename(pipeline_filepath, pipeline_filepath + '.bkp')
        # Copy from temp dir to final folder
        copy(join(temp_dir, 'pipeline.config'), pipeline_filepath)

def execute_train():
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--yaml', default='config/parameters.yaml', help='Config file YAML format')
    parser.add_argument('-c', '--checkpoint_path', help='Path to save your checkpoints')
    parser.add_argument('-p', '--pipeline_file', help='Path to your pipeline config file.')
    parser.add_argument('-f', '--fine_tune_checkpoint', default='', help='Path to your checkpoint to fine tuning. ie: ./checkpoints/')
    parser.add_argument('-s', '--num_train_steps', type=int, help='Total number of training steps.')
    parser.add_argument('-n', '--num_workers', default=64, type=int, help='Number of cores that can be used for the training job.')
    parser.add_argument('--checkpoint_every_n', type=int, help='The number of steps per checkpoint.')
    args = parser.parse_args()

    try:
        with open(args.yaml, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file {}'.format(args.yaml))
        print(e)
        exit()

    checkpoint_path = args.checkpoint_path if args.checkpoint_path else config['pipeline_config']['checkpoint_save_path']
    pipeline_file = args.pipeline_file if args.pipeline_file else config['pipeline_config']['pipeline_config_filepath']
    num_train_steps = args.num_train_steps if args.num_train_steps else config['pipeline_config']['num_train_steps']
    checkpoint_every_n = args.checkpoint_every_n if args.checkpoint_every_n else config['pipeline_config']['checkpoint_every_n']

    print("Executing train...")

    if args.fine_tune_checkpoint:
        print("Loading checkpoint from: {}".format(args.fine_tune_checkpoint))
        checkpoint_filepath = get_last_checkpoint(args.fine_tune_checkpoint)
        if not checkpoint_filepath:
            return False
        print(checkpoint_filepath)
        set_checkpoint_at_pipeline(pipeline_file, checkpoint_filepath)

    subprocess.run(["python3", "/tensorflow/models/research/object_detection/model_main_tf2.py",
        "--model_dir={}".format(checkpoint_path),
        "--pipeline_config_path={}".format(pipeline_file),
        "--checkpoint_every_n={}".format(checkpoint_every_n),
        "--num_train_steps={}".format(num_train_steps),
        "--num_workers={}".format(args.num_workers),
        "--alsologtostderr"
    ])


if __name__ == "__main__":
    execute_train()
