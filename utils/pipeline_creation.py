from object_detection.utils import config_util
from object_detection import model_lib_v2
from shutil import copy, rmtree
import utils.models_zoo
import wget
import tarfile
import re
from os.path import join, exists, basename
import tempfile

zoo = utils.models_zoo.zoo()

class model_config:

    def __init__(self, model_name, labelmap_path = '', num_classes = 0, train_record_path = '', test_record_path = '', batch_size = 1, output_filepath = '', fine_tune_checkpoint = None, data_dir = 'data'):
        self.model_name = model_name
        self.labelmap_path = labelmap_path
        self.num_classes = num_classes
        self.train_record_path = train_record_path
        self.test_record_path = test_record_path
        self.batch_size = batch_size
        self.fine_tune_checkpoint = fine_tune_checkpoint
        self.data_dir = data_dir
        self.temp_dir = tempfile.TemporaryDirectory()
        print(f"Created temporary directory {self.temp_dir.name}.")
        if not output_filepath:
            self.output_filepath = join(self.data_dir, self.__get_folder_model())
        else:
            self.output_filepath = output_filepath

    def __download_model(self):
        url = zoo.get_link_model(self.model_name)
        targz_output_filename = basename(url)
        print(f"Downloading model {self.model_name}")
        targz_output_filepath = join(self.data_dir, targz_output_filename)
        if not(exists(targz_output_filepath)):
            wget.download(url, out=targz_output_filepath)
        return targz_output_filepath

    def __get_folder_model(self):

        return join(
                self.data_dir,
                zoo.get_folder_model(self.model_name)
        )

    def __get_finetune_checkpoint(self):

        if self.fine_tune_checkpoint:
            return self.fine_tune_checkpoint
        else:
            checkpoint_filepath = join(
                self.__get_folder_model(),
                'checkpoint/ckpt-0'
            )
            return checkpoint_filepath

    def __unzip_model(self, targz_filepath):
        print(f"\nUnzipping file {targz_filepath}")
        # Unzip tar.gz file
        tar = tarfile.open(targz_filepath)
        tar.extractall(path=self.data_dir)
        tar.close()
        # Get the folder's name
        model_folder = targz_filepath.replace('.tar.gz', '')
        return model_folder

    def __download_and_unzip_model(self):
        targz_filename = self.__download_model()
        return self.__unzip_model(targz_filename)

    def __regular_expression_pipeline_config(self, pipeline_config, output_pipeline_config):
        # Read model's config file
        with open(pipeline_config) as f:
            config_content = f.read()

        # Set labelmap path
        config_content = re.sub('label_map_path: ".*?"',
                        'label_map_path: "{}"'.format(self.labelmap_path), config_content)

        # Set fine_tune_checkpoint path
        config_content = re.sub('fine_tune_checkpoint: ".*?"',
                        'fine_tune_checkpoint: "{}"'.format(self.__get_finetune_checkpoint()), config_content)

        # Set train tf-record file path
        config_content = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")',
                        'input_path: "{}"'.format(self.train_record_path), config_content)

        # Set test tf-record file path
        config_content = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")',
                        'input_path: "{}"'.format(self.test_record_path), config_content)

        # Set number of classes.
        config_content = re.sub('num_classes: \d+',
                        'num_classes: {}'.format(self.num_classes), config_content)

        # Set batch size
        config_content = re.sub('batch_size: [0-9]+',
                        'batch_size: {}'.format(self.batch_size), config_content)

        # Set fine-tune checkpoint type to detection
        config_content = re.sub('fine_tune_checkpoint_type: "classification"',
                        'fine_tune_checkpoint_type: "{}"'.format('detection'), config_content)

        with open(output_pipeline_config, 'w') as f:
            f.write(config_content)

    def __dynamic_pipeline_config(self, config_path):
        '''
        Source: https://stackoverflow.com/questions/55323907/dynamically-editing-pipeline-config-for-tensorflow-object-detection
        '''
        # Load the pipeline config as a dictionary
        pipeline_config_dict = config_util.get_configs_from_pipeline_file(config_path)

        # Example 1: Override the train tfrecord path
        pipeline_config_dict['train_input_config'].tf_record_input_reader.input_path[0] = self.train_record_path
        # Example 2: Override the eval tfrecord path
        pipeline_config_dict['eval_input_config'].tf_record_input_reader.input_path[0] = self.test_record_path

        pipeline_config = config_util.create_pipeline_proto_from_configs(pipeline_config_dict)
        # Example 2: Save the pipeline config to disk
        config_util.save_pipeline_config(pipeline_config, self.temp_dir.name)
        # Copy from temp dir to final folder
        copy(join(self.temp_dir.name, 'pipeline.config'), self.output_filepath)

    def create_pipeline_config(self):
        # Download and unzip model data
        self.__download_and_unzip_model()
        # Create pipeline file
        input_pipeline_config = join(self.__get_folder_model(), 'pipeline.config')
        temp_pipeline_config = join(self.temp_dir.name, 'pipeline.config')
        self.__regular_expression_pipeline_config(input_pipeline_config, temp_pipeline_config)
        self.__dynamic_pipeline_config(temp_pipeline_config)
        # Clear temp dir
        self.temp_dir.cleanup()

        return self.output_filepath