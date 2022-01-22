from object_detection.utils import config_util
from object_detection import model_lib_v2
from shutil import move
from utils.models_links import models_links_dic
import wget
import tarfile
import re
import os


class model_config:

    def __init__(self, model_name, labelmap_path = '', num_classes = 0, train_record_path = '', test_record_path = '', batch_size = 1, output_filepath = '', fine_tune_checkpoint = None, data_dir = 'data', temp_dir = 'tmp'):
        self.model_name = model_name
        self.labelmap_path = labelmap_path
        self.num_classes = num_classes
        self.train_record_path = train_record_path
        self.test_record_path = test_record_path
        self.batch_size = batch_size
        self.output_filepath = output_filepath
        self.fine_tune_checkpoint = fine_tune_checkpoint
        self.data_dir = data_dir
        self.temp_dir = temp_dir

    @classmethod
    def get_link_model(cls, model_name):
        return models_links_dic[model_name]

    def download_model(self):
        url = self.get_link_model(self.model_name)
        targz_output_filename = os.path.basename(url)
        print(f"Downloading model {self.model_name}")
        targz_output_filepath = os.path.join('data', targz_output_filename)
        if not(os.path.exists(targz_output_filepath)):
            wget.download(url, targz_output_filepath)
        return targz_output_filepath

    @classmethod
    def get_folder_model(cls, model_name):
        default = cls(model_name = model_name)
        url_model = default.get_link_model(model_name)
        model_tar_filename = os.path.basename(url_model)
        model_dir = os.path.join(default.data_dir, model_tar_filename.replace('.tar.gz', ''))
        return model_dir

    def get_finetune_checkpoint(self):
        return self.fine_tune_checkpoint if self.fine_tune_checkpoint else os.path.join(self.get_folder_model(self.model_name), 'checkpoint/ckpt-0')

    def unzip_model(self, targz_filepath):
        print(f"\nUnzipping file {targz_filepath}")
        # Descompactar o arquivo tar.gz
        tar = tarfile.open(targz_filepath)
        tar.extractall(path=self.data_dir)
        tar.close()
        # Obtem o nome da pasta removendo '.tar.gz'
        model_folder = targz_filepath.replace('.tar.gz', '')
        return model_folder

    def download_and_unzip_model(self):
        targz_filename = self.download_model()
        return self.unzip_model(targz_filename)

    def regular_expression_pipeline_config(self, pipeline_config, output_pipeline_config):
        # Realiza a leitura do arquivo de configurado do modelo escolhido
        with open(pipeline_config) as f:
            config_content = f.read()

        # Set labelmap path
        config_content = re.sub('label_map_path: ".*?"',
                        'label_map_path: "{}"'.format(self.labelmap_path), config_content)

        # Set fine_tune_checkpoint path
        config_content = re.sub('fine_tune_checkpoint: ".*?"',
                        'fine_tune_checkpoint: "{}"'.format(self.get_finetune_checkpoint()), config_content)

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

    def dynamic_pipeline_config(self, config_path):
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
        config_util.save_pipeline_config(pipeline_config, self.temp_dir)

        move(os.path.join(self.temp_dir, 'pipeline.config'), self.output_filepath)

    def create_pipeline_config(self):
        self.download_and_unzip_model()
        input_pipeline_config = os.path.join(self.get_folder_model(self.model_name), 'pipeline.config')
        temp_pipeline_config = os.path.join(self.temp_dir, 'pipeline.config')
        os.makedirs(self.temp_dir, exist_ok=True)
        self.regular_expression_pipeline_config(input_pipeline_config, temp_pipeline_config)
        self.dynamic_pipeline_config(temp_pipeline_config)
        return self.output_filepath