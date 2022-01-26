from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
"""
Source code originally published in https://github.com/datitran/raccoon_dataset

Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
"""
import argparse
import yaml
import os
import io
import pandas as pd
import tensorflow.compat.v1 as tf
#import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict
from utils.class_to_int import class_text_to_int


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def create_tf_record(csv_input, path, output_path):

    examples = pd.read_csv(csv_input)
    grouped = split(examples, 'filename')

    writer = tf.python_io.TFRecordWriter(output_path)
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./')
    parser.add_argument('--config_file', default='configs/parameters.yaml')
    parser.add_argument('--input_train_csv', help='Input train csv filepath.')
    parser.add_argument('--input_test_csv', help='Input test csv filepath.')
    parser.add_argument('--images_train_dir', help='Train images folder.')
    parser.add_argument('--images_test_dir', help='Test images folder.')
    parser.add_argument('--output_train_tfrecord', help='Output train TFRecord filepath.')
    parser.add_argument('--output_test_tfrecord', help='Output test TFRecord filepath.')
    parser.add_argument('--only_train', action='store_true', help='Execute only for train dataset')
    args = parser.parse_args()

    try:
        with open(args.config_file, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print('Error reading the config file {}'.format(args.config_file))
        exit()

    config_input_train_csv = config['preprocess']['output_data_aug_csv'] if config['pipeline_config']['use_data_aug'] else config['pipeline_config']['input_train_csv']
    config_train_images_dir = config['preprocess']['output_data_aug_imgs_folder'] if config['pipeline_config']['use_data_aug'] else config['pipeline_config']['input_train_img_folder']

    train_csv_filepath = os.path.join(args.base_dir, args.input_train_csv) if args.input_train_csv else config_input_train_csv
    train_images_dir = os.path.join(args.base_dir, args.images_train_dir) if args.images_train_dir else config_train_images_dir
    train_output_path = os.path.join(args.base_dir, args.output_train_tfrecord) if args.output_train_tfrecord else config['pipeline_config']['train_record_path']

    create_tf_record(train_csv_filepath, train_images_dir, train_output_path)

    if not (args.only_train):
        test_csv_filepath = os.path.join(args.base_dir, args.input_test_csv) if args.input_test_csv else config['pipeline_config']['input_test_csv']
        test_images_dir = os.path.join(args.base_dir, args.images_test_dir) if args.images_test_dir else config['pipeline_config']['input_test_img_folder']
        test_output_path = os.path.join(args.base_dir, args.output_test_tfrecord) if args.output_test_tfrecord else config['pipeline_config']['test_record_path']

        create_tf_record(test_csv_filepath, test_images_dir, test_output_path)