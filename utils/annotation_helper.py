import os
import tensorflow as tf
from PIL import Image
import numpy as np
from pascal_voc_writer import Writer
from object_detection.utils import label_map_util
from utils.inference_helper import run_inference_for_single_image


def generate_annotation(model, label_map, image_path, output_path):
  image = Image.open(image_path)
  image_width, image_height = image.size
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = np.array(image)
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)

  # List of the strings that is used to add correct label for each box.
  category_index = label_map_util.create_category_index_from_labelmap(label_map, use_display_name=True)

  boxes = np.array(output_dict['detection_boxes'])
  classes = np.array(output_dict['detection_classes'])
  scores = np.array(output_dict['detection_scores'])

  boxes = np.squeeze(boxes)
  classes = np.squeeze(classes)
  scores = np.squeeze(scores)

  writer = Writer(image_path, image_width, image_height)

  for index, score in enumerate(scores):
    if score < 0.5:
      continue

    label = category_index[classes[index]]['name']
    ymin, xmin, ymax, xmax = boxes[index]

    writer.addObject(label, int(xmin * image_width), int(ymin * image_height),
                     int(xmax * image_width), int(ymax * image_height))

  filename = os.path.basename(image_path)
  annotation_file = os.path.splitext(filename)[0] + '.xml'
  annotation_path = os.path.join(output_path, annotation_file)
  writer.save(annotation_path)
  print('Generating file {}...'.format(annotation_path))