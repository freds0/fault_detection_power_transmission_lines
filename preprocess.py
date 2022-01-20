import os
import cv2
import argparse
import pandas as pd
import tqdm
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imgaug as ia
import time


def select_data(data):
    selected_data = data[data['class'] == 'isolador_falha']
    selected_data = list(selected_data.sort_values('filename').drop_duplicates('filename', keep='last')['filename'])
    return selected_data

# specify augmentations that will be executed on each image randomly


def img_func_horizontal(images, random_state, parents, hooks):
    '''
    Replace in every image each 10 row with black pixels:
    '''
    for img in images:
        img[::10] = 0
    return images


def img_func_vertical(images, random_state, parents, hooks):
    '''
    Replace in every image each 10 cols with black pixels:
    '''
    for img in images:
        img[:,::10] = 0
    return images

def keypoint_func(keypoints_on_images, random_state, parents, hooks):
    return keypoints_on_images
'''
seq = iaa.SomeOf(2, [
    # Color
    iaa.GammaContrast((0.3, 3)),         
    iaa.LinearContrast((0.8, 1.2)),      
    iaa.AddToHueAndSaturation((-10, 10)), 
    iaa.Invert(0.05),                      
    iaa.Solarize(0.1, threshold=(32, 128)),
    # Localizations
    iaa.Affine(
         scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},                     # Scale images to a value of 50 to 150% of their original size:
         translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},       # Translate images by -20 to +20% on x- and y-axis independently:
         rotate=(-15, 15),                                             # Rotate images by -45 to 45 degrees:
         shear=(-15, 15),                                              # Shear images by -16 to 16 degrees:         
    ),
    # Deformations
    iaa.PiecewiseAffine(scale=(0.01, 0.03)),                # Apply affine transformations that differ between local neighbourhoods.
    iaa.PerspectiveTransform(scale=(0.05, 0.15)),           # Apply random four point perspective transformations to images.
    iaa.ElasticTransformation(alpha=(1.0, 2.0), sigma=0.25),  # Transform images by moving pixels locally around using displacement fields.
    # Dropout
    iaa.Dropout(p=(0.05, 0.1)),                           # Augmenter that sets a certain fraction of pixels in images to zero.
    iaa.CoarseDropout((0.01,0.05), size_percent=(0.1, 0.2)),          # Augmenter that sets rectangular areas within images to zero.
    iaa.CoarsePepper((0.01,0.05), size_percent=(0.1, 0.2)),  # Replace rectangular areas in images with black-ish pixel noise.
    iaa.Salt((0.05, 0.1)),                                          # Replace pixels in images with salt noise, i.e. white-ish pixels.
    iaa.CoarseSalt((0.01,0.05), size_percent=(0.1, 0.2)),         # Replace rectangular areas in images with white-ish pixel noise.
    iaa.Pepper((0.05, 0.1)),                                        # Replace pixels in images with pepper noise, i.e. black-ish pixels.
    iaa.CoarsePepper((0.01,0.05), size_percent=(0.1, 0.2)),       # Replace rectangular areas in images with black-ish pixel noise.
    iaa.SaltAndPepper((0.05, 0.1)),                                 # Replace pixels in images with salt/pepper noise (white/black-ish colors).
    iaa.CoarseSaltAndPepper((0.01,0.05), size_percent=(0.1, 0.2)),# Replace rectangular areas in images with white/black-ish pixel noise.
    iaa.CoarseSaltAndPepper((0.01,0.05), 
        size_percent=(0.1, 0.2), per_channel=True
    ),
    # Noise
    iaa.Add((-30, 30), per_channel=0.1),                    # Add random values between -40 and 40 to images. In 50% of all images the values differ per channel (3 sampled value)
    iaa.AddElementwise((-40, 40), per_channel=0.5),          # Add values to the pixels of images with possibly different values for neighbouring pixels.    
    iaa.ImpulseNoise((0.05, 0.1)),                          # Add impulse noise to images.
    iaa.AdditiveGaussianNoise(scale=(20, 30)),               # Add noise sampled from gaussian distributions elementwise to images.
    iaa.AdditiveLaplaceNoise(scale=(0.1*255, 0.2*255)),           # Add noise sampled from laplace distributions elementwise to images.
    # Flip
    iaa.Fliplr(0.5),
    # Degradation
    iaa.JpegCompression(compression=(80, 95)),
    iaa.GaussianBlur(sigma=(1.0,3.0)),
    iaa.AverageBlur(k=(2, 5)),
    iaa.MedianBlur(k=(3, 5)), 
    iaa.BilateralBlur(                            # Blur/Denoise an image using a bilateral filter.
        d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250)
    ),
    iaa.MotionBlur(k=(5,10)),
    # Weather
    iaa.Clouds(),
    iaa.Fog(),
    iaa.Snowflakes(flake_size=(0.7, 0.95), speed=(0.001, 0.03)),
    iaa.Rain(speed=(0.1, 0.3)),
    # Lambda
    iaa.Lambda(img_func_vertical, keypoint_func),
    iaa.Lambda(img_func_horizontal, keypoint_func),
]) # apply augmenters in random order# apply augmenters in random order
'''
seq = iaa.Sequential([
    iaa.SomeOf(1, [
        # Color
        iaa.GammaContrast((0.5, 2)),
        iaa.LinearContrast((0.9, 1.1)),
        iaa.AddToHueAndSaturation((-5, 5)),
        iaa.Invert(0.05),
        iaa.Solarize(0.05, threshold=(32, 128)),

        # Dropout
        iaa.Dropout(p=(0.05, 0.1)),  # Augmenter that sets a certain fraction of pixels in images to zero.
        iaa.CoarseDropout((0.01, 0.05), size_percent=(0.1, 0.2)),
        # Augmenter that sets rectangular areas within images to zero.
        iaa.CoarsePepper((0.01, 0.05), size_percent=(0.1, 0.2)),
        # Replace rectangular areas in images with black-ish pixel noise.
        iaa.Salt((0.05, 0.1)),  # Replace pixels in images with salt noise, i.e. white-ish pixels.
        iaa.CoarseSalt((0.01, 0.05), size_percent=(0.1, 0.2)),
        # Replace rectangular areas in images with white-ish pixel noise.
        iaa.Pepper((0.05, 0.1)),  # Replace pixels in images with pepper noise, i.e. black-ish pixels.
        iaa.CoarsePepper((0.01, 0.05), size_percent=(0.1, 0.2)),
        # Replace rectangular areas in images with black-ish pixel noise.
        iaa.SaltAndPepper((0.05, 0.1)),  # Replace pixels in images with salt/pepper noise (white/black-ish colors).
        iaa.CoarseSaltAndPepper((0.01, 0.05), size_percent=(0.1, 0.2)),
        # Replace rectangular areas in images with white/black-ish pixel noise.
        iaa.CoarseSaltAndPepper((0.01, 0.05),
                                size_percent=(0.1, 0.2), per_channel=True
                                ),
        # Noise
        iaa.Add((-30, 30), per_channel=0.1),
        # Add random values between -20 and 20 to images. In 50% of all images the values differ per channel (3 sampled value)
        iaa.AddElementwise((-20, 20), per_channel=0.5),
        # Add values to the pixels of images with possibly different values for neighbouring pixels.
        iaa.ImpulseNoise((0.05, 0.1)),  # Add impulse noise to images.
        iaa.AdditiveGaussianNoise(scale=(20, 30)),  # Add noise sampled from gaussian distributions elementwise to images.
        iaa.AdditiveLaplaceNoise(scale=(0.1 * 255, 0.2 * 255)), # Add noise sampled from laplace distributions elementwise to images.
        # Flip
        iaa.Fliplr(0.5),
        # Degradation
        iaa.JpegCompression(compression=(80, 95)),
        iaa.GaussianBlur(sigma=(1.0, 3.0)),
        iaa.AverageBlur(k=(2, 5)),
        iaa.MedianBlur(k=(3, 5)),
        iaa.BilateralBlur(  # Blur/Denoise an image using a bilateral filter.
            d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250)
        ),
        iaa.MotionBlur(k=(5, 10)),
        # Weather
        iaa.Clouds(),
        iaa.Fog(),
        iaa.Snowflakes(flake_size=(0.7, 0.95), speed=(0.001, 0.03)),
        iaa.Rain(speed=(0.1, 0.3)),
        # Lambda
        iaa.Lambda(img_func_vertical, keypoint_func),
        iaa.Lambda(img_func_horizontal, keypoint_func),
    ]),
    # Localizations
    iaa.SomeOf(1, [
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # Scale images to a value of 50 to 150% of their original size:
        ),
        iaa.Affine(
            translate_percent = {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # Translate images by -20 to +20% on x- and y-axis independently:
        ),
        iaa.Affine(
            rotate = (-15, 15),  # Rotate images by -45 to 45 degrees:
        ),
        iaa.Affine(
            shear = (-15, 15),  # Shear images by -16 to 16 degrees:
        ),
        # Deformations
        iaa.PiecewiseAffine(scale=(0.01, 0.03)),  # Apply affine transformations that differ between local neighbourhoods.
        iaa.PerspectiveTransform(scale=(0.05, 0.15)),  # Apply random four point perspective transformations to images.
        iaa.ElasticTransformation(alpha=(1.0, 2.0), sigma=0.25),
        # Transform images by moving pixels locally around using displacement fields.
    ])
])


def aug_image(filename: str, df: pd.DataFrame, folder: str, augmentations: int) -> (list, list):
    """
    This function will:
     1. load the image based on the filename from the given folder
     2. load all given bounding boxes to that image from the given DataFrame
     3. apply augmentations specified by the seq variable above
     4. output images and bounding_boxes
    :param filename: str object that defines the image to be augmented
    :param df: DataFrame that stores all given bounding box information to each image
    :param folder: defines where to find the image
    :param augmentations: defines the number of augmentations to be done
    :return: list of augmented images, list of bouding_boxes for each augmented image
    """
    # load image
    img = cv2.imread(os.path.join(folder, filename))
    # create empty list for bounding_boxes
    bbs = list()
    # iterate over DataFrame to get each bounding box for that image
    for _, row in df[df.filename == filename].iterrows():
        x1 = row.xmin
        y1 = row.ymin
        x2 = row.xmax
        y2 = row.ymax
        bbs.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=row['class']))
    # concatenate all bounding boxes fro that image
    bbs = BoundingBoxesOnImage(bbs, shape=img.shape[:-1])

    # replicate the image augmentations times
    images = [img for _ in range(augmentations)]
    # replicate the bounding boxes augmentations times
    bbss = [bbs for _ in range(augmentations)]

    # augment images with bounding_boxes
    image_aug, bbs_aug = seq(images=images, bounding_boxes=bbss)

    return image_aug, bbs_aug


def save_augmentations(images: list, bbs: list, df: pd.DataFrame, filename: str, folder: str, resize: bool = False,
                       shape: (int, int) = (None, None)) -> pd.DataFrame:
    """
    This function will:
    1. store each augmented image in a new folder
    2. append the bounding_boxes from the augmented_images to the given DataFrame
    :param images: list of augmented images
    :param bbs: list of concatenated bounding boxes that relate to an augmentated image
    :param df: DataFrame that will store the information about the new bounding boxes from the augmented images
    :param filename: original filename of the original image
    :param folder: str object that defines the path to the output folder for the augmentated images
    :param resize: defines if the image should be resized or not after the augmentation
    :param shape: if the image will be reshaped, it will be reshaped into this shape
    :return: DataFrame
    """

    # iterate over the images
    for [i, img_a], bb_a in zip(enumerate(images), bbs):
        # define new name
        new_filename = os.path.basename(filename).replace('.jpg', '')
        aug_img_name = f'{new_filename}_data_aug{i}.jpg'
        # check if image should be resized
        org_shape = (None, None)
        if resize:
            org_shape = img_a.shape[:-1]
            img_a = cv2.resize(img_a, shape, interpolation=cv2.INTER_NEAREST)

        # clean bb_a --> use only bounding boxes that are still in the frame (cropping can lead to bounding boxes being
        # removed from the images)
        bb_a = bb_a.remove_out_of_image().clip_out_of_image()

        # iterate over the bounding boxes
        at_least_one_box = False
        for bbs in bb_a:
            if resize:
                bbs = bbs.project(org_shape, shape)
            arr = bbs.compute_out_of_image_fraction(img_a)
            if arr < 0.8:
                at_least_one_box = True
                x1 = int(bbs.x1)
                y1 = int(bbs.y1)
                x2 = int(bbs.x2)
                y2 = int(bbs.y2)
                c = bbs.label
                # append extracted data to the DataFrame
                height, width = img_a.shape[:-1]
                df = df.append(pd.DataFrame(data=[aug_img_name, width, height, c, x1, y1, x2, y2],
                                            index=df.columns.tolist()).T)
        if at_least_one_box:
            # save image at specified folder
            cv2.imwrite(os.path.join(folder, aug_img_name), img_a)

    return df


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='./')
    parser.add_argument('--input_folder', default='images/test')
    parser.add_argument('--output_folder', default='images/test_aug')
    parser.add_argument('--input_csv', default='annotations/test.csv')
    parser.add_argument('--output_csv', default='annotations/test_aug.csv')
    parser.add_argument('--image_extension', default='.jpg')
    parser.add_argument('--augmentations', default=5)
    parser.add_argument('--seed', default=42, help='seed: int or -1 for random')
    parser.add_argument('--resize', action='store_false')
    parser.add_argument('--new_shape', default='', help='Ex. (1280, 720)')
    args = parser.parse_args()

    if args.seed == -1:
        ia.seed(int(time.time() * 10**7))
    else:
        ia.seed(int(args.seed))

    # define number of augmentations per image
    augmentations = args.augmentations
    # specify if the image should be resized
    if args.resize:
        # define shape (should be equal to requested shape of the object detection model
        new_shape = args.new_shape
    # define input folder
    input_folder = os.path.join(args.base_dir, args.input_folder)
    # define and create output_folder
    output_folder = os.path.join(args.base_dir, args.output_folder)
    # create output folder
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # 1. load DataFrame with annotations
    data = pd.read_csv(os.path.join(args.base_dir, args.input_csv))
    # 2. select data
    img_list = select_data(data)
    # 3 get a list of selected images
    #img_list = list(selected_data.sort_values('filename').drop_duplicates('filename', keep='last')['filename'])

    print('Number of images found: {}'.format(len(img_list)))

    # create a new pandas table for the augmented images' bounding boxes
    aug_data= pd.DataFrame(columns=data.columns.tolist())

    # 2. iterate over the images and augmentate them
    for filename in tqdm.tqdm(img_list):
        # augment image
        aug_images, aug_bbs = aug_image(filename, data, input_folder, args.augmentations)
        # store augmentations in new DataFrame and save image
        aug_data = save_augmentations(aug_images, aug_bbs, aug_data, filename, output_folder, args.resize, new_shape)

    # save new DataFrame
    #aug_data.to_csv(os.path.join(args.base_dir, args.output_csv))

    # Merge csv datasets
    new_data =  pd.concat([data,aug_data], ignore_index=True)
    new_data.to_csv(os.path.join(args.base_dir, args.output_csv), index=False)
