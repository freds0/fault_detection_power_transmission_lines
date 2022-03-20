import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imgaug as ia
import pandas as pd
import random
import cv2
import os

ia.seed(42)

def img_func_horizontal(images, random_state, parents, hooks):
    '''
    Replace in every image each n row with black pixels:
    '''
    n = random.randint(2,50)
    for img in images:
        img[::n] = 0
    return images


def img_func_vertical(images, random_state, parents, hooks):
    '''
    Replace in every image each n cols with black pixels:
    '''
    n = random.randint(2,50)
    for img in images:
        img[:,::n] = 0
    return images


def keypoint_func(keypoints_on_images, random_state, parents, hooks):
    return keypoints_on_images


def define_augmentations(config):
    '''
    Specify augmentations that will be executed on each image randomly
    :param config:
    :return: seq
    '''
    seq = iaa.Sequential([
        iaa.SomeOf(2, [
            # Colors
            iaa.OneOf([
                iaa.GammaContrast(
                    (config['GammaContrastMin'], config['GammaContrastMax'])),

                iaa.LinearContrast(
                    (config['LinearContrastMin'], config['LinearContrastMax'])),

                iaa.AddToHueAndSaturation(
                    (config['AddToHueAndSaturationMin'], config['AddToHueAndSaturationMax'])),
            ]),
            # Dropout
            iaa.OneOf([
                # Augmenter that sets a certain fraction of pixels in images to zero.
                iaa.Dropout(
                    p=(config['DropoutMin'], config['DropoutMax'])),

                # Augmenter that sets rectangular areas within images to zero.
                iaa.CoarseDropout(
                    p=(config['CoarseDropoutPerMin'], config['CoarseDropoutPerMax']),
                    size_percent=(config['CoarseDropoutPerSizeMin'], config['CoarseDropoutPerSizeMax'])),

                # Augmenter that sets pixels in images with pepper noise, i.e. black-ish pixels.
                iaa.Pepper(
                    (config['PepperMin'], config['PepperMax'])),

                # Augmenter that sets rectangular areas in images with black-ish pixels.
                iaa.CoarsePepper(
                    p=(config['CoarsePepperPerMin'], config['CoarsePepperPerMax']),
                    size_percent=(config['CoarsePepperPerSizeMin'], config['CoarsePepperPerSizeMax'])),

                # Augmenter that sets pixels in images with salt noise, i.e. white-ish pixels.
                iaa.Salt(
                    (config['SaltMin'], config['SaltMax'])),

                # Augmenter that sets rectangular areas in images with white-ish pixel noise.
                iaa.CoarseSalt(
                    p=(config['CoarseSaltPerMin'], config['CoarseSaltPerMax']),
                    size_percent=(config['CoarseSaltPerSizeMin'], config['CoarseSaltPerSizeMax'])),

                # Augmenter that sets pixels in images with salt/pepper noise (white/black-ish colors).
                iaa.SaltAndPepper(
                    p=(config['SaltAndPepperMin'], config['SaltAndPepperMax'])),

                iaa.CoarseSaltAndPepper(
                    p=(config['CoarseSaltAndPepperMin'], config['CoarseSaltAndPepperMax']),
                    size_percent=(config['CoarseSaltAndPepperPerSizeMin'], config['CoarseSaltAndPepperPerSizeMax']),
                    per_channel=False),

                iaa.CoarseSaltAndPepper(
                    p=(config['CoarseSaltAndPepperMin'], config['CoarseSaltAndPepperMax']),
                    size_percent=(config['CoarseSaltAndPepperPerSizeMin'], config['CoarseSaltAndPepperPerSizeMax']),
                    per_channel=True)
            ]),
            # Noises
            iaa.OneOf([
                iaa.Add(
                    value=(config['AddValueMin'], config['AddValueMax']),
                    per_channel=config['AddValuePercentagePerChannel']),

                iaa.AddElementwise(
                    value=(config['AddElementwiseValueMin'], config['AddElementwiseValueMax']),
                    per_channel=config['AddElementwisePercentagePerChannel']),

                # Add impulse noise to images.
                iaa.ImpulseNoise(
                    p=(config['ImpulseNoisePerMin'], config['ImpulseNoisePerMin'])),

                # Add noise sampled from gaussian distributions elementwise to images.
                iaa.AdditiveGaussianNoise(
                    scale=(config['AdditiveGaussianNoiseScaleMin'], config['AdditiveGaussianNoiseScaleMax'])),

                # Add noise sampled from laplace distributions elementwise to images.
                iaa.AdditiveLaplaceNoise(
                    scale=(config['AdditiveLaplaceNoiseScaleMin'], config['AdditiveLaplaceNoiseScaleMax']))
            ]),
            # Localizations
            iaa.OneOf([
                iaa.Affine(
                    scale={"x": (config['ScaleXMin'], config['ScaleXMax']),
                           "y": (config['ScaleYMin'], config['ScaleYMax'])}),

                # Translate images by percentage on x- and y-axis independently:
                iaa.Affine(
                    translate_percent = {"x": (config['TranslateXMin'], config['TranslateXMax']),
                                         "y": (config['TranslateYMin'], config['TranslateYMax'])}),

                # Rotate images by some degrees.
                iaa.Affine(
                    rotate = (config['RotateMin'], config['RotateMax'])),

                iaa.Affine(
                    shear = (config['ShearMin'], config['ShearMax']))
            ]),
            # Deformations
            iaa.OneOf([
                # Apply affine transformations that differ between local neighbourhoods.
                iaa.PiecewiseAffine(
                    scale=(config['PieceWiseScaleMin'], config['PieceWiseScaleMax'])),

                # Apply random four point perspective transformations to images.
                iaa.PerspectiveTransform(
                    scale=(config['PerspectiveTransformScaleMin'], config['PerspectiveTransformScaleMax'])),

                # Transform images by moving pixels locally around using displacement fields.
                iaa.ElasticTransformation(
                    alpha=(config['ElasticTransformationAlphaMin'], config['ElasticTransformationAlphaMax']),
                    sigma=(config['ElasticTransformationSigmaMax'], config['ElasticTransformationSigmaMax']))
            ]),
            # Degradation
            iaa.OneOf([
                iaa.JpegCompression(
                    compression=(config['JpegCompressionMin'], config['JpegCompressionMax'])),

                iaa.GaussianBlur(
                    sigma=(config['GaussianBlurMin'], config['GaussianBlurMax'])),

                iaa.AverageBlur(
                    k=(config['AverageBlurKMin'], config['AverageBlurKMax'])),

                iaa.MedianBlur(
                    k=(config['MedianBlurKMin'], config['MedianBlurKMax'])),

                # Blur/Denoise an image using a bilateral filter.
                iaa.BilateralBlur(
                    d=(config['BilateralBlurDistanceMin'], config['BilateralBlurDistanceMax']),
                    sigma_color=(config['BilateralBlurSigmaColorMin'], config['BilateralBlurSigmaColorMax']),
                    sigma_space=(config['BilateralBlurSigmaSpaceMin'], config['BilateralBlurSigmaColorMax'])),

                iaa.MotionBlur(k=(config['MotionBlurMin'], config['MotionBlurMax'])),
            ]),
            # Weather
            iaa.OneOf([
                iaa.Clouds(),

                iaa.Fog(),

                iaa.Snowflakes(
                    flake_size=(config['SnowflakesSizeMin'], config['SnowflakesSizeMax']),
                    speed=(config['SnowflakesSpeedMin'], config['SnowflakesSpeedMax'])),
            ]),

            # Horizontal Flip
            iaa.Sometimes(0.4, iaa.Fliplr(config['HorizontalFlipPercentage'])),

            # Vertical/Horizontal Lines
            iaa.Sometimes(0.1, iaa.Lambda(img_func_vertical, keypoint_func)),
            iaa.Sometimes(0.1, iaa.Lambda(img_func_horizontal, keypoint_func))
        ])
    ])
    return seq


def aug_image(filename: str, df: pd.DataFrame, config, folder: str, augmentations: int) -> (list, list):
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

    data_aug_sequence = define_augmentations(config)
    # augment images with bounding_boxes
    image_aug, bbs_aug = data_aug_sequence(images=images, bounding_boxes=bbss)

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

                df = pd.concat([df, pd.DataFrame(data=[aug_img_name, width, height, c, x1, y1, x2, y2],
                                            index=df.columns.tolist()).T])
        if at_least_one_box:
            # save image at specified folder
            cv2.imwrite(os.path.join(folder, aug_img_name), img_a)

    return df

