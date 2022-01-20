from utils.configuration import config

def config_prepare():
    my_config = config(
        model_name = "ssd_mobilenet_v2_320x320",
        labelmap_path="data/furnas_dataset_v0.06/data/label_map.pbtxt",
        fine_tune_checkpoint="data/ssd_mobilenet_v2_320x320_coco17_tpu-8/checkpoint/ckpt-0",
        num_classes=6,
        train_record_path="data/furnas_dataset_v0.06/train.record",
        test_record_path="data/furnas_dataset_v0.06/test.record",
        batch_size=1
    )

    folder = my_config.download_and_unzip_model()
    config_filepath = my_config.create_pipeline_config()
    print(f'Pipeline config created at {config_filepath}')


if __name__ == "__main__":
    config_prepare()