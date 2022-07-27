# Use the following code to directly register your dataset for training using the generated MSCOCO format json files.

from detectron2.data.datasets import register_coco_instances

register_coco_instances("Dataset_train", {}, "/home/path/Dataset_train.json", "/home/path/datasetdirectory/train/") # Registers a dataset named "Dataset_train" by taking annotations from the "Dataset_train.json" file.
register_coco_instances("Dataset_test", {}, "/home/path/Dataset_test.json", "/home/path/datasetdirectory/test/")
dataset_metadata = MetadataCatalog.get("Dataset_train").set(thing_classes=['Class_one', 'Class_two', 'Class_three'])
