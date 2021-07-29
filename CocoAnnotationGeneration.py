# Register a custom labelme dataset for Detectron2

# Function to find the dataset dicts in the required format

def get_dataset_dicts(directory):
    classes = ['Class_one', 'Class_two', 'Class_three']
    dataset_dicts = []
    idx = 0
    list_sorted = sorted([file for file in os.listdir(directory) if file.endswith('.json')])
    
    for idx, filename in enumerate(list_sorted):
        json_file = os.path.join(directory, filename)
        with open(json_file) as f:
            img_anns = json.load(f)

        record = {}
        
        filename = os.path.join(directory, img_anns["imagePath"])
        
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = img_anns["imageHeight"]
        record["width"] = img_anns["imageWidth"]
      
        annos = img_anns["shapes"]
        objs = []
        for anno in annos:
            px = [a[0] for a in anno['points']]
            py = [a[1] for a in anno['points']]
            poly = [(x, y) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": classes.index(anno['label']),
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts



# Register the Dataset

for d in ["train", "test"]:
    DatasetCatalog.register("Dataset_" + d, lambda d=d: get_dataset_dicts("/home/path/Datasetdirectory/" + d))
    MetadataCatalog.get("Dataset_" + d).set(thing_classes=['Class_one', 'Class_two', 'Class_three'])
strawberry_metadata = MetadataCatalog.get("Dataset_train")



# Converting annotations to mscoco format

from detectron2.data.datasets.coco import convert_to_coco_json

for f in ["train", "test"]:
    convert_to_coco_json("Dataset_" + f, "/home/path/Dataset_{}.json".format(f), allow_cached=False) # Saves 2 files named "Dataset_train" and "Dataset_test"
