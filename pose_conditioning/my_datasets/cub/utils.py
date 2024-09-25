import os
from dataclasses import dataclass

@dataclass(frozen=True)
class CUB_Data_Point:
    id: int
    path: str
    cls_idx: int
    cls_name: str
    is_train: bool

def get_cub_images(root):
    cls_to_name_map = {
        int(k) - 1: v
        for k, v in [
            x.strip().split(" ")
            for x in open(os.path.join(root, "classes.txt"), "r").readlines()
        ]
    }

    id_to_cls_split = {
        int(k): int(v) - 1
        for k, v in [
            x.strip().split(" ")
            for x in open(os.path.join(root, "image_class_labels.txt"), "r").readlines()
        ]
    }

    id_to_trts_split = {
        int(k): int(v) == 1
        for k, v in [
            x.strip().split(" ")
            for x in open(os.path.join(root, "train_test_split.txt"), "r").readlines()
        ]
    }

    id_to_img_path = {
        int(k): v
        for k, v in [
            x.strip().split(" ")
            for x in open(os.path.join(root, "images.txt"), "r").readlines()
        ]
    }

    data = []
    for id, path in id_to_img_path.items():
        cls_idx = id_to_cls_split[id]
        full_path = os.path.join(root, "images", path)
        data.append(
            CUB_Data_Point(
                id, full_path, cls_idx, cls_to_name_map[cls_idx], id_to_trts_split[id]
            )
        )
        
    return data

def filter_cub(cub_data, cls_idx=0, is_train=False):
    return list(
        filter(lambda x: x.cls_idx == cls_idx and x.is_train == is_train, cub_data)
    )