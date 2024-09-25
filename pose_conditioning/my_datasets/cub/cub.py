import os
import random
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import List

from PIL import Image
import datasets

#CUB_ROOT_DIR = Path("/local/scratch_1/datasets/CUB_200_2011")
CUB_ROOT_DIR = Path("/local/scratch/cv_datasets/CUB_200_2011")
CUB_COND_DIR = Path("/local/scratch/carlyn.1/cub_pose")

ATT_FILE = "/home/carlyn.1/diffusion/datasets/cub/attributes.txt"

@dataclass
class ImageAttribute:
    id: str
    is_present: bool
    certanty_id: int
    attribute_parts: List[str]
    
def get_attribute_key(att_file):
    # Get attribute tree
    id_to_attribute_map = {}
    with open(att_file, 'r') as f:
        for line in f.readlines():
            id, name = line.split(" ")
            root_att, base_att = name.strip().split("::")
            id_to_attribute_map[id] = [root_att, base_att]
    
    return id_to_attribute_map

def get_image_attributes(attr_map, img_att_path):
    img_attributes = defaultdict(list)
    with open(img_att_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            img_id, att_id, is_present, cert_id = line.split()[:4]
            img_att = ImageAttribute(att_id, is_present=="1", int(cert_id), attr_map[att_id])
            img_attributes[img_id].append(img_att)
    return img_attributes

def get_cub_classnames(cls_path):
    class_names = []
    with open(cls_path, 'r') as f:
        class_names = [line.split(" ")[-1].strip() for line in f.readlines()]
    return class_names

def load_image(path):
    with open(path, 'r') as f:
        return Image.open(path).convert("RGB")

class CUB(datasets.GeneratorBasedBuilder):
    """CUB dataset with attributes."""
    def _info(self):
        return datasets.DatasetInfo(
            description="CUB attribute dataset",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=get_cub_classnames(CUB_ROOT_DIR / "classes.txt")),
                    #"attributes": datasets.Value("string")
                    #"attributes_root": datasets.Sequence(feature=datasets.Value(dtype="string")),
                    #"attributes_desc": datasets.Sequence(feature=datasets.Value(dtype="string")),
                    #"base_caption": datasets.Value(dtype="string")
                    "text": datasets.Value(dtype="string"),
                    "conditioning_image": datasets.Image(),
                }
            )
        )

    def _split_generators(self, dl_manager):
        train_ids = []
        val_ids = []
        with open(CUB_ROOT_DIR / "train_test_split.txt", 'r') as f:
            for line in f.readlines():
                img_id, is_train = line.split(" ")
                is_train = is_train.strip() == "1"
                if is_train:
                    train_ids.append(img_id)
                else:
                    val_ids.append(img_id)
        
        train_classes = []
        val_classes = []            
        with open(CUB_ROOT_DIR / "image_class_labels.txt", 'r') as f:
            for line in f.readlines():
                img_id, cls_lbl = line.strip().split(" ")
                cls_lbl = int(cls_lbl) - 1 # They are listed 1 - 200
                if img_id in train_ids:
                    train_classes.append(cls_lbl)
                elif img_id in val_ids:
                    val_classes.append(cls_lbl)
                else:
                    raise AssertionError(f"{img_id} not in train or validation splits.")
                
        lbl_to_cls_name_map = {}
        with open(CUB_ROOT_DIR / "classes.txt", 'r') as f:
            for line in f.readlines():
                idx, name = line.split(" ")
                name = name.split(".")[1].replace("_", " ")
                idx = int(idx) - 1
                lbl_to_cls_name_map[idx] = name
                
        
        #id_to_attribute_map = get_attribute_key(ATT_FILE)
        #img_attributes = get_image_attributes(id_to_attribute_map, CUB_ROOT_DIR / "attributes" / "image_attribute_labels.txt")
        
        #train_atts = [img_attributes[id] for id in train_ids]
        #val_atts = [img_attributes[id] for id in val_ids]
        
        train_paths = []
        val_paths = []
        
        train_cond_paths = []
        val_cond_paths = []
        
        with open(CUB_ROOT_DIR / "images.txt", 'r') as f:
            for line in f.readlines():
                img_id, short_path = line.strip().split(" ")
                fname = short_path.split("/")[-1]
                full_path = CUB_ROOT_DIR / "images" / short_path
                cond_full_path = CUB_COND_DIR / fname
                if img_id in train_ids:
                    train_paths.append(full_path)
                    train_cond_paths.append(cond_full_path)
                elif img_id in val_ids:
                    val_paths.append(full_path)
                    val_cond_paths.append(cond_full_path)
                else:
                    raise AssertionError(f"{img_id} not in train or validation splits.")
                
        train_cls_names = [lbl_to_cls_name_map[idx] for idx in train_classes]
        val_cls_names = [lbl_to_cls_name_map[idx] for idx in train_classes]
        
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": train_paths, "labels" : train_classes, "class_names": train_cls_names, "ids": train_ids, "cond_paths": train_cond_paths}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepaths": val_paths, "labels" : val_classes, "class_names": val_cls_names, "ids": val_ids, "cond_paths": val_cond_paths}),
        ]

    def _generate_examples(self, filepaths, labels, class_names, ids, cond_paths):
        for filepath, label, cls_name, id, cond_path in zip(filepaths, labels, class_names, ids, cond_paths):
            img = load_image(filepath)
            cond_img = load_image(cond_path)
            #attribute_root = [at.attribute_parts[0] for at in atts]
            #attribute_desc = [at.attribute_parts[1] for at in atts]
            #attribute_text = build_image_string(random.sample(atts, NUM_BIRD_ATTRIBUTES))
            #yield id, {
            #    "image": img,
            #    "label" : label,
            #    "attributes_root" : attribute_root, 
            #    "attributes_desc" : attribute_desc,
            #    "base_caption" : "A photo of a bird"
            #}
            caption = f"A photo of a {cls_name}"
            yield id, {
                "image": img,
                "label" : label,
                "text" : caption,
                "conditioning_image": cond_img,
            }

