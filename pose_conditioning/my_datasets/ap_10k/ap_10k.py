import os
import json
import types
import random
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import List

from PIL import Image
import datasets

AP10K_ROOT_DIR = Path("/local/scratch/carlyn.1/datasets/ap-10k/")


def collect_and_sort_classnames(ann_data):
    class_names = sorted(
        [(c["id"], c["name"]) for c in ann_data["categories"]], key=lambda x: x[0]
    )
    class_names = [c[0] for c in class_names]
    return class_names


def get_ap10k_classnames(root_path):
    ann_path = root_path / "annotations/ap10k-train-split1.json"
    with open(ann_path, "r") as f:
        ann_data = json.load(f)
        class_names = collect_and_sort_classnames(ann_data)
        return class_names


def load_image(path):
    with open(path, "r") as f:
        return Image.open(path).convert("RGB")


class AP10K(datasets.GeneratorBasedBuilder):
    """AP10K dataset with pose conditioning."""

    def _info(self):
        return datasets.DatasetInfo(
            description="CUB attribute dataset",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(
                        names=get_ap10k_classnames(AP10K_ROOT_DIR)
                    ),
                    # "attributes": datasets.Value("string")
                    # "attributes_root": datasets.Sequence(feature=datasets.Value(dtype="string")),
                    # "attributes_desc": datasets.Sequence(feature=datasets.Value(dtype="string")),
                    # "base_caption": datasets.Value(dtype="string")
                    "text": datasets.Value(dtype="string"),
                    "conditioning_image": datasets.Image(),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        def collect(ann_data):
            # TODO: Add background data labels too. WIll have to add a map from idx to name manually based on GitHub
            data = defaultdict(list)
            categories = collect_and_sort_classnames(ann_data)
            for i, ann in enumerate(ann_data["annotations"]):
                img_id = ann["image_id"]
                data["ids"].append(img_id)
                cat_id = ann["category_id"] - 1
                data["class_idx"].append(cat_id)  # Starts at 1, so substract 1
                data["class_names"].append(categories[cat_id])
                data["img_paths"].append(
                    AP10K_ROOT_DIR / "data" / ann["images"][i]["file_name"]
                )
                assert (
                    ann["images"][i]["id"] == img_id
                ), "Data not aligned. Image ids did not match."

                # Based on the file here: https://github.com/AlexTheBad/AP-10K/blob/main/tools/dataset/parse_animalpose_dataset.py
                # It seems that the format is <x>, <y>, <is_valid>, <x2>, <y2>, <is_valid2>, ...?
                # As long as the <is_valid> is not 0, it should be good. Even though the file only puts either 0 or 2
                ann_kps = ann["keypoints"]
                for i in range(len(ann_kps) // 3):
                    si = i * 3
                    x, y, is_valid = ann_kps[si : si + 3]
                    data["keypoints"].append((x, y))
                    data["keypoint_visible"].append(is_valid > 0)

            return data

        train_data = collect(AP10K_ROOT_DIR / "annotations/ap10k-train-split1.json")
        val_data = collect(AP10K_ROOT_DIR / "annotations/ap10k-train-split1.json")
        test_data = collect(AP10K_ROOT_DIR / "annotations/ap10k-train-split1.json")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={**train_data},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={**val_data},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={**test_data},
            ),
        ]

    def _generate_examples(self, filepaths, labels, class_names, ids, cond_paths):
        # TODO: finish here
        for filepath, label, cls_name, id, cond_path in zip(
            filepaths, labels, class_names, ids, cond_paths
        ):
            img = load_image(filepath)
            cond_img = load_image(cond_path)
            # attribute_root = [at.attribute_parts[0] for at in atts]
            # attribute_desc = [at.attribute_parts[1] for at in atts]
            # attribute_text = build_image_string(random.sample(atts, NUM_BIRD_ATTRIBUTES))
            # yield id, {
            #    "image": img,
            #    "label" : label,
            #    "attributes_root" : attribute_root,
            #    "attributes_desc" : attribute_desc,
            #    "base_caption" : "A photo of a bird"
            # }
            caption = f"A photo of a {cls_name}"
            yield (
                id,
                {
                    "image": img,
                    "label": label,
                    "text": caption,
                    "conditioning_image": cond_img,
                },
            )
