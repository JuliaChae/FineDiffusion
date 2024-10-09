import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from mmengine.structures import InstanceData
from mmpose.structures import PoseDataSample
from mmpose.visualization import PoseLocalVisualizer

from PIL import Image
import datasets

AP10K_ROOT_DIR = Path("/local/scratch/carlyn.1/datasets/ap-10k/")
SPLIT = 1


def collect_and_sort_classnames(ann_data):
    class_names = sorted(
        [(c["id"], c["name"]) for c in ann_data["categories"]], key=lambda x: x[0]
    )
    class_names = [c[1] for c in class_names]
    return class_names


def get_ap10k_classnames(root_path):
    ann_path = root_path / f"annotations/ap10k-train-split{SPLIT}.json"
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
                    "text": datasets.Value(dtype="string"),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        def collect(ann_path):
            # TODO: Add background data labels too. WIll have to add a map from idx to name manually based on GitHub
            with open(ann_path, "r") as f:
                ann_data = json.load(f)

            data = defaultdict(list)
            categories = collect_and_sort_classnames(ann_data)

            imgs_with_mul = []
            all_img_ids = [ann["image_id"] for ann in ann_data["annotations"]]
            for img_id in set(all_img_ids):
                c = all_img_ids.count(img_id)
                if c > 1:
                    # print(img_id, c) # id 71 has 11 annotations!!!
                    imgs_with_mul.append(img_id)

            image_info_map = {x["id"]: x for x in ann_data["images"]}
            for i, ann in enumerate(ann_data["annotations"]):
                img_id = ann["image_id"]
                if img_id in imgs_with_mul:
                    continue  # For now, let's skip images with mulitiple annotations in them. Later we can use bounding boxes to crop out.
                data["ids"].append(img_id)
                cat_id = ann["category_id"] - 1
                data["class_idx"].append(cat_id)  # Starts at 1, so substract 1
                data["class_names"].append(categories[cat_id])
                data["img_paths"].append(
                    AP10K_ROOT_DIR / "data" / image_info_map[img_id]["file_name"]
                )

            return data

        train_data = collect(
            AP10K_ROOT_DIR / f"annotations/ap10k-train-split{SPLIT}.json"
        )
        val_data = collect(AP10K_ROOT_DIR / f"annotations/ap10k-val-split{SPLIT}.json")
        test_data = collect(
            AP10K_ROOT_DIR / f"annotations/ap10k-test-split{SPLIT}.json"
        )

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

    def _generate_examples(
        self,
        ids,
        class_idx,
        class_names,
        img_paths,
    ):
        for (
            id,
            label,
            cls_name,
            filepath,
        ) in zip(
            ids,
            class_idx,
            class_names,
            img_paths,
        ):
            img = load_image(filepath)

            caption = f"A photo of a {cls_name}"
            yield (
                id,
                {
                    "image": img,
                    "text": caption,
                },
            )
