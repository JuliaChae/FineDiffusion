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


def get_colors():
    ap10k_kpt_colors = {
        "left_eye": (255, 0, 255),
        "right_eye": (170, 0, 255),
        "nose": (255, 0, 0),
        "neck": (255, 85, 0),
        "root_of_tail": (41, 163, 129),  # Custom!
        "left_shoulder": (85, 255, 0),
        "left_elbow": (0, 255, 0),
        "left_front_paw": (0, 255, 85),  # In place of left wrist
        "right_shoulder": (255, 170, 0),
        "right_elbow": (255, 255, 0),
        "right_front_paw": (170, 255, 85),  # In place of right wrist
        "left_hip": (0, 85, 255),
        "left_knee": (0, 0, 255),
        "left_back_paw": (85, 0, 255),  # In place of left ankle
        "right_hip": (0, 85, 255),
        "right_knee": (0, 255, 255),
        "right_back_paw": (0, 170, 255),  # In place of right ankle
    }

    ap10k_link_colors = {
        (1, 2): (153, 0, 153),
        (1, 3): (153, 0, 102),
        (2, 3): (102, 0, 153),
        (3, 4): (0, 0, 153),  # Neck
        (4, 5): (100, 100, 100),  # Back
        (4, 6): (153, 51, 0),  # Left shoulderblade
        (6, 7): (102, 153, 0),  # Left arm
        (7, 8): (51, 153, 0),  # Left forearm
        (4, 9): (153, 0, 0),  # Right shoulderblade
        (9, 10): (153, 102, 0),  # Right arm
        (10, 11): (153, 153, 0),  # Right forearm
        (5, 12): (0, 153, 153),  # Left upper leg
        (12, 13): (0, 102, 153),  # Left lower leg
        (13, 14): (0, 51, 153),  # Left ankle/foot
        (5, 15): (0, 153, 0),  # right upper leg
        (15, 16): (0, 153, 51),  # right lower leg
        (16, 17): (0, 153, 102),  # right ankle/foot
    }

    ann_path = AP10K_ROOT_DIR / f"annotations/ap10k-train-split{SPLIT}.json"
    with open(ann_path, "r") as f:
        ann_data = json.load(f)

    keypoint_names = ann_data["categories"][0][
        "keypoints"
    ]  # names are the same across all animals (categories)
    skeleton_links = ann_data["categories"][0][
        "skeleton"
    ]  # skeltons are the same across all animals (categories)

    kpt_colors = [tuple(list(ap10k_kpt_colors[name])) for name in keypoint_names]
    link_colors = [
        tuple(list(ap10k_link_colors[tuple(link)])) for link in skeleton_links
    ]

    return kpt_colors, link_colors


def collect_and_sort_classnames(ann_data):
    class_names = sorted(
        [(c["id"], c["name"]) for c in ann_data["categories"]], key=lambda x: x[0]
    )
    class_names = [c[1] for c in class_names]
    return class_names


def collect_and_sort_skeletons(ann_data):
    skeletons = sorted(
        [(c["id"], c["skeleton"]) for c in ann_data["categories"]], key=lambda x: x[0]
    )
    skeletons = [c[1] for c in skeletons]
    # Have to subtract all values by 1, since the skeleton is 1 indexed
    for i, skeleton in enumerate(skeletons):
        for j, link in enumerate(skeleton):
            skeletons[i][j][0] -= 1
            skeletons[i][j][1] -= 1
    return skeletons


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
        def collect(ann_path):
            # TODO: Add background data labels too. WIll have to add a map from idx to name manually based on GitHub
            with open(ann_path, "r") as f:
                ann_data = json.load(f)

            data = defaultdict(list)
            categories = collect_and_sort_classnames(ann_data)
            skeletons = collect_and_sort_skeletons(ann_data)

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
                data["skeletons"].append(skeletons[cat_id])
                data["img_paths"].append(
                    AP10K_ROOT_DIR / "data" / image_info_map[img_id]["file_name"]
                )
                # assert (
                #    image_info_sorted[i]["id"] == img_id
                # ), f"Data not aligned. Image ids did not match. ({image_info_sorted[i]['id']} != {img_id})"

                # Based on the file here: https://github.com/AlexTheBad/AP-10K/blob/main/tools/dataset/parse_animalpose_dataset.py
                # It seems that the format is <x>, <y>, <is_valid>, <x2>, <y2>, <is_valid2>, ...?
                # As long as the <is_valid> is not 0, it should be good. Even though the file only puts either 0 or 2
                ann_kps = ann["keypoints"]
                keypoints = []
                keypoint_visible = []
                for i in range(len(ann_kps) // 3):
                    si = i * 3
                    x, y, is_valid = ann_kps[si : si + 3]
                    keypoints.append([x, y])
                    keypoint_visible.append(is_valid > 0)
                data["keypoints"].append(keypoints)
                data["keypoint_visible"].append(keypoint_visible)

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

    def create_cond_img(
        self, id, img_shape, kpts, kptv, skeleton, kpt_colors, link_colors
    ):
        image = np.zeros(img_shape)

        pose_local_visualizer = PoseLocalVisualizer(
            radius=8, line_width=3, link_color=link_colors, kpt_color=kpt_colors
        )
        gt_instances = InstanceData()
        gt_instances.keypoints = np.array([kpts])
        gt_instances.keypoints_visible = np.array([kptv]).astype(np.uint8)
        gt_pose_data_sample = PoseDataSample()
        gt_pose_data_sample.gt_instances = gt_instances
        dataset_meta = {"skeleton_links": skeleton}
        pose_local_visualizer.set_dataset_meta(dataset_meta)
        pose_img = pose_local_visualizer.add_datasample(
            f"image_{id}", image, gt_pose_data_sample, draw_pred=False
        )
        return Image.fromarray(pose_img)

    def _generate_examples(
        self,
        ids,
        class_idx,
        class_names,
        skeletons,
        img_paths,
        keypoints,
        keypoint_visible,
    ):
        kpt_colors, link_colors = get_colors()

        for id, label, cls_name, skeleton, filepath, kpts, kptv in zip(
            ids,
            class_idx,
            class_names,
            skeletons,
            img_paths,
            keypoints,
            keypoint_visible,
        ):
            img = load_image(filepath)
            cond_img = self.create_cond_img(
                id, np.array(img).shape, kpts, kptv, skeleton, kpt_colors, link_colors
            )

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
