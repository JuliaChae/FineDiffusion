from pathlib import Path
from collections import defaultdict

from PIL import Image
import datasets

from pose_cond_tools.dataset_tools import AnimalPoseTools

ANIMALPOSE_ROOT_DIR = Path("/local/scratch/carlyn.1/datasets/animalpose/")
ANIMALPOSE_TOOLS = AnimalPoseTools(str(ANIMALPOSE_ROOT_DIR))


def get_colors():
    face_color = (100, 200, 5)
    limb_color = (255, 0, 0)
    other_color = (200, 200, 50)
    kp_color = (0, 0, 255)
    body_color = (0, 0, 255)

    animalpose_kpt_colors = {
        "left_ear": kp_color,
        "right_ear": kp_color,
        "left_eye": kp_color,
        "right_eye": kp_color,
        "nose": kp_color,
        "throat": kp_color,
        "tailbase": kp_color,
        "withers": kp_color,
        "left_front_elbow": kp_color,
        "left_front_knee": kp_color,
        "left_front_paw": kp_color,
        "right_front_elbow": kp_color,
        "right_front_knee": kp_color,
        "right_front_paw": kp_color,
        "left_back_elbow": kp_color,
        "left_back_knee": kp_color,
        "left_back_paw": kp_color,
        "right_back_elbow": kp_color,
        "right_back_knee": kp_color,
        "right_back_paw": kp_color,
    }

    # [[0, 1], [0, 2], [1, 2], [0, 3], [1, 4], [2, 17], [18, 19],
    # [5, 9], [6, 10], [7, 11], [8, 12], [9, 13], [10, 14], [11, 15], [12, 16]]
    animalpose_link_colors = {
        (0, 1): (137, 255, 255),  #
        (0, 2): face_color,  #
        (1, 2): face_color,  #
        (0, 3): face_color,  #
        (1, 4): face_color,  #
        (2, 17): (75, 162, 138),
        (18, 19): body_color,
        (5, 9): limb_color,  #
        (6, 10): limb_color,  #
        (7, 11): limb_color,
        (8, 12): limb_color,  #
        (9, 13): limb_color,  #
        (10, 14): limb_color,  #
        (11, 15): limb_color,  #
        (12, 16): limb_color,  #
        (19, 18): body_color,  #
        (19, 8): body_color,  #
        (19, 7): body_color,  #
        (17, 6): body_color,  #
        (17, 5): body_color,  #
        (18, 17): body_color,  #
    }

    ann_data = ANIMALPOSE_TOOLS.annotation_data

    keypoint_names = ann_data["categories"][1][
        "keypoints"
    ]  # names are the same across all animals (categories)
    skeleton_links = ann_data["categories"][1][
        "skeleton"
    ]  # skeltons are the same across all animals (categories)

    kpt_colors = [tuple(list(animalpose_kpt_colors[name])) for name in keypoint_names]
    link_colors = [
        tuple(list(animalpose_link_colors[tuple(link)])) for link in skeleton_links
    ]

    return kpt_colors, link_colors


def collect_and_sort_classnames():
    class_names = sorted(
        [(c["id"], c["name"]) for c in ANIMALPOSE_TOOLS.annotation_data["categories"]],
        key=lambda x: x[0],
    )
    class_names = [c[1] for c in class_names]
    return class_names


def collect_and_sort_skeletons():
    skeletons = sorted(
        [
            (c["id"], c["skeleton"])
            for c in ANIMALPOSE_TOOLS.annotation_data["categories"]
        ],
        key=lambda x: x[0],
    )
    skeletons = [c[1] for c in skeletons]
    return skeletons


def get_animalpose_classnames():
    class_names = collect_and_sort_classnames()
    return class_names


def load_image(path):
    return Image.open(path).convert("RGB")


class AnimalPose(datasets.GeneratorBasedBuilder):
    """AP10K dataset with pose conditioning."""

    def _info(self):
        return datasets.DatasetInfo(
            description="Animalpose pose dataset",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "label": datasets.ClassLabel(names=get_animalpose_classnames()),
                    "text": datasets.Value(dtype="string"),
                    "conditioning_image": datasets.Image(),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        data = defaultdict(list)
        categories = collect_and_sort_classnames()
        skeletons = collect_and_sort_skeletons()

        ann_data = ANIMALPOSE_TOOLS.annotation_data

        imgs_with_mul = []
        all_img_ids = [ann["image_id"] for ann in ann_data["annotations"]]
        for img_id in set(all_img_ids):
            c = all_img_ids.count(img_id)
            if c > 1:
                imgs_with_mul.append(img_id)

        for i, ann in enumerate(ann_data["annotations"]):
            img_id = ann["image_id"]
            if img_id in imgs_with_mul:
                continue  # For now, let's skip images with mulitiple annotations in them. Later we can use bounding boxes to crop out.
            data["ids"].append(img_id)
            cat_id = ann["category_id"] - 1
            data["class_idx"].append(cat_id)  # Starts at 1, so substract 1
            data["class_names"].append(categories[cat_id])
            data["skeletons"].append(skeletons[cat_id])
            img_path = ANIMALPOSE_TOOLS.get_image_path_from_annotation(ann)
            data["img_paths"].append(img_path)
            data["keypoints"].append(
                ANIMALPOSE_TOOLS.get_keypoints(ann, keep_visible=True)
            )

        train_data = data

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={**train_data},
            ),
        ]

    def _generate_examples(
        self,
        ids,
        class_idx,
        class_names,
        skeletons,
        img_paths,
        keypoints,
    ):
        kpt_colors, link_colors = get_colors()

        for id, label, cls_name, skeleton, filepath, kpts in zip(
            ids, class_idx, class_names, skeletons, img_paths, keypoints
        ):
            img = load_image(filepath)
            cond_img = ANIMALPOSE_TOOLS.create_pose_image(
                None,
                return_black_background=True,
                kpt_color=kpt_colors,
                link_color=link_colors,
                keypoints=kpts,
                skeleton=skeleton,
                image=img,
                id=id,
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
