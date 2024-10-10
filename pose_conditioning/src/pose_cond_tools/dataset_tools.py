import os
import json
import random
import copy

from abc import ABC, abstractmethod
from typing import List

import numpy as np
from PIL import Image

from mmengine.structures import InstanceData
from mmpose.structures import PoseDataSample
from mmpose.visualization import PoseLocalVisualizer


class DatasetTools(ABC):
    @abstractmethod
    def crop_img_and_adjust_points(
        self, img: Image, bbox: List[int], keypoints: List[List[int]]
    ):
        pass

    @abstractmethod
    def get_image_paths(self):
        pass

    @abstractmethod
    def sample_random(self):
        pass

    @abstractmethod
    def sample_random_from_category(self, category):
        pass

    @abstractmethod
    def get_keypoints(self, annotation, get_only_visible=False, keep_visible=False):
        pass

    @abstractmethod
    def get_bbox(self, annotation):
        pass

    @abstractmethod
    def get_skeleton(self, category_id):
        pass

    @abstractmethod
    def get_image_path_from_annotation(self, annotation):
        pass

    @abstractmethod
    def create_pose_image(
        self,
        annotation,
        return_black_background,
        kpt_color,
        link_color,
        keypoints,
        skeleton,
        image,
        id,
    ):
        pass


class AP10KTools(DatasetTools):
    def __init__(self, root):
        pass

    def crop_img_and_adjust_points(
        self, img: Image, bbox: List[int], keypoints: List[List[int]]
    ):
        pass

    def get_image_paths(self):
        pass

    def sample_random(self):
        pass

    def sample_random_from_category(self, category):
        pass

    def get_keypoints(self, annotation, get_only_visible=False, keep_visible=False):
        pass

    def get_bbox(self, annotation):
        pass

    def get_skeleton(self, category_id):
        pass

    def get_image_path_from_annotation(self, annotation):
        pass

    def create_pose_image(
        self,
        annotation,
        return_black_background,
        kpt_color,
        link_color,
        keypoints,
        skeleton,
        image,
        id,
    ):
        pass


class AnimalPoseTools(DatasetTools):
    def __init__(self, root):
        self.root = root
        with open(os.path.join(root, "keypoints.json"), "r") as f:
            self.annotation_data = json.load(f)

    def crop_img_and_adjust_points(
        self, img: Image, bbox: List[int], keypoints: List[List[int]]
    ):
        x1, y1, x2, y2 = bbox
        cropped_img = img.crop((x1, y1, x2, y2))
        cropped_keypoints = []
        for x, y, is_visible in keypoints:
            if is_visible > 0:
                cropped_keypoints.append([x - x1, y - y1, is_visible])
            else:
                cropped_keypoints.append([0, 0, 0])

        return cropped_img, cropped_keypoints

    def get_image_paths(self):
        img_paths = []
        for local_path in self.annotation_data["images"].values():
            img_paths.append(os.path.join(self.root, "images", local_path))
        return img_paths

    def sample_random(self):
        annotation = random.sample(self.annotation_data["annotations"], k=1)[0]
        image_path = self.get_image_path_from_annotation(annotation)
        return image_path, annotation

    def sample_random_from_category(self, category):
        filtered_samples = [
            ann
            for ann in self.annotation_data["annotations"]
            if ann["category_id"] == category
        ]
        annotation = random.sample(filtered_samples, k=1)[0]
        image_path = self.get_image_path_from_annotation(annotation)
        return image_path, annotation

    def get_keypoints(self, annotation, get_only_visible=False, keep_visible=False):
        keypoints = []
        for x, y, is_valid in annotation["keypoints"]:
            if is_valid > 0 or not get_only_visible:
                pts = [x, y]
                if keep_visible:
                    pts += [is_valid]
                keypoints.append(pts)
        return keypoints

    def get_bbox(self, annotation):
        return annotation["bbox"]

    def get_skeleton(self, category_id):
        skeleton_links = copy.deepcopy(
            self.annotation_data["categories"][category_id]["skeleton"]
        )
        # Dataset seems to be missing links according to pictures
        skeleton_links += [[19, 18]]
        skeleton_links += [[19, 8]]
        skeleton_links += [[19, 7]]
        skeleton_links += [[17, 6]]
        skeleton_links += [[17, 5]]
        skeleton_links += [[18, 17]]

        return skeleton_links

    def get_image_path_from_annotation(self, annotation):
        local_image_path = self.annotation_data["images"][str(annotation["image_id"])]
        image_path = os.path.join(self.root, "images", local_image_path)
        return image_path

    def create_pose_image(
        self,
        annotation,
        return_black_background=False,
        kpt_color="red",
        link_color="green",
        keypoints=None,
        skeleton=None,
        image=None,
        id=0,
    ):
        if keypoints is None:
            keypoints = annotation["keypoints"]
        if image is None:
            image = Image.open(self.get_image_path_from_annotation(annotation))
        if skeleton is None:
            skeleton = self.get_skeleton(annotation["category_id"])

        xy_keypoints = []
        is_visible_pts = []
        for x, y, is_visible in keypoints:
            xy_keypoints.append([x, y])
            is_visible_pts.append(is_visible)

        pose_local_visualizer = PoseLocalVisualizer(
            radius=4, line_width=2, link_color=link_color, kpt_color=kpt_color
        )
        if return_black_background:
            image = np.zeros_like(np.array(image))
        gt_instances = InstanceData()
        gt_instances.keypoints = np.array([xy_keypoints])
        gt_instances.keypoints_visible = np.array([is_visible_pts])
        gt_pose_data_sample = PoseDataSample()
        gt_pose_data_sample.gt_instances = gt_instances
        dataset_meta = {"skeleton_links": skeleton}
        pose_local_visualizer.set_dataset_meta(dataset_meta)
        pose_img = pose_local_visualizer.add_datasample(
            f"image_{id}", np.array(image), gt_pose_data_sample, draw_pred=False
        )

        return Image.fromarray(pose_img)
