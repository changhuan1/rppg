"""UBFC-Phys stress loader for multi-ROI RGB temporal sequences."""

import csv
import glob
import os
import re

import cv2
import numpy as np
import pandas as pd

from dataset.data_loader.BaseLoader import BaseLoader


class UBFCPHYSStressRGBLoader(BaseLoader):
    """Loads UBFC-Phys as stress/non-stress multi-ROI RGB sequences.

    Labels:
        T1 -> 0, non-stress
        T2/T3 -> 1, stress
    """

    ROI_NAMES = ("forehead", "left_cheek", "right_cheek", "nose", "chin")

    def __init__(self, name, data_path, config_data, device=None):
        super().__init__(name, data_path, config_data, device)

    def __getitem__(self, index):
        data = np.load(self.inputs[index]).astype(np.float32)  # [K, T, 3]
        label = np.load(self.labels[index]).astype(np.int64)
        label = np.asarray(label).reshape(-1)[0]

        item_path = self.inputs[index]
        item_path_filename = item_path.split(os.sep)[-1]
        split_idx = item_path_filename.rindex("_")
        filename = item_path_filename[:split_idx]
        chunk_id = item_path_filename[split_idx + 6:].split(".")[0]
        return data, label, filename, chunk_id

    def get_raw_data(self, data_path):
        data_dirs = glob.glob(data_path + os.sep + "s*" + os.sep + "vid_s*_T*.avi")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")

        dirs = []
        for data_dir in data_dirs:
            filename = os.path.basename(data_dir)
            match = re.search(r"vid_(s\d+)_T(\d+)\.avi", filename)
            if match is None:
                continue
            subject = match.group(1)
            task = "T" + match.group(2)
            dirs.append({
                "index": f"{subject}_{task}",
                "path": data_dir,
                "subject": subject,
                "task": task,
            })
        return sorted(dirs, key=lambda x: x["index"])

    def split_raw_data(self, data_dirs, begin, end):
        if begin == 0 and end == 1:
            return data_dirs

        data_by_subject = {}
        for data in data_dirs:
            data_by_subject.setdefault(data["subject"], []).append(data)

        subjects = sorted(data_by_subject.keys())
        choose_range = range(int(begin * len(subjects)), int(end * len(subjects)))
        selected_subjects = [subjects[i] for i in choose_range]

        split_dirs = []
        for subject in selected_subjects:
            split_dirs.extend(data_by_subject[subject])
        return sorted(split_dirs, key=lambda x: x["index"])

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        saved_filename = data_dirs[i]["index"]
        frames = self.read_video(data_dirs[i]["path"])
        face_frames = self.crop_face_resize(
            frames,
            config_preprocess.CROP_FACE.DO_CROP_FACE,
            config_preprocess.CROP_FACE.BACKEND,
            config_preprocess.CROP_FACE.USE_LARGE_FACE_BOX,
            config_preprocess.CROP_FACE.LARGE_BOX_COEF,
            config_preprocess.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION,
            config_preprocess.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY,
            config_preprocess.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX,
            config_preprocess.RESIZE.W,
            config_preprocess.RESIZE.H,
        )

        roi_rgb = self.extract_roi_rgb_sequences(face_frames)
        roi_rgb = self.temporal_standardize(roi_rgb)
        clips = self.chunk_roi_sequences(roi_rgb, config_preprocess.CHUNK_LENGTH)
        label = self.task_to_label(data_dirs[i]["task"])
        labels = np.asarray([label for _ in range(len(clips))], dtype=np.int64)

        input_name_list, _ = self.save_multi_process(clips, labels, saved_filename)
        file_list_dict[i] = input_name_list

    def load_preprocessed_data(self):
        file_list_df = pd.read_csv(self.file_list_path)
        inputs = file_list_df["input_files"].tolist()
        if not inputs:
            raise ValueError(self.dataset_name + " dataset loading data error!")
        inputs = sorted(inputs)
        labels = [input_file.replace("input", "label") for input_file in inputs]
        self.inputs = inputs
        self.labels = labels
        self.preprocessed_data_len = len(inputs)

    @staticmethod
    def read_video(video_file):
        vid_obj = cv2.VideoCapture(video_file)
        success, frame = vid_obj.read()
        frames = []
        while success:
            frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB)
            frames.append(frame)
            success, frame = vid_obj.read()
        vid_obj.release()
        return np.asarray(frames)

    @staticmethod
    def task_to_label(task):
        if task == "T1":
            return 0
        if task in ("T2", "T3"):
            return 1
        raise ValueError(f"Unsupported UBFC-Phys task for stress label: {task}")

    @staticmethod
    def extract_roi_rgb_sequences(frames):
        """Returns [K, T, 3] ROI mean RGB sequences from face-aligned frames."""
        total_frames, height, width, _ = frames.shape
        boxes = UBFCPHYSStressRGBLoader.get_roi_boxes(width, height)
        roi_rgb = np.zeros((len(boxes), total_frames, 3), dtype=np.float32)

        for roi_idx, (x1, y1, x2, y2) in enumerate(boxes):
            roi = frames[:, y1:y2, x1:x2, :]
            if roi.size == 0:
                raise ValueError(f"Empty ROI generated: {(x1, y1, x2, y2)}")
            roi_rgb[roi_idx] = roi.mean(axis=(1, 2))
        return roi_rgb

    @staticmethod
    def get_roi_boxes(width, height):
        def box(x1, y1, x2, y2):
            return (
                int(round(x1 * width)),
                int(round(y1 * height)),
                int(round(x2 * width)),
                int(round(y2 * height)),
            )

        return [
            box(0.25, 0.08, 0.75, 0.25),  # forehead
            box(0.12, 0.38, 0.38, 0.68),  # left cheek
            box(0.62, 0.38, 0.88, 0.68),  # right cheek
            box(0.40, 0.32, 0.60, 0.62),  # nose
            box(0.35, 0.68, 0.65, 0.88),  # chin
        ]

    @staticmethod
    def temporal_standardize(roi_rgb):
        mean = roi_rgb.mean(axis=1, keepdims=True)
        std = roi_rgb.std(axis=1, keepdims=True)
        return (roi_rgb - mean) / (std + 1e-6)

    @staticmethod
    def chunk_roi_sequences(roi_rgb, chunk_length):
        clip_num = roi_rgb.shape[1] // chunk_length
        clips = [
            roi_rgb[:, i * chunk_length:(i + 1) * chunk_length, :]
            for i in range(clip_num)
        ]
        if not clips:
            raise ValueError("Video is shorter than CHUNK_LENGTH; no stress clips generated.")
        return np.asarray(clips, dtype=np.float32)
