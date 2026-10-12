"""PhysDrive RGB-NIR 数据集的 dataloader。"""

import glob
import os

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio

from dataset.data_loader.BaseLoader import BaseLoader


class PhysDriveLoader(BaseLoader):
    """用于读取 PhysDrive 配对 RGB/NIR 视频和 BVP 标签的 loader。

    原始数据目录结构：
        PhysDrive(Publication)/
            AMW3/
                AS/
                    Video/RGB.mp4
                    Video/IR.mp4
                    Label/BVP.mat

    预处理后的 clip 保存为：
        *_rgb{i}.npy    RGB clip
        *_nir{i}.npy    NIR clip
        *_label{i}.npy  BVP 标签 clip
    """

    MIN_LABEL_STD = 1e-8

    def __init__(self, name, data_path, config_data):
        super().__init__(name, data_path, config_data)

    def __getitem__(self, index):
        """返回 RGB、BVP 标签、NIR、样本名和 chunk id。"""
        rgb = np.load(self.rgbs[index], mmap_mode="r")
        label = np.load(self.labels[index], mmap_mode="r")
        nir = np.load(self.nirs[index], mmap_mode="r")

        rgb = self._format_clip(rgb)
        nir = self._format_clip(nir)
        label = np.array(label, dtype=np.float32, copy=True)

        filename, chunk_id = self.parse_rgb_filename(self.rgbs[index])
        return rgb, label, nir, filename, chunk_id

    def _format_clip(self, clip):
        if self.data_format == "NDCHW":
            clip = np.transpose(clip, (0, 3, 1, 2))
        elif self.data_format == "NCDHW":
            clip = np.transpose(clip, (3, 0, 1, 2))
        elif self.data_format == "NDHWC":
            pass
        else:
            raise ValueError("Unsupported Data Format!")
        return np.array(clip, dtype=np.float32, copy=True)

    def get_raw_data(self, raw_data_path):
        """获取所有可用的 subject/trial 目录。"""
        subject_dirs = sorted(
            d for d in glob.glob(raw_data_path + os.sep + "*")
            if os.path.isdir(d) and not os.path.basename(d).startswith("._")
        )
        if not subject_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")

        dirs = []
        for subject_dir in subject_dirs:
            subject = os.path.basename(subject_dir)
            trial_dirs = sorted(
                d for d in glob.glob(subject_dir + os.sep + "*")
                if os.path.isdir(d) and not os.path.basename(d).startswith("._")
            )
            for trial_dir in trial_dirs:
                trial = os.path.basename(trial_dir)
                rgb_path = os.path.join(trial_dir, "Video", "RGB.mp4")
                nir_path = os.path.join(trial_dir, "Video", "IR.mp4")
                bvp_path = os.path.join(trial_dir, "Label", "BVP.mat")
                if os.path.exists(rgb_path) and os.path.exists(nir_path) and os.path.exists(bvp_path):
                    dirs.append({
                        "index": "{}_{}".format(subject, trial),
                        "path": trial_dir,
                        "subject": subject,
                        "trial": trial,
                    })
        if not dirs:
            raise ValueError(self.dataset_name + " valid trial paths empty!")
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """按照 BEGIN/END 在 subject 维度切分数据，避免 subject 泄漏，如下：
            1.先把所有 trial 按 subject 分组， 例如 AMW3/AS、AMW3/AT、AMW3/BS 都属于 subject AMW3。
            2.对所有 subject 排序
            3.根据 YAML 里的 BEGIN/END 切 subject 列表 """
        if begin == 0 and end == 1:
            return data_dirs

        data_info = {}
        for data in data_dirs:
            data_info.setdefault(data["subject"], []).append(data)
        subj_list = sorted(data_info.keys())
        subj_range = list(range(int(begin * len(subj_list)), int(end * len(subj_list))))
        print("used PhysDrive subject ids for split:", [subj_list[i] for i in subj_range])

        data_dirs_new = []
        for i in subj_range:
            data_dirs_new += data_info[subj_list[i]]
        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """供 BaseLoader.preprocess_dataset 调用的多进程预处理子任务。"""
        trial_dir = data_dirs[i]["path"]
        saved_filename = data_dirs[i]["index"]

        rgb, nir = self.read_video_pair_resized(
            os.path.join(trial_dir, "Video", "RGB.mp4"),
            os.path.join(trial_dir, "Video", "IR.mp4"),
            config_preprocess,
        )
        bvp = self.read_bvp(os.path.join(trial_dir, "Label", "BVP.mat"))

        if rgb.shape[0] <= 0:
            raise ValueError("PhysDrive trial has no video frames: {}".format(trial_dir))
        bvp = BaseLoader.resample_ppg(bvp, rgb.shape[0])

        rgb_clips, nir_clips, bvp_clips = self.transform_label_and_chunk(rgb, nir, bvp, config_preprocess)
        input_name_list, _, _ = self.save_multimodal_process(rgb_clips, nir_clips, bvp_clips, saved_filename)
        file_list_dict[i] = input_name_list

    def preprocess_multimodal(self, rgb_frames, nir_frames, bvps, config_preprocess):
        """对已读入内存的 RGB/NIR/BVP 做同步裁剪、标签处理和切片。"""
        rgb_frames, nir_frames = self.crop_face_resize_pair(
            rgb_frames,
            nir_frames,
            config_preprocess.CROP_FACE.DO_CROP_FACE,
            config_preprocess.CROP_FACE.USE_LARGE_FACE_BOX,
            config_preprocess.CROP_FACE.LARGE_BOX_COEF,
            config_preprocess.CROP_FACE.DETECTION.DO_DYNAMIC_DETECTION,
            config_preprocess.CROP_FACE.DETECTION.DYNAMIC_DETECTION_FREQUENCY,
            config_preprocess.CROP_FACE.DETECTION.USE_MEDIAN_FACE_BOX,
            config_preprocess.RESIZE.W,
            config_preprocess.RESIZE.H,
        )
        return self.transform_label_and_chunk(rgb_frames, nir_frames, bvps, config_preprocess)

    def transform_label_and_chunk(self, rgb_frames, nir_frames, bvps, config_preprocess):
        """根据配置转换 RGB/NIR 数据和 BVP 标签，并按 CHUNK_LENGTH 切成 clip。"""
        rgb_data = self.apply_data_transforms(rgb_frames, config_preprocess.DATA_TYPE)
        nir_data = self.apply_data_transforms(nir_frames, config_preprocess.DATA_TYPE)

        if config_preprocess.LABEL_TYPE == "Raw":
            pass
        elif config_preprocess.LABEL_TYPE == "DiffNormalized":
            bvps = BaseLoader.diff_normalize_label(bvps)
        elif config_preprocess.LABEL_TYPE == "Standardized":
            bvps = BaseLoader.standardized_label(bvps)
        else:
            raise ValueError("Unsupported label type!")

        if config_preprocess.DO_CHUNK:
            rgb_clips, bvp_clips = self.chunk(rgb_data, bvps, config_preprocess.CHUNK_LENGTH)
            nir_clips, _ = self.chunk(nir_data, bvps, config_preprocess.CHUNK_LENGTH)
            rgb_clips, nir_clips, bvp_clips = self.filter_flat_label_clips(rgb_clips, nir_clips, bvp_clips)
            return rgb_clips, nir_clips, bvp_clips
        return np.array([rgb_data]), np.array([nir_data]), np.array([bvps])

    def filter_flat_label_clips(self, rgb_clips, nir_clips, bvp_clips):
        """过滤 BVP 标准差过小的 clip，并同步移除对应 RGB/NIR clip。"""
        if len(bvp_clips) == 0:
            return rgb_clips, nir_clips, bvp_clips

        keep_mask = np.array([np.std(bvp_clip) > self.MIN_LABEL_STD for bvp_clip in bvp_clips])
        removed_count = int(np.sum(~keep_mask))
        if removed_count > 0:
            print(
                "Filtered {} PhysDrive clips with near-constant BVP labels.".format(removed_count)
            )
        return rgb_clips[keep_mask], nir_clips[keep_mask], bvp_clips[keep_mask]

    def multi_process_manager(self, data_dirs, config_preprocess, multi_process_quota=2):
        """限制 PhysDrive 预处理并行数，避免高分辨率 RGB/NIR 视频占用过多内存。"""
        return super().multi_process_manager(data_dirs, config_preprocess, multi_process_quota)

    def crop_face_resize_pair(
            self,
            rgb_frames,
            nir_frames,
            use_face_detection,
            use_larger_box,
            larger_box_coef,
            use_dynamic_detection,
            detection_freq,
            use_median_box,
            width,
            height):
        """使用 RGB 上的人脸框同步裁剪 RGB/NIR，然后 resize。"""
        if rgb_frames.shape[0] != nir_frames.shape[0]:
            raise ValueError("RGB and NIR frame counts must match before preprocessing.")

        if use_dynamic_detection:
            num_dynamic_det = int(np.ceil(rgb_frames.shape[0] / detection_freq))
        else:
            num_dynamic_det = 1

        face_region_all = []
        for idx in range(num_dynamic_det):
            frame_idx = min(detection_freq * idx, rgb_frames.shape[0] - 1)
            if use_face_detection:
                face_region_all.append(self.face_detection(rgb_frames[frame_idx], use_larger_box, larger_box_coef))
            else:
                face_region_all.append([0, 0, rgb_frames.shape[2], rgb_frames.shape[1]])
        face_region_all = np.asarray(face_region_all, dtype="int")
        face_region_median = np.median(face_region_all, axis=0).astype("int") if use_median_box else None

        resized_rgb = np.zeros((rgb_frames.shape[0], height, width, rgb_frames.shape[-1]), dtype=np.float32)
        resized_nir = np.zeros((nir_frames.shape[0], height, width, nir_frames.shape[-1]), dtype=np.float32)

        for frame_idx in range(rgb_frames.shape[0]):
            reference_index = frame_idx // detection_freq if use_dynamic_detection else 0
            face_region = face_region_median if use_median_box else face_region_all[reference_index]
            rgb_frame = rgb_frames[frame_idx]
            nir_frame = nir_frames[frame_idx]

            if use_face_detection:
                x, y, w, h = face_region
                y0, y1 = max(y, 0), min(y + h, rgb_frame.shape[0])
                x0, x1 = max(x, 0), min(x + w, rgb_frame.shape[1])
                if y1 > y0 and x1 > x0:
                    rgb_frame = rgb_frame[y0:y1, x0:x1]
                    x_scale = nir_frame.shape[1] / rgb_frames[frame_idx].shape[1]
                    y_scale = nir_frame.shape[0] / rgb_frames[frame_idx].shape[0]
                    nir_x0 = max(int(round(x0 * x_scale)), 0)
                    nir_x1 = min(int(round(x1 * x_scale)), nir_frame.shape[1])
                    nir_y0 = max(int(round(y0 * y_scale)), 0)
                    nir_y1 = min(int(round(y1 * y_scale)), nir_frame.shape[0])
                    if nir_y1 > nir_y0 and nir_x1 > nir_x0:
                        nir_frame = nir_frame[nir_y0:nir_y1, nir_x0:nir_x1]

            resized_rgb[frame_idx] = cv2.resize(rgb_frame, (width, height), interpolation=cv2.INTER_AREA)
            nir_resized = cv2.resize(nir_frame, (width, height), interpolation=cv2.INTER_AREA)
            if nir_resized.ndim == 2:
                nir_resized = np.expand_dims(nir_resized, axis=-1)
            resized_nir[frame_idx] = nir_resized

        return resized_rgb, resized_nir

    def face_detection(self, frame, use_larger_box=False, larger_box_coef=1.0):
        """在 RGB 帧上检测最大人脸框，并按配置放大裁剪区域。"""
        detector = cv2.CascadeClassifier("./dataset/haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) if frame.ndim == 3 else frame
        face_zone = detector.detectMultiScale(gray)
        if len(face_zone) < 1:
            print("ERROR: No Face Detected")
            face_box_coor = [0, 0, frame.shape[1], frame.shape[0]]
        else:
            areas = face_zone[:, 2] * face_zone[:, 3]
            face_box_coor = face_zone[np.argmax(areas)].astype(np.float32)
            if len(face_zone) > 1:
                print("Warning: More than one faces are detected(Only cropping the biggest one.)")

        if use_larger_box:
            face_box_coor[0] = max(0, face_box_coor[0] - (larger_box_coef - 1.0) / 2 * face_box_coor[2])
            face_box_coor[1] = max(0, face_box_coor[1] - (larger_box_coef - 1.0) / 2 * face_box_coor[3])
            face_box_coor[2] = larger_box_coef * face_box_coor[2]
            face_box_coor[3] = larger_box_coef * face_box_coor[3]
        return face_box_coor

    @staticmethod
    def apply_data_transforms(frames, data_types):
        """复用 BaseLoader 的数据变换方式处理单个视频模态。"""
        if len(data_types) == 1 and data_types[0] == "Raw":
            return frames
        data = []
        for data_type in data_types:
            f_c = frames.copy()
            if data_type == "Raw":
                data.append(f_c)
            elif data_type == "DiffNormalized":
                data.append(BaseLoader.diff_normalize_data(f_c))
            elif data_type == "Standardized":
                data.append(BaseLoader.standardized_data(f_c))
            else:
                raise ValueError("Unsupported data type!")
        return np.concatenate(data, axis=-1)

    def save_multimodal_process(self, rgb_clips, nir_clips, bvp_clips, filename):
        """保存 RGB、NIR 和 BVP clip，并返回 RGB 文件列表给 BaseLoader 建表。"""
        if not os.path.exists(self.cached_path):
            os.makedirs(self.cached_path, exist_ok=True)
        rgb_path_name_list = []
        nir_path_name_list = []
        label_path_name_list = []
        for count in range(len(bvp_clips)):
            rgb_path_name = self.cached_path + os.sep + "{0}_rgb{1}.npy".format(filename, str(count))
            nir_path_name = self.cached_path + os.sep + "{0}_nir{1}.npy".format(filename, str(count))
            label_path_name = self.cached_path + os.sep + "{0}_label{1}.npy".format(filename, str(count))
            rgb_path_name_list.append(rgb_path_name)
            nir_path_name_list.append(nir_path_name)
            label_path_name_list.append(label_path_name)
            np.save(rgb_path_name, rgb_clips[count].astype(np.float32))
            np.save(nir_path_name, nir_clips[count].astype(np.float32))
            np.save(label_path_name, bvp_clips[count].astype(np.float32))
        return rgb_path_name_list, nir_path_name_list, label_path_name_list

    def load_preprocessed_data(self):
        """从 file list 读取 RGB 文件，并推导配套的 NIR 和 BVP label 路径。"""
        file_list_df = pd.read_csv(self.file_list_path)
        inputs = sorted(file_list_df["input_files"].tolist())
        if not inputs:
            raise ValueError(self.dataset_name + " dataset loading data error!")
        labels = [self.replace_rgb_suffix(input_file, "label") for input_file in inputs]
        nirs = [self.replace_rgb_suffix(input_file, "nir") for input_file in inputs]

        missing = [path for path in labels + nirs if not os.path.exists(path)]
        if missing:
            raise ValueError("Missing PhysDrive preprocessed pair file: {}".format(missing[0]))

        self.rgbs = inputs
        # BaseLoader.__len__ 使用 self.inputs，这里保留同一份 RGB 路径列表以兼容父类。
        self.inputs = self.rgbs
        self.labels = labels
        self.nirs = nirs
        self.preprocessed_data_len = len(inputs)

    def build_file_list_retroactive(self, data_dirs, begin, end):
        """根据已保存的 RGB 缓存文件重建 file list。"""
        data_dirs_subset = self.split_raw_data(data_dirs, begin, end)
        filename_list = list({data["index"] for data in data_dirs_subset})
        file_list = []
        for fname in filename_list:
            file_list += list(glob.glob(self.cached_path + os.sep + "{0}_rgb*.npy".format(fname)))

        if not file_list:
            raise ValueError(
                self.dataset_name,
                "File list empty. Check preprocessed PhysDrive data folder exists and is not empty.",
            )

        file_list_df = pd.DataFrame(file_list, columns=["input_files"])
        os.makedirs(os.path.dirname(self.file_list_path), exist_ok=True)
        file_list_df.to_csv(self.file_list_path)

    @staticmethod
    def parse_rgb_filename(rgb_path):
        """从 RGB 缓存路径中解析样本名和 chunk id。"""
        stem = os.path.splitext(os.path.basename(rgb_path))[0]
        if "_rgb" in stem:
            filename, chunk_id = stem.rsplit("_rgb", 1)
            return filename, chunk_id
        raise ValueError("Unsupported PhysDrive RGB cache filename: {}".format(rgb_path))

    @staticmethod
    def replace_rgb_suffix(rgb_path, target_suffix):
        """把 RGB 缓存路径替换成对应 NIR 或 label 路径。"""
        if "_rgb" in os.path.splitext(os.path.basename(rgb_path))[0]:
            prefix, tail = rgb_path.rsplit("_rgb", 1)
            return "{}_{}{}".format(prefix, target_suffix, tail)
        raise ValueError("Unsupported PhysDrive RGB cache filename: {}".format(rgb_path))

    def read_video_pair_resized(self, rgb_file, nir_file, config_preprocess):
        """同步读取 RGB/NIR 视频，并在保存帧前完成裁剪和 resize。"""
        rgb_cap = cv2.VideoCapture(rgb_file)
        nir_cap = cv2.VideoCapture(nir_file)
        if not rgb_cap.isOpened():
            raise ValueError("Could not open PhysDrive RGB video: {}".format(rgb_file))
        if not nir_cap.isOpened():
            raise ValueError("Could not open PhysDrive NIR video: {}".format(nir_file))

        crop_cfg = config_preprocess.CROP_FACE
        detect_cfg = crop_cfg.DETECTION
        width = config_preprocess.RESIZE.W
        height = config_preprocess.RESIZE.H
        use_face_detection = crop_cfg.DO_CROP_FACE
        detection_freq = detect_cfg.DYNAMIC_DETECTION_FREQUENCY
        use_dynamic_detection = detect_cfg.DO_DYNAMIC_DETECTION
        use_median_box = detect_cfg.USE_MEDIAN_FACE_BOX

        median_box = None
        if use_face_detection and use_median_box:
            median_box = self.collect_median_face_region(
                rgb_file,
                use_dynamic_detection,
                detection_freq,
                crop_cfg.USE_LARGE_FACE_BOX,
                crop_cfg.LARGE_BOX_COEF,
            )

        rgb_frames = []
        nir_frames = []
        face_region = None
        frame_idx = 0

        while True:
            rgb_success, rgb_frame = rgb_cap.read()
            nir_success, nir_frame = nir_cap.read()
            if not rgb_success or not nir_success:
                break

            rgb_frame = cv2.cvtColor(np.asarray(rgb_frame), cv2.COLOR_BGR2RGB)
            nir_frame = cv2.cvtColor(np.asarray(nir_frame), cv2.COLOR_BGR2GRAY)
            nir_frame = np.expand_dims(nir_frame, axis=-1)

            if use_face_detection:
                if median_box is not None:
                    face_region = median_box
                elif face_region is None or (use_dynamic_detection and frame_idx % detection_freq == 0):
                    face_region = self.face_detection(
                        rgb_frame,
                        crop_cfg.USE_LARGE_FACE_BOX,
                        crop_cfg.LARGE_BOX_COEF,
                    )

                rgb_frame, nir_frame = self.crop_pair_with_rgb_region(rgb_frame, nir_frame, face_region)

            rgb_frames.append(cv2.resize(rgb_frame, (width, height), interpolation=cv2.INTER_AREA))
            nir_resized = cv2.resize(nir_frame, (width, height), interpolation=cv2.INTER_AREA)
            if nir_resized.ndim == 2:
                nir_resized = np.expand_dims(nir_resized, axis=-1)
            nir_frames.append(nir_resized)
            frame_idx += 1

        rgb_cap.release()
        nir_cap.release()
        return np.asarray(rgb_frames), np.asarray(nir_frames)

    def collect_median_face_region(
            self,
            rgb_file,
            use_dynamic_detection,
            detection_freq,
            use_larger_box,
            larger_box_coef):
        """收集多次 RGB 人脸检测框，并返回中位数框。"""
        rgb_cap = cv2.VideoCapture(rgb_file)
        if not rgb_cap.isOpened():
            raise ValueError("Could not open PhysDrive RGB video: {}".format(rgb_file))
        face_regions = []
        frame_idx = 0
        while True:
            success, frame = rgb_cap.read()
            if not success:
                break
            if frame_idx == 0 or use_dynamic_detection and frame_idx % detection_freq == 0:
                frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB)
                face_regions.append(self.face_detection(frame, use_larger_box, larger_box_coef))
            frame_idx += 1
        rgb_cap.release()
        if not face_regions:
            return None
        return np.median(np.asarray(face_regions, dtype="int"), axis=0).astype("int")

    @staticmethod
    def crop_pair_with_rgb_region(rgb_frame, nir_frame, face_region):
        """用 RGB 人脸框裁剪 RGB，并按分辨率比例映射后裁剪 NIR。"""
        x, y, w, h = np.asarray(face_region, dtype="int")
        y0, y1 = max(y, 0), min(y + h, rgb_frame.shape[0])
        x0, x1 = max(x, 0), min(x + w, rgb_frame.shape[1])
        if y1 <= y0 or x1 <= x0:
            return rgb_frame, nir_frame

        rgb_cropped = rgb_frame[y0:y1, x0:x1]
        x_scale = nir_frame.shape[1] / rgb_frame.shape[1]
        y_scale = nir_frame.shape[0] / rgb_frame.shape[0]
        nir_x0 = max(int(round(x0 * x_scale)), 0)
        nir_x1 = min(int(round(x1 * x_scale)), nir_frame.shape[1])
        nir_y0 = max(int(round(y0 * y_scale)), 0)
        nir_y1 = min(int(round(y1 * y_scale)), nir_frame.shape[0])
        if nir_y1 <= nir_y0 or nir_x1 <= nir_x0:
            return rgb_cropped, nir_frame
        return rgb_cropped, nir_frame[nir_y0:nir_y1, nir_x0:nir_x1]

    @staticmethod
    def read_rgb_video(video_file):
        """读取 RGB 视频为 T,H,W,3。"""
        vid_obj = cv2.VideoCapture(video_file)
        vid_obj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = vid_obj.read()
        frames = []
        while success:
            frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB)
            frames.append(frame)
            success, frame = vid_obj.read()
        vid_obj.release()
        return np.asarray(frames)

    @staticmethod
    def read_nir_video(video_file):
        """读取 NIR 视频为单通道 T,H,W,1。"""
        vid_obj = cv2.VideoCapture(video_file)
        vid_obj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = vid_obj.read()
        frames = []
        while success:
            gray = cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2GRAY)
            frames.append(np.expand_dims(gray, axis=-1))
            success, frame = vid_obj.read()
        vid_obj.release()
        return np.asarray(frames)

    @staticmethod
    def read_bvp(bvp_file):
        """读取 PhysDrive 的 BVP.mat 并返回一维 BVP 信号。"""
        mat = sio.loadmat(bvp_file)
        if "BVP" not in mat:
            keys = [key for key in mat.keys() if not key.startswith("__")]
            raise ValueError("BVP variable not found in {}. Available keys: {}".format(bvp_file, keys))
        return np.asarray(mat["BVP"]).reshape(-1)
