import os
import pickle
import cv2
import numpy as np 
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.rf import organizer as org
from dataset.rf.proc import create_fast_slow_matrix, find_range
# import scipy.signal as sig
# from scipy.io import loadmat
# import h5py

def standardize_ppg(data):
    data = np.array(data)
    data_mean = np.mean(data)
    data_std = np.std(data)
    return (data - data_mean) / data_std


def standardize_rgb(video, mean, std):
    """Standardize an RGB clip with its complete-trial channel statistics."""
    video = np.asarray(video, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    return (video - mean) / std


def read_rgb_image(image_path):
    """Read one PNG as RGB without imageio's long-worker memory growth."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'Could not read RGB image: {image_path}')
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def flatten_iq_window(rf_data):
    """Flatten RF I/Q range-window data from (2, window, T) to (2 * window, T)."""
    assert rf_data.ndim == 3 and rf_data.shape[0] == 2, (
        f'Expected RF data with shape (2, window_size, T), but got {rf_data.shape}'
    )
    return np.transpose(rf_data, axes=(1, 0, 2)).reshape(-1, rf_data.shape[-1])


def read_split_file(split_file, split_name):
    """Reads Equipleth split ids from a txt/csv file or fold pkl."""
    if not split_file:
        raise ValueError("Equipleth requires DATA.SPLIT_FILE to list subjects/sessions.")
    if not os.path.exists(split_file):
        raise ValueError("Equipleth split file does not exist: {}".format(split_file))

    _, ext = os.path.splitext(split_file)
    if ext == ".pkl":
        with open(split_file, "rb") as f:
            fold_data = pickle.load(f)
        if isinstance(fold_data, list):
            fold_data = fold_data[0]
        if not isinstance(fold_data, dict):
            raise ValueError("Equipleth pkl split must contain a dict or a list of dicts.")

        split_key = "val" if split_name == "valid" else split_name
        if split_key not in fold_data:
            raise ValueError("Equipleth pkl split missing key: {}".format(split_key))
        split = fold_data[split_key]
        if not split:
            raise ValueError("Equipleth pkl split is empty for key: {}".format(split_key))
        return split

    split = []
    with open(split_file, "r") as f:
        for line in f:
            item = line.strip()
            if item and not item.startswith("#"):
                split.append(item.split(",")[0].strip())
    if not split:
        raise ValueError("Equipleth split file is empty: {}".format(split_file))
    return split


class EquiplethDataset(Dataset):
    def __init__(self, name=None, data_path=None, config_data=None, split=None, frame_length = 128, fs_ppg=30, video_length = 900, \
                 fs_rf = 120, window_size = 5, freq_slope=60.012e12, sample_rate_rf=5e6, chirp_samples=256, ppg_offset=25,
                 num_samples=30, data_format='NCDHW', use_rf=True):
        rgb_data_type = 'Raw'
        label_type = 'Standardized'
        force_recompute_rgb_stats = False
        if config_data is not None:
            split = read_split_file(config_data.SPLIT_FILE, name)
            frame_length = config_data.PREPROCESS.CHUNK_LENGTH
            fs_ppg = config_data.FS if config_data.FS else fs_ppg
            ppg_offset = config_data.PREPROCESS.PPG_OFFSET
            data_format = config_data.DATA_FORMAT
            use_rf = config_data.USE_RF
            if name == "train":
                num_samples = config_data.PREPROCESS.NUM_SAMPLES
            force_recompute_rgb_stats = config_data.DO_PREPROCESS
            rgb_data_types = config_data.PREPROCESS.DATA_TYPE
            if len(rgb_data_types) != 1:
                raise ValueError('Equipleth PREPROCESS.DATA_TYPE must contain one value.')
            rgb_data_type = rgb_data_types[0]
            label_type = config_data.PREPROCESS.LABEL_TYPE
            supported_types = {'Raw', 'Standardized'}
            if rgb_data_type not in supported_types or label_type not in supported_types:
                raise ValueError('Equipleth currently supports only Raw and Standardized.')
        if data_path is None:
            raise ValueError("EquiplethDataset requires data_path.")
        if split is None:
            raise ValueError("EquiplethDataset requires split ids.")

        self.data_path = data_path
        self.name = name
        self.split = split
        self.rgb_data_type = rgb_data_type
        self.label_type = label_type
        self.data_format = data_format
        self.use_rf = use_rf
        self.force_recompute_rgb_stats = force_recompute_rgb_stats

        # There is an offset in capturing the signals w.r.t the ground truth.
        self.ppg_offset = ppg_offset

        # number of frame for input of model
        self.frame_length = frame_length
        self.video_length = video_length
        # 每个 subject/trial 在训练阶段随机抽取的 clip 数量。
        self.num_samples = num_samples
        
        self.rgb_path = os.path.join(data_path, 'rgb_files')
        self.rf_path = os.path.join(data_path, 'rf_files')

        # Standardized RGB uses one per-channel mean/std pair for the complete
        # trial. Only the statistics are retained; RGB clips remain lazily read.
        self.rgb_stats = None
        if self.rgb_data_type == 'Standardized':
            self.rgb_stats = self._load_or_compute_rgb_stats()


        # ppg configs
        self.ppg_list = []

        # RF configs
        self.window_size = window_size
        self.fs_rf = fs_rf
        self.freq_slope = freq_slope
        self.sample_rate_rf = sample_rate_rf
        self.sample_ratio = int(fs_rf / fs_ppg)  # fs along slow time axis
        self.chirp_samples = chirp_samples

        

        # load ppgs
        for subject in self.split:

            file_path = os.path.join(self.rgb_path, subject)

            ppg = np.load(os.path.join(file_path, 'rgbd_ppg.npy'))
            if self.label_type == 'Standardized':
                ppg = standardize_ppg(ppg)
            self.ppg_list.append(ppg)


        # Complete-trial PPG signals, optionally standardized before clip sampling.
        self.ppgs = self.ppg_list


        # load RFs and create fast-slow matrix
        self.rfs = []
        rf_subjects = self.split if self.use_rf else []
        for subject in rf_subjects:

            subject_rf = subject[2:]

            file_path = os.path.join(self.rf_path, subject_rf)

            rf_fptr = open(os.path.join(file_path, 'rf.pkl'), 'rb')
            s = pickle.load(rf_fptr)

            # Organize the raw data from the RF.
            # Number of samples is set ot 256 for our experiments.
            rf_organizer = org.Organizer(s, 1, 1, 1, 2*self.chirp_samples)
            frames = rf_organizer.organize()

            # The RF read adds zero alternatively to the samples. Remove these zeros.
            frames = frames[:,:,:,0::2]

            # Process the organized RF data
            data_f = create_fast_slow_matrix(frames)
            range_index = find_range(data_f, self.sample_rate_rf, self.freq_slope, self.chirp_samples)
            # Get the windowed raw data for the network
            raw_data = data_f[:, range_index-self.window_size//2:range_index+self.window_size//2 + 1]  #(rf_length:512, window_size:25cm)
            # Note that item is a complex number due to the nature of the algorithm used to extract and process the pickle file.
            # Hence for simplicity we separate the real and imaginary parts into 2 separate channels.
            raw_data = np.array([np.real(raw_data),  np.imag(raw_data)]) # (2, rf_length, window_size)
            raw_data = np.transpose(raw_data, axes=(0,2,1))  # (2, window_size, rf_length)

            self.rfs.append(raw_data)
        
        # all possible sampling combinations.
        self.all_idxs = []
        max_start = self.video_length - self.frame_length - self.ppg_offset
        if max_start < 0:
            raise ValueError(
                f'Frame length {self.frame_length} is longer than available video length '
                f'{self.video_length} after PPG offset {self.ppg_offset}.'
            )

        for i in range(len(self.split)):
            if self.name == "train":
                # 训练阶段随机采样 clip 起点，用于增加数据多样性。
                cur_frames = np.random.randint(low=0, high=max_start + 1, size=self.num_samples)
            else:
                # 验证/测试阶段使用确定性的顺序 clip 起点。
                # 如果常规步长没有覆盖到 max_start，则补上 max_start，
                # 保证最后一段有效的 RGB/RF 片段和偏移后的 PPG 标签被覆盖。
                cur_frames = np.arange(0, max_start + 1, self.frame_length)
                if len(cur_frames) == 0:
                    cur_frames = np.array([0])
                elif cur_frames[-1] != max_start:
                    cur_frames = np.append(cur_frames, max_start)

            rf_cur_frames = cur_frames * self.sample_ratio
            for chunk_id, (rf_cur_frame, cur_frame) in enumerate(zip(rf_cur_frames, cur_frames)):
                self.all_idxs.append((i, int(cur_frame), int(rf_cur_frame), chunk_id))

    def _load_or_compute_rgb_stats(self):
        """读取缓存，并按需计算当前 split 中各 trial 的 RGB 统计量。"""
        cache_path = os.path.join(self.rgb_path, '.equipleth_rgb_trial_stats.pkl')
        cached_subjects = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as cache_file:
                    cached_subjects = pickle.load(cache_file)
            except (OSError, EOFError, pickle.UnpicklingError) as error:
                print(f'Warning: could not read Equipleth RGB statistics cache: {error}')

        if self.force_recompute_rgb_stats:
            subjects_to_compute = list(self.split)
        else:
            subjects_to_compute = [
                subject for subject in self.split if subject not in cached_subjects
            ]

        if subjects_to_compute:
            action = 'Recomputing' if self.force_recompute_rgb_stats else 'Computing'
            for subject in tqdm(
                subjects_to_compute,
                desc=f'{action} {self.name} RGB trial statistics',
                ncols=80,
            ):
                cached_subjects[subject] = self._compute_rgb_trial_stats(subject)

            temporary_path = f'{cache_path}.{os.getpid()}.tmp'
            try:
                with open(temporary_path, 'wb') as cache_file:
                    pickle.dump(cached_subjects, cache_file)
                os.replace(temporary_path, cache_path)
            except OSError as error:
                print(f'Warning: could not save Equipleth RGB statistics cache: {error}')

        return [cached_subjects[subject] for subject in self.split]

    def _compute_rgb_trial_stats(self, subject):
        """计算单个 trial 有效帧的逐通道 RGB 均值和标准差。"""
        channel_sum = np.zeros(3, dtype=np.float64)
        channel_squared_sum = np.zeros(3, dtype=np.float64)
        pixel_count = 0

        for frame_index in range(self.video_length - self.ppg_offset):
            image_path = os.path.join(
                self.rgb_path, subject, f'rgbd_rgb_{frame_index}.png'
            )
            image = read_rgb_image(image_path)
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(
                    f'Expected RGB image at {image_path}, got shape {image.shape}.'
                )
            image = image.astype(np.float64, copy=False)
            channel_sum += image.sum(axis=(0, 1))
            channel_squared_sum += np.square(image).sum(axis=(0, 1))
            pixel_count += image.shape[0] * image.shape[1]

        mean = channel_sum / pixel_count
        variance = channel_squared_sum / pixel_count - np.square(mean)
        std = np.sqrt(variance)
        return {'mean': mean.astype(np.float32), 'std': std.astype(np.float32)}


    def __len__(self):

        return len(self.all_idxs)
    

    def __getitem__(self, idx):


        # video and ppg are synchronized
        index, frame_start, rf_start, chunk_id = self.all_idxs[idx]
        

        # load video
        video = []

        for img_idx in range(self.frame_length):
            image_path = os.path.join(self.rgb_path, 
                                      str(self.split[index]),
                                      f'rgbd_rgb_{frame_start+img_idx}.png'
                                      )
            
            video.append(read_rgb_image(image_path))
        video = np.array(video)

        if(len(video.shape)<4):
            video = np.expand_dims(video, axis=3)

        if self.rgb_data_type == 'Standardized':
            stats = self.rgb_stats[index]
            video = standardize_rgb(video, stats['mean'], stats['std'])

        if self.data_format == 'NDCHW':
            video = np.transpose(video, axes=(0, 3, 1, 2))
        elif self.data_format == 'NCDHW':
            video = np.transpose(video, axes=(3, 0, 1, 2))
        else:
            raise ValueError(f'Unsupported Equipleth data format: {self.data_format}')

        # RGB/RF frames are aligned to PPG samples after the dataset-specific label offset.
        ppg_start = frame_start + self.ppg_offset
        ppg = self.ppgs[index][ppg_start:ppg_start+self.frame_length]


        if video.dtype == np.uint16:
            video = video.astype(np.int32)

        assert self.frame_length == len(ppg), f'Expected signal of length {self.frame_length}, but got signal of length {len(ppg)}'

        if not self.use_rf:
            return (
                np.array(video),
                np.array(ppg),
                self.split[index],
                chunk_id,
            )

        rf = self.rfs[index][:,:,rf_start:rf_start+self.sample_ratio*self.frame_length]
        rf = flatten_iq_window(rf)

        return (
            np.array(video),
            np.array(ppg),
            np.array(rf).astype('float32'),
            self.split[index],
            chunk_id,
            frame_start,
        )
    

# NOTE: ZJU loader code is intentionally kept commented for future reuse.
# The current active loader in this file is only EquiplethDataset.
# class MultiModalityDatasetUCLA_ZJU(Dataset):
#     def __init__(self, data_path, split, frame_length = 300, fs_ppg=30, video_length = 5100, \
#                  fs_rf = 120, window_size = 5, freq_slope=60.012e12, sample_rate_rf=6e6, chirp_samples=256):
        
#         self.data_path = data_path
#         self.split = split

#         # There is an offset in capturing the signals w.r.t the ground truth.
#         self.ppg_offset = 0

#         # number of frame for input of model
#         self.frame_length = frame_length
#         self.video_length = video_length
#         # number of samples in one trial.
#         self.num_samples = 30
        
#         self.rgb_path = os.path.join(data_path, 'rgb_h5_0')
#         self.rf_path = os.path.join(data_path, 'rf_files_refined')
#         self.ppg_path = os.path.join(data_path, 'ppg')
        


#         # ppg configs
#         self.ppg_list = []

#         # RF configs
#         self.window_size = window_size
#         self.fs_rf = fs_rf
#         self.freq_slope = freq_slope
#         self.sample_rate_rf = sample_rate_rf
#         self.sample_ratio = int(fs_rf / fs_ppg)  # fs along slow time axis
  

        

#         # load ppgs
#         for file in self.split:
#             "subject: 45_2"
#             subject = file.split('_')[0]

#             ppg = np.load(os.path.join(self.ppg_path, subject, file+'.npy'))
#             self.ppg_list.append(standardize_ppg(ppg))


#         # normalize PPGs
#         self.ppgs = self.ppg_list


#         # load RFs and create fast-slow matrix
#         self.rfs = []
#         self.all_idxs = []

#         index = 0 # rf mat file index
#         for i in range(len(self.split)):
#             subject = self.split[i]
#             file_path = os.path.join(self.rf_path, subject)

#             time_stamp = np.load(os.path.join(file_path, 'rf_time.npy'))


#             for idx, mat_file in enumerate(sorted(os.listdir(file_path))[:-1]):
#                 cur_time_str = time_stamp[idx]
#                 h, m, s = cur_time_str.strip().split(':')
#                 cur_time = int(h)*3600 + int(m)*60 + int(s)

#                 cur_frame = cur_time * 30

#                 raw_data = loadmat(os.path.join(file_path, mat_file))['window_profile'][:] #(window_size, T)
#                 raw_data = raw_data[:, ::85]
            
#                 raw_data = np.array([np.real(raw_data),  np.imag(raw_data)]) # (2, window_size, T)
                
#             # raw_data = np.transpose(raw_data, axes=(0,2,1))  # (2, window_size, rf_length)
#                 # cur_rf_frame = cur_frame * self.sample_ratio
#                 self.all_idxs.append(((index, i), (0, cur_frame)))

#                 self.rfs.append(raw_data)

#                 index += 1
        
        
#     def __len__(self):

#         return len(self.all_idxs)
    

#     def __getitem__(self, idx):


#         # video and ppg are synchronized
#         (rf_index, frame_index),  (rf_start, frame_start) = self.all_idxs[idx]
        

#         # load video
 

#         with h5py.File(os.path.join(self.rgb_path, str(self.split[frame_index])+'.h5'), 'r') as f:
#             video = f['imgs'][frame_start:frame_start+self.frame_length]

#         # for img_idx in range(self.frame_length):
#         #     image_path = os.path.join(self.rgb_path, 
#         #                               str(self.split[index]),
#         #                               f'rgbd_rgb_{frame_start+img_idx}.png'
#         #                               )
            
#         #     video.append(imageio.imread(image_path))
#         video = np.array(video)

#         if(len(video.shape)<4):
#             video = np.expand_dims(video, axis=3)

#         # transpose to (C, T, H, W)
#         video = np.transpose(video, axes=(3,0,1,2))

#         # video = self.videos[index][frame_start:frame_start+self.frame_length]
#         ppg = self.ppgs[frame_index][frame_start:frame_start+self.frame_length]
#         rf = self.rfs[rf_index][:,:,rf_start:rf_start+self.sample_ratio*self.frame_length]
#         rf = flatten_iq_window(rf)


#         if video.dtype == np.uint16:
#             video = video.astype(np.int32)

#         assert self.frame_length == len(ppg), f'Expected signal of length {self.frame_length}, but got signal of length {len(ppg)}'

#         if video.shape[1] < self.frame_length:
#             print(self.split[frame_index])
#         return np.array(video), np.array(ppg), np.array(rf).astype('float32')
