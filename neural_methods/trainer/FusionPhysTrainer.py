"""Trainer for FusionPhys."""

import os
import re
import time

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.FusionPhys import FusionPhys
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm


class FusionPhysTrainer(BaseTrainer):
    SUPPORTED_MODALITY_MODES = (
        "camera_only",
        "camera_camera_fusion",
        "camera_rf_fusion",
    )

    def __init__(self, config, data_loader):
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.MODEL.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.config = config
        self.modality_mode = config.MODEL.FUSIONPHYS.MODALITY_MODE
        self.min_valid_loss = None
        self.best_epoch = 0
        self.start_epoch = 0
        self.profile_dataloader = os.environ.get("RPPG_PROFILE_DATALOADER", "0") == "1"

        self._validate_modality_mode()
        self._validate_temporal_length(config)
        self.model = self._build_model().to(self.device)
        if self.num_of_gpu > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(self.num_of_gpu)))

        if config.MODEL.RESUME:
            self.start_epoch = self._resume_from_checkpoint(config.MODEL.RESUME)

        self.criterion = Neg_Pearson()

        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.TRAIN.LR,
                weight_decay=config.TRAIN.WEIGHT_DECAY,
            )
            remaining_epochs = config.TRAIN.EPOCHS - self.start_epoch
            if remaining_epochs <= 0:
                raise ValueError(
                    "TRAIN.EPOCHS ({}) must be greater than resumed start epoch ({}).".format(
                        config.TRAIN.EPOCHS, self.start_epoch
                    )
                )
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config.TRAIN.LR,
                epochs=remaining_epochs,
                steps_per_epoch=self.num_train_batches,
            )
        elif config.TOOLBOX_MODE == "only_test":
            pass
        else:
            raise ValueError("FusionPhys trainer initialized in incorrect toolbox mode!")

    def _validate_modality_mode(self):
        """检查配置中的模态模式是否受 FusionPhysTrainer 支持。"""
        if self.modality_mode not in self.SUPPORTED_MODALITY_MODES:
            raise ValueError("Unsupported FusionPhys modality mode: {}".format(self.modality_mode))

    def _build_model(self):
        """根据配置的模态组合构建 FusionPhys 模型。"""
        model_cfg = self.config.MODEL.FUSIONPHYS
        in_chs = tuple(model_cfg.IN_CHS) if len(model_cfg.IN_CHS) > 0 else None
        dim_expansion = model_cfg.DIM_EXPANSION if model_cfg.DIM_EXPANSION > 0 else None
        return FusionPhys(
            modality_mode=self.modality_mode,
            in_ch=model_cfg.IN_CH,
            in_chs=in_chs,
            dim=model_cfg.DIM,
            dim_expansion=dim_expansion,
            input_size=model_cfg.INPUT_SIZE,
            input_frames=model_cfg.INPUT_FRAMES,
            submodality_grid_size=model_cfg.SUBMODALITY_GRID_SIZE,
            kernel_ratio=model_cfg.KERNEL_RATIO,
            wavelet_level=model_cfg.WAVELET_LEVEL,
            num_blocks=model_cfg.PIB_BLOCKS,
            pib_keep_prob_start=model_cfg.PIB_KEEP_PROB_START,
            pib_keep_prob_end=model_cfg.PIB_KEEP_PROB_END,
        )

    @staticmethod
    def _validate_temporal_length(config):
        """检查片段长度是否满足当前小波层数的要求。"""
        required_multiple = 2 ** config.MODEL.FUSIONPHYS.WAVELET_LEVEL
        split_names = ["TEST"]
        if config.TOOLBOX_MODE == "train_and_test":
            split_names.insert(0, "TRAIN")
            if not config.TEST.USE_LAST_EPOCH:
                split_names.insert(1, "VALID")

        for split_name in split_names:
            data_cfg = getattr(config, split_name).DATA
            chunk_length = data_cfg.PREPROCESS.CHUNK_LENGTH
            if chunk_length % required_multiple != 0:
                raise ValueError(
                    "FusionPhys requires {}.DATA.PREPROCESS.CHUNK_LENGTH ({}) to be a "
                    "multiple of {} because the configured MWT level is {}.".format(
                        split_name,
                        chunk_length,
                        required_multiple,
                        config.MODEL.FUSIONPHYS.WAVELET_LEVEL,
                    )
                )

    def _resume_from_checkpoint(self, checkpoint_path):
        """加载断点权重，并返回继续训练时应开始的 epoch。"""
        if not os.path.exists(checkpoint_path):
            raise ValueError("Resume checkpoint path does not exist: {}".format(checkpoint_path))

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError:
            model_is_parallel = isinstance(self.model, torch.nn.DataParallel)
            has_module_prefix = any(key.startswith("module.") for key in state_dict.keys())
            if model_is_parallel and not has_module_prefix:
                state_dict = {"module." + key: value for key, value in state_dict.items()}
            elif not model_is_parallel and has_module_prefix:
                state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
            else:
                raise
            self.model.load_state_dict(state_dict)

        resume_epoch = self._infer_epoch_from_checkpoint(checkpoint_path)
        start_epoch = resume_epoch + 1
        print("Resumed FusionPhys model from: {}".format(checkpoint_path))
        print("Continuing training from epoch: {}".format(start_epoch))
        return start_epoch

    @staticmethod
    def _infer_epoch_from_checkpoint(checkpoint_path):
        """从 _Epoch{n}.pth 格式的文件名中解析保存的 epoch。"""
        match = re.search(r"_Epoch(\d+)\.pth$", os.path.basename(checkpoint_path))
        if match is None:
            raise ValueError(
                "Could not infer resume epoch from checkpoint filename: {}. "
                "Expected a suffix like '_Epoch39.pth'.".format(checkpoint_path)
            )
        return int(match.group(1))

    def _unpack_batch(self, batch):
        """把 batch 移到目标设备，并根据模态模式拆出 RGB/NIR/RF。"""
        if len(batch) < 2:
            raise ValueError("FusionPhys batches must contain video and ppg label.")
        video = batch[0].to(self.device, dtype=torch.float32, non_blocking=True)
        labels = batch[1].to(self.device, dtype=torch.float32, non_blocking=True)
        nir = None
        rf = None

        if self.modality_mode == "camera_only":
            return video, labels, nir, rf

        if len(batch) < 3:
            raise ValueError("FusionPhys {} mode requires paired modality data in the batch.".format(self.modality_mode))

        if self.modality_mode == "camera_camera_fusion":
            nir = batch[2].to(self.device, dtype=torch.float32, non_blocking=True)
        elif self.modality_mode == "camera_rf_fusion":
            rf = batch[2].to(self.device, dtype=torch.float32, non_blocking=True)

        return video, labels, nir, rf

    def _forward(self, video, nir=None, rf=None):
        """根据 MODALITY_MODE 用对应的输入形式执行模型前向。"""
        if self.modality_mode == "camera_only":
            return self.model(video)
        if self.modality_mode == "camera_camera_fusion":
            return self.model(video, nir)
        if self.modality_mode == "camera_rf_fusion":
            return self.model(video, rf)
        raise ValueError("Unsupported FusionPhys modality mode: {}".format(self.modality_mode))

    def train(self, data_loader):
        """训练 FusionPhys，并在需要时根据验证集选择最佳 epoch。"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        mean_training_losses = []
        mean_valid_losses = []
        lrs = []

        for epoch in range(self.start_epoch, self.max_epoch_num):
            print("")
            print(f"====Training Epoch: {epoch}====")
            self.model.train()
            train_loss = []
            tbar = tqdm(data_loader["train"], dynamic_ncols=True)
            epoch_start_time = time.perf_counter()
            end_time = time.perf_counter()
            profile_data_times = []
            profile_compute_times = []
            profile_total_times = []
            for _, batch in enumerate(tbar):
                data_time = time.perf_counter() - end_time
                batch_start_time = time.perf_counter()
                tbar.set_description("Train epoch %s" % epoch)
                video, labels, nir, rf = self._unpack_batch(batch)

                self.optimizer.zero_grad()
                preds = self._forward(video, nir=nir, rf=rf)
                loss = self.criterion(preds, labels)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                train_loss.append(loss.item())
                lrs.append(self.optimizer.param_groups[0]["lr"])
                if self.profile_dataloader:
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    compute_time = time.perf_counter() - batch_start_time
                    total_time = data_time + compute_time
                    profile_data_times.append(data_time)
                    profile_compute_times.append(compute_time)
                    profile_total_times.append(total_time)
                    tbar.set_postfix(
                        loss=loss.item(),
                        data="{:.2f}s".format(data_time),
                        compute="{:.2f}s".format(compute_time),
                        batch="{:.2f}s".format(total_time),
                    )
                else:
                    tbar.set_postfix(loss=loss.item())
                end_time = time.perf_counter()

            if self.profile_dataloader and profile_total_times:
                epoch_wall_time = time.perf_counter() - epoch_start_time
                print(
                    "Profile epoch {}: wall={:.2f}s, wall/it={:.2f}s, "
                    "data avg/max={:.2f}/{:.2f}s, "
                    "compute avg/max={:.2f}/{:.2f}s, "
                    "batch avg/max={:.2f}/{:.2f}s".format(
                        epoch,
                        epoch_wall_time,
                        epoch_wall_time / len(profile_total_times),
                        float(np.mean(profile_data_times)),
                        float(np.max(profile_data_times)),
                        float(np.mean(profile_compute_times)),
                        float(np.max(profile_compute_times)),
                        float(np.mean(profile_total_times)),
                        float(np.max(profile_total_times)),
                    )
                )

            mean_training_losses.append(np.mean(np.asarray(train_loss)))
            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print("validation loss: ", valid_loss)
                if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))

        if not self.config.TEST.USE_LAST_EPOCH:
            print("best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))
        self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        """在验证集上计算 loss，不更新模型参数。"""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print("")
        print("===Validating===")
        valid_loss = []
        self.model.eval()
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for _, batch in enumerate(vbar):
                vbar.set_description("Validation")
                video, labels, nir, rf = self._unpack_batch(batch)
                preds = self._forward(video, nir=nir, rf=rf)
                loss = self.criterion(preds, labels)
                valid_loss.append(loss.item())
                vbar.set_postfix(loss=loss.item())
        return np.mean(np.asarray(valid_loss))

    def test(self, data_loader):
        """执行测试推理，按样本重组 clip，并计算 rPPG 指标。"""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print("")
        print("===Testing===")
        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH, map_location=self.device))
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                model_path = os.path.join(
                    self.model_dir,
                    self.model_file_name + "_Epoch" + str(self.max_epoch_num - 1) + ".pth",
                )
                print("Testing uses last epoch as non-pretrained model!")
            else:
                model_path = os.path.join(
                    self.model_dir,
                    self.model_file_name + "_Epoch" + str(self.best_epoch) + ".pth",
                )
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
            print(model_path)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.model = self.model.to(self.device)
        self.model.eval()
        predictions = {}
        labels = {}
        sample_offset = 0
        with torch.no_grad():
            for _, batch in enumerate(data_loader["test"]):
                video, batch_labels, nir, rf = self._unpack_batch(batch)
                preds = self._forward(video, nir=nir, rf=rf)
                batch_size = preds.shape[0]
                for idx in range(batch_size):
                    if len(batch) >= 5:
                        subj_index = batch[3][idx]
                        sort_index = int(batch[4][idx])
                        if len(batch) >= 6:
                            frame_start = int(batch[5][idx])
                            sort_index = (frame_start, sort_index)
                    elif len(batch) >= 4:
                        subj_index = batch[2][idx]
                        sort_index = int(batch[3][idx])
                    else:
                        subj_index = "sample_{}".format(sample_offset + idx)
                        sort_index = 0

                    if subj_index not in predictions:
                        predictions[subj_index] = {}
                        labels[subj_index] = {}
                    predictions[subj_index][sort_index] = preds[idx].detach().cpu().view(-1)
                    labels[subj_index][sort_index] = batch_labels[idx].detach().cpu().view(-1)
                sample_offset += batch_size

        print("")
        calculate_metrics(predictions, labels, self.config)

    def save_model(self, index):
        """按 toolbox 的命名方式保存当前模型权重。"""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, self.model_file_name + "_Epoch" + str(index) + ".pth")
        torch.save(self.model.state_dict(), model_path)
        print("Saved Model Path: ", model_path)
