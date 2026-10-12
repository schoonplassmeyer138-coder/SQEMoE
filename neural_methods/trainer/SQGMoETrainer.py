"""Trainer for the RGB-RF SQGMoE rPPG model."""

import csv
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from evaluation.metrics import calculate_metrics
from neural_methods.loss.SQGMoELoss import SQGMoELoss
from neural_methods.model.SQGMoE import SQGMoE
from neural_methods.trainer.BaseTrainer import BaseTrainer


class SQGMoETrainer(BaseTrainer):
    """Train, validate and test SQGMoE on paired RGB-RF waveform data."""

    def __init__(self, config, data_loader):
        super().__init__()
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.MODEL.MODEL_FILE_NAME
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.min_valid_loss = None
        self.best_epoch = 0
        self.start_epoch = 0

        self.model = self._build_model().to(self.device)
        if self.num_of_gpu > 1:
            self.model = torch.nn.DataParallel(
                self.model, device_ids=list(range(self.num_of_gpu))
            )

        if config.MODEL.RESUME:
            self.start_epoch = self._resume_from_checkpoint(config.MODEL.RESUME)

        # 保留融合波形和 CogMoE 损失，并为 RGB/RF 投影分别增加 PPG 监督。
        self.criterion = SQGMoELoss(
            rgb_ppg_weight=config.MODEL.SQGMOE.RGB_PPG_WEIGHT,
            rf_ppg_weight=config.MODEL.SQGMOE.RF_PPG_WEIGHT,
        )
        self.grad_clip = float(config.TRAIN.GRAD_CLIP)

        if config.TOOLBOX_MODE == "train_and_test":
            if data_loader["train"] is None:
                raise ValueError("No data for SQGMoE training.")
            self.num_train_batches = len(data_loader["train"])
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.TRAIN.LR,
                weight_decay=config.TRAIN.WEIGHT_DECAY,
            )
            remaining_epochs = self.max_epoch_num - self.start_epoch
            if remaining_epochs <= 0:
                raise ValueError(
                    "TRAIN.EPOCHS ({}) must be greater than resumed start epoch "
                    "({}).".format(self.max_epoch_num, self.start_epoch)
                )
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config.TRAIN.LR,
                epochs=remaining_epochs,
                steps_per_epoch=self.num_train_batches,
            )
        elif config.TOOLBOX_MODE != "only_test":
            raise ValueError("SQGMoE trainer initialized in incorrect toolbox mode!")

    def _build_model(self):
        model_cfg = self.config.MODEL.SQGMOE
        data_cfg = (
            self.config.TEST.DATA
            if self.config.TOOLBOX_MODE == "only_test"
            else self.config.TRAIN.DATA
        )
        if data_cfg.FS <= 0:
            raise ValueError("SQGMoE requires a positive DATA.FS for DFT quality.")
        return SQGMoE(
            rgb_encoder=model_cfg.RGB_ENCODER,
            num_layers=model_cfg.NUM_LAYERS,
            dropout=model_cfg.DROPOUT,
            quality_fs=data_cfg.FS,
            quality_hr_range=tuple(model_cfg.QUALITY_HR_RANGE),
            quality_spectrum_range=tuple(model_cfg.QUALITY_SPECTRUM_RANGE),
            quality_peak_sigma_hz=model_cfg.QUALITY_PEAK_SIGMA_HZ,
            quality_snr_midpoint_db=model_cfg.QUALITY_SNR_MIDPOINT_DB,
        )

    def _unpack_batch(self, batch):
        """Read EquiPleth's RGB, PPG and RF tensors and move them to device."""
        if len(batch) < 3:
            raise ValueError("SQGMoE requires a batch containing RGB, PPG and RF.")
        rgb = batch[0].to(self.device, dtype=torch.float32, non_blocking=True)
        labels = batch[1].to(self.device, dtype=torch.float32, non_blocking=True)
        rf = batch[2].to(self.device, dtype=torch.float32, non_blocking=True)
        return rgb, labels, rf

    @staticmethod
    def _new_quality_accumulator():
        """Create constant-memory accumulators for RGB/RF quality monitoring."""
        return {"snr_sum": None, "quality_sum": None, "sample_count": 0}

    @staticmethod
    def _update_quality_accumulator(accumulator, outputs):
        """Accumulate per-sample linear SNR and quality without retaining graphs."""
        quality_details = outputs["quality_details"]
        linear_snr = quality_details["snr"].detach()
        quality = outputs["quality"].detach()
        if linear_snr.ndim != 2 or linear_snr.shape[1] != 2:
            raise ValueError(
                "SQGMoE linear SNR must be [B,2], got {}.".format(
                    tuple(linear_snr.shape)
                )
            )
        if quality.shape != linear_snr.shape:
            raise ValueError(
                "SQGMoE quality and SNR must have the same [B,2] shape."
            )

        # Only two detached values per metric remain on the GPU during an epoch.
        # This avoids retaining a batch's forward graph or all per-sample tensors.
        batch_snr_sum = linear_snr.to(dtype=torch.float64).sum(dim=0)
        batch_quality_sum = quality.to(dtype=torch.float64).sum(dim=0)
        if accumulator["snr_sum"] is None:
            accumulator["snr_sum"] = torch.zeros_like(batch_snr_sum)
            accumulator["quality_sum"] = torch.zeros_like(batch_quality_sum)
        accumulator["snr_sum"] += batch_snr_sum
        accumulator["quality_sum"] += batch_quality_sum
        accumulator["sample_count"] += linear_snr.shape[0]

    @staticmethod
    def _finalize_quality_accumulator(accumulator):
        """Return epoch means ordered as RGB then RF."""
        sample_count = accumulator["sample_count"]
        if sample_count <= 0:
            raise ValueError("Cannot compute SQGMoE quality statistics without samples.")
        mean_snr = (accumulator["snr_sum"] / sample_count).cpu()
        mean_quality = (accumulator["quality_sum"] / sample_count).cpu()
        return {
            "rgb_snr": mean_snr[0].item(),
            "rf_snr": mean_snr[1].item(),
            "rgb_quality": mean_quality[0].item(),
            "rf_quality": mean_quality[1].item(),
        }

    def _save_quality_history(self, history, make_plot=False):
        """Save per-epoch train/validation SNR and quality statistics."""
        output_dir = os.path.join(self.config.LOG.PATH, "plots")
        os.makedirs(output_dir, exist_ok=True)
        filename_root = self.model_file_name + "_snr_quality"
        csv_path = os.path.join(output_dir, filename_root + ".csv")
        fieldnames = [
            "epoch",
            "train_rgb_snr",
            "train_rf_snr",
            "train_rgb_quality",
            "train_rf_quality",
            "valid_rgb_snr",
            "valid_rf_snr",
            "valid_rgb_quality",
            "valid_rf_quality",
        ]
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)

        if not make_plot:
            return

        epochs = [record["epoch"] for record in history]
        figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        for split, linestyle in (("train", "-"), ("valid", "--")):
            axes[0].plot(
                epochs,
                [record[split + "_rgb_snr"] for record in history],
                linestyle,
                label=split.capitalize() + " RGB",
            )
            axes[0].plot(
                epochs,
                [record[split + "_rf_snr"] for record in history],
                linestyle,
                label=split.capitalize() + " RF",
            )
            axes[1].plot(
                epochs,
                [record[split + "_rgb_quality"] for record in history],
                linestyle,
                label=split.capitalize() + " RGB",
            )
            axes[1].plot(
                epochs,
                [record[split + "_rf_quality"] for record in history],
                linestyle,
                label=split.capitalize() + " RF",
            )

        axes[0].set_ylabel("Mean linear SNR")
        axes[0].set_title("RGB/RF linear SNR")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Mean quality")
        axes[1].set_title("RGB/RF quality score")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        figure.tight_layout()
        plot_path = os.path.join(output_dir, filename_root + ".pdf")
        figure.savefig(plot_path, dpi=300)
        plt.close(figure)
        print("Saving SQGMoE SNR/quality statistics to:", output_dir)

    def train(self, data_loader):
        mean_training_losses = []
        mean_valid_losses = []
        learning_rates = []
        quality_history = []

        for epoch in range(self.start_epoch, self.max_epoch_num):
            print("")
            print("====Training Epoch: {}====".format(epoch))
            self.model.train()
            epoch_losses = []
            train_quality_accumulator = self._new_quality_accumulator()
            progress = tqdm(data_loader["train"], dynamic_ncols=True)

            for batch in progress:
                progress.set_description("Train epoch {}".format(epoch))
                rgb, labels, rf = self._unpack_batch(batch)

                self.optimizer.zero_grad()
                outputs = self.model(rgb, rf, return_details=True)
                self._update_quality_accumulator(
                    train_quality_accumulator, outputs
                )
                loss_dict = self.criterion(outputs, labels, epoch=epoch)
                loss = loss_dict["total"]
                loss.backward()
                if self.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.optimizer.step()
                self.scheduler.step()

                epoch_losses.append(loss.item())
                learning_rates.append(self.optimizer.param_groups[0]["lr"])
                progress.set_postfix(
                    total=loss.item(),
                    waveform=loss_dict["waveform"].item(),
                    rgb_ppg=loss_dict["rgb_ppg"].item(),
                    rf_ppg=loss_dict["rf_ppg"].item(),
                    noise=loss_dict["noise"].item(),
                    refinement=loss_dict["refinement"].item(),
                    gate=loss_dict["gate_balance"].item(),
                )

            mean_training_losses.append(float(np.mean(epoch_losses)))
            train_quality_stats = self._finalize_quality_accumulator(
                train_quality_accumulator
            )
            self.save_model(epoch)

            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss, valid_quality_stats = self.valid(
                    data_loader, epoch=epoch, return_quality_stats=True
                )
                mean_valid_losses.append(valid_loss)
                print("validation loss: ", valid_loss)
                if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
            else:
                valid_quality_stats = {
                    "rgb_snr": float("nan"),
                    "rf_snr": float("nan"),
                    "rgb_quality": float("nan"),
                    "rf_quality": float("nan"),
                }

            quality_history.append(
                {
                    "epoch": epoch,
                    "train_rgb_snr": train_quality_stats["rgb_snr"],
                    "train_rf_snr": train_quality_stats["rf_snr"],
                    "train_rgb_quality": train_quality_stats["rgb_quality"],
                    "train_rf_quality": train_quality_stats["rf_quality"],
                    "valid_rgb_snr": valid_quality_stats["rgb_snr"],
                    "valid_rf_snr": valid_quality_stats["rf_snr"],
                    "valid_rgb_quality": valid_quality_stats["rgb_quality"],
                    "valid_rf_quality": valid_quality_stats["rf_quality"],
                }
            )
            # Rewrite the small CSV every epoch so interrupted runs retain data.
            self._save_quality_history(quality_history)
            print(
                "train mean linear SNR (RGB/RF): {:.4f}/{:.4f}, "
                "quality: {:.4f}/{:.4f}".format(
                    train_quality_stats["rgb_snr"],
                    train_quality_stats["rf_snr"],
                    train_quality_stats["rgb_quality"],
                    train_quality_stats["rf_quality"],
                )
            )
            if not self.config.TEST.USE_LAST_EPOCH:
                print(
                    "valid mean linear SNR (RGB/RF): {:.4f}/{:.4f}, "
                    "quality: {:.4f}/{:.4f}".format(
                        valid_quality_stats["rgb_snr"],
                        valid_quality_stats["rf_snr"],
                        valid_quality_stats["rgb_quality"],
                        valid_quality_stats["rf_quality"],
                    )
                )

        if not self.config.TEST.USE_LAST_EPOCH:
            print(
                "best trained epoch: {}, min_val_loss: {}".format(
                    self.best_epoch, self.min_valid_loss
                )
            )
        self.plot_losses_and_lrs(
            mean_training_losses, mean_valid_losses, learning_rates, self.config
        )
        self._save_quality_history(quality_history, make_plot=True)

    def valid(self, data_loader, epoch=0, return_quality_stats=False):
        if data_loader["valid"] is None:
            raise ValueError("No data for validation.")

        print("")
        print("===Validating===")
        losses = []
        quality_accumulator = self._new_quality_accumulator()
        self.model.eval()
        with torch.no_grad():
            progress = tqdm(data_loader["valid"], dynamic_ncols=True)
            for batch in progress:
                rgb, labels, rf = self._unpack_batch(batch)
                outputs = self.model(rgb, rf, return_details=True)
                self._update_quality_accumulator(quality_accumulator, outputs)
                loss_dict = self.criterion(outputs, labels, epoch=epoch)
                loss = loss_dict["total"]
                losses.append(loss.item())
                progress.set_postfix(
                    total=loss.item(),
                    waveform=loss_dict["waveform"].item(),
                    rgb_ppg=loss_dict["rgb_ppg"].item(),
                    rf_ppg=loss_dict["rf_ppg"].item(),
                    noise=loss_dict["noise"].item(),
                    refinement=loss_dict["refinement"].item(),
                    gate=loss_dict["gate_balance"].item(),
                )
        mean_loss = float(np.mean(losses))
        if return_quality_stats:
            return mean_loss, self._finalize_quality_accumulator(
                quality_accumulator
            )
        return mean_loss

    def test(self, data_loader):
        if data_loader["test"] is None:
            raise ValueError("No data for testing.")

        print("")
        print("===Testing===")
        model_path = self._resolve_test_model_path()
        print(model_path)
        self._load_model_state(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        predictions = {}
        labels = {}
        sample_offset = 0
        with torch.no_grad():
            for batch in tqdm(data_loader["test"], dynamic_ncols=True):
                rgb, batch_labels, rf = self._unpack_batch(batch)
                batch_predictions = self.model(rgb, rf)

                for index in range(batch_predictions.shape[0]):
                    subject, sort_index = self._sample_key(
                        batch, index, sample_offset
                    )
                    if subject not in predictions:
                        predictions[subject] = {}
                        labels[subject] = {}
                    predictions[subject][sort_index] = (
                        batch_predictions[index].detach().cpu().view(-1)
                    )
                    labels[subject][sort_index] = (
                        batch_labels[index].detach().cpu().view(-1)
                    )
                sample_offset += batch_predictions.shape[0]

        print("")
        calculate_metrics(predictions, labels, self.config)

    @staticmethod
    def _sample_key(batch, index, sample_offset):
        """Get subject and chronological key from EquiPleth batch metadata."""
        if len(batch) >= 5:
            subject = batch[3][index]
            chunk_id = int(batch[4][index])
            if len(batch) >= 6:
                frame_start = int(batch[5][index])
                return subject, (frame_start, chunk_id)
            return subject, chunk_id
        return "sample_{}".format(sample_offset + index), 0

    def _resolve_test_model_path(self):
        if self.config.TOOLBOX_MODE == "only_test":
            model_path = self.config.INFERENCE.MODEL_PATH
            if not os.path.exists(model_path):
                raise ValueError(
                    "Inference model path error! Please check INFERENCE.MODEL_PATH."
                )
            print("Testing uses pretrained SQGMoE model!")
            return model_path

        if self.config.TEST.USE_LAST_EPOCH:
            epoch = self.max_epoch_num - 1
            print("Testing uses last training epoch!")
        else:
            epoch = self.best_epoch
            print("Testing uses best validation epoch!")
        return os.path.join(
            self.model_dir,
            self.model_file_name + "_Epoch" + str(epoch) + ".pth",
        )

    def _resume_from_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise ValueError(
                "Resume checkpoint path does not exist: {}".format(checkpoint_path)
            )
        self._load_model_state(checkpoint_path)
        match = re.search(r"_Epoch(\d+)\.pth$", os.path.basename(checkpoint_path))
        if match is None:
            raise ValueError(
                "Could not infer epoch from checkpoint filename: {}".format(
                    checkpoint_path
                )
            )
        start_epoch = int(match.group(1)) + 1
        print("Resumed SQGMoE model from: {}".format(checkpoint_path))
        print("Continuing training from epoch: {}".format(start_epoch))
        return start_epoch

    def _load_model_state(self, checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        model_is_parallel = isinstance(self.model, torch.nn.DataParallel)
        has_module_prefix = any(key.startswith("module.") for key in state_dict)
        if model_is_parallel and not has_module_prefix:
            state_dict = {"module." + key: value for key, value in state_dict.items()}
        elif not model_is_parallel and has_module_prefix:
            state_dict = {
                key.replace("module.", "", 1): value
                for key, value in state_dict.items()
            }
        self.model.load_state_dict(state_dict)

    def save_model(self, epoch):
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(
            self.model_dir,
            self.model_file_name + "_Epoch" + str(epoch) + ".pth",
        )
        torch.save(self.model.state_dict(), model_path)
        print("Saved Model Path: ", model_path)
