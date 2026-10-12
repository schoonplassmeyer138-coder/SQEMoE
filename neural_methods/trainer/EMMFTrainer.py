"""Trainer for EMMF RGB-RF rPPG models."""

import os
import re

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.EMMFLoss import EMMFLoss
from neural_methods.model.EMMF import TrustAdaptiveFusionFT, TrustFusion
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm


class EMMFTrainer(BaseTrainer):
    def __init__(self, config, data_loader):
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.MODEL.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        #默认为0，如果需要从checkpoint恢复训练，则会在_resume_from_checkpoint方法中更新为正确的起始epoch。确保训练循环从正确的epoch开始，避免覆盖之前的训练结果。
        self.start_epoch = 0

        self.model = self._build_model(config).to(self.device)
        if self.num_of_gpu > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(self.num_of_gpu)))

        if config.MODEL.RESUME:
            self.start_epoch = self._resume_from_checkpoint(config.MODEL.RESUME)

        self.criterion = EMMFLoss(config)

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
            raise ValueError("EMMF trainer initialized in incorrect toolbox mode!")

    def _build_model(self, config):
        if config.MODEL.EMMF.MODEL_TYPE == "TrustFusion":
            return TrustFusion()
        if config.MODEL.EMMF.MODEL_TYPE == "TrustAdaptiveFusionFT":
            return TrustAdaptiveFusionFT()
        raise ValueError("Unsupported EMMF model type: {}".format(config.MODEL.EMMF.MODEL_TYPE))
    
    # 从checkpoint恢复模型权重，并推断恢复的epoch以继续训练。
    def _resume_from_checkpoint(self, checkpoint_path):
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
        print("Resumed EMMF model from: {}".format(checkpoint_path))
        print("Continuing training from epoch: {}".format(start_epoch))
        return start_epoch
    
    # 解析checkpoint文件名以推断恢复的epoch，期望文件名以"_Epoch{number}.pth"结尾，例如"EMMF_Epoch39.pth"。
    @staticmethod
    def _infer_epoch_from_checkpoint(checkpoint_path):
        match = re.search(r"_Epoch(\d+)\.pth$", os.path.basename(checkpoint_path))
        if match is None:
            raise ValueError(
                "Could not infer resume epoch from checkpoint filename: {}. "
                "Expected a suffix like '_Epoch39.pth'.".format(checkpoint_path)
            )
        return int(match.group(1))

    def _unpack_batch(self, batch):
        if len(batch) < 3:
            raise ValueError("EMMF batches must contain video, ppg label, and RF data.")
        video = batch[0].to(self.device, dtype=torch.float32, non_blocking=True)
        labels = batch[1].to(self.device, dtype=torch.float32, non_blocking=True)
        rf = batch[2].to(self.device, dtype=torch.float32, non_blocking=True)
        return video, labels, rf

    def _forward(self, video, rf):
        outputs = self.model(video, rf)
        pred_rgb = self._squeeze_signal(outputs[0])
        pred_rf = self._squeeze_signal(outputs[5])
        pred_fused = self._squeeze_signal(outputs[-1] if len(outputs) > 10 else 0.5 * (pred_rgb + pred_rf))
        return {
            "rgb": pred_rgb,
            "rf": pred_rf,
            "fused": pred_fused,
            "gamma_rgb": outputs[1],
            "v_rgb": outputs[2],
            "alpha_rgb": outputs[3],
            "beta_rgb": outputs[4],
            "gamma_rf": outputs[6],
            "v_rf": outputs[7],
            "alpha_rf": outputs[8],
            "beta_rf": outputs[9],
        }

    @staticmethod
    def _squeeze_signal(signal):
        if signal.dim() == 3 and signal.shape[-1] == 1:
            return signal.squeeze(-1)
        return signal

    @staticmethod
    def _detach_sample_signal(signal, index):
        # 从 batch 中取出单个样本，并整理成一维 CPU 张量，便于离线可视化保存。
        return signal[index].detach().cpu().view(-1)

    def train(self, data_loader):
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
            tbar = tqdm(data_loader["train"], ncols=80)
            for _, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                video, labels, rf = self._unpack_batch(batch)

                self.optimizer.zero_grad()
                outputs = self._forward(video, rf)
                loss = self.criterion(outputs, labels, epoch=epoch)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                train_loss.append(loss.item())
                lrs.append(self.optimizer.param_groups[0]["lr"])
                tbar.set_postfix(loss=loss.item(), snr_w=self.criterion._snr_weight(epoch))

            mean_training_losses.append(np.mean(np.asarray(train_loss)))
            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss = self.valid(data_loader, epoch=epoch)
                mean_valid_losses.append(valid_loss)
                print("validation loss: ", valid_loss)
                if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))

        if not self.config.TEST.USE_LAST_EPOCH:
            print("best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))
        self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader, epoch=None):
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
                video, labels, rf = self._unpack_batch(batch)
                outputs = self._forward(video, rf)
                loss = self.criterion(outputs, labels, epoch=epoch)
                valid_loss.append(loss.item())
                vbar.set_postfix(loss=loss.item())
        return np.mean(np.asarray(valid_loss))

    def test(self, data_loader):
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
        visual_outputs = {}
        sample_offset = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader["test"]):
                video, batch_labels, rf = self._unpack_batch(batch)
                outputs = self._forward(video, rf)
                pred_fused = outputs["fused"]
                batch_size = pred_fused.shape[0]
                for idx in range(batch_size):
                    if len(batch) >= 5:
                        subj_index = batch[3][idx]
                        sort_index = int(batch[4][idx])
                        if len(batch) >= 6:
                            frame_start = int(batch[5][idx])
                            sort_index = (frame_start, sort_index)
                    else:
                        subj_index = "sample_{}".format(sample_offset + idx)
                        sort_index = 0

                    if subj_index not in predictions:
                        predictions[subj_index] = {}
                        labels[subj_index] = {}
                        visual_outputs[subj_index] = {}
                    predictions[subj_index][sort_index] = pred_fused[idx].detach().cpu().view(-1)
                    labels[subj_index][sort_index] = batch_labels[idx].detach().cpu().view(-1)
                    visual_outputs[subj_index][sort_index] = {
                        "fused": self._detach_sample_signal(outputs["fused"], idx),
                        "rgb": self._detach_sample_signal(outputs["rgb"], idx),
                        "rf": self._detach_sample_signal(outputs["rf"], idx),
                        "label": self._detach_sample_signal(batch_labels, idx),
                        "gamma_rgb": self._detach_sample_signal(outputs["gamma_rgb"], idx),
                        "gamma_rf": self._detach_sample_signal(outputs["gamma_rf"], idx),
                        "v_rgb": self._detach_sample_signal(outputs["v_rgb"], idx),
                        "v_rf": self._detach_sample_signal(outputs["v_rf"], idx),
                    }
                sample_offset += batch_size

        print("")
        self.save_test_outputs(
            predictions,
            labels,
            self.config,
            extra_outputs={"emmf_visual": visual_outputs},
        )
        calculate_metrics(predictions, labels, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, self.model_file_name + "_Epoch" + str(index) + ".pth")
        torch.save(self.model.state_dict(), model_path)
        print("Saved Model Path: ", model_path)
