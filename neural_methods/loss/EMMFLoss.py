"""Losses for EMMF RGB-RF fusion models."""

import torch
from torch import nn

from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.loss.SNRLoss import SNRLoss_dB_Signals


class EMMFLoss(nn.Module):
    """Dynamic joint loss used by the EMMF trainer."""

    def __init__(self, config):
        super().__init__()
        self.fs = config.TRAIN.DATA.FS
        self.lambda_rf = config.MODEL.EMMF.LAMBDA_RF
        self.lambda_rgb = config.MODEL.EMMF.LAMBDA_RGB
        self.lambda_fusion = config.MODEL.EMMF.LAMBDA_FUSION
        self.lambda_npc_modal = config.MODEL.EMMF.LAMBDA_NPC_MODAL
        self.lambda_snr_modal = config.MODEL.EMMF.LAMBDA_SNR_MODAL
        self.lambda_unc_modal = config.MODEL.EMMF.LAMBDA_UNC_MODAL
        self.lambda_npc_fusion = config.MODEL.EMMF.LAMBDA_NPC_FUSION
        self.lambda_snr_fusion = config.MODEL.EMMF.LAMBDA_SNR_FUSION
        self.uncertainty_tau = config.MODEL.EMMF.UNCERTAINTY_TAU
        self.snr_dynamic_weight = config.MODEL.EMMF.SNR_DYNAMIC_WEIGHT
        self.snr_dynamic_start = config.MODEL.EMMF.SNR_DYNAMIC_START
        self.snr_dynamic_epochs = config.MODEL.EMMF.SNR_DYNAMIC_EPOCHS

        self.npc_loss = Neg_Pearson()
        self.snr_loss = SNRLoss_dB_Signals()

    def forward(self, outputs, labels, epoch=None):
        snr_weight = self._snr_weight(epoch)
        fusion_loss = self._fusion_loss(outputs["fused"], labels, snr_weight)
        rgb_loss = self._modal_loss(
            outputs["rgb"], outputs["gamma_rgb"], outputs["v_rgb"],
            outputs["alpha_rgb"], outputs["beta_rgb"], labels, snr_weight)
        rf_loss = self._modal_loss(
            outputs["rf"], outputs["gamma_rf"], outputs["v_rf"],
            outputs["alpha_rf"], outputs["beta_rf"], labels, snr_weight)

        loss = (
            self.lambda_fusion * fusion_loss
            + self.lambda_rgb * rgb_loss
            + self.lambda_rf * rf_loss
        )
        return torch.nan_to_num(loss, nan=0.0, posinf=2.0, neginf=2.0)

    def _snr_weight(self, epoch):
        if not self.snr_dynamic_weight:
            return 1.0
        if epoch is None:
            epoch = self.snr_dynamic_epochs
        if self.snr_dynamic_epochs <= 0:
            return 1.0

        progress = min(max(float(epoch), 0.0), float(self.snr_dynamic_epochs)) / float(self.snr_dynamic_epochs)
        return self.snr_dynamic_start * ((1.0 / self.snr_dynamic_start) ** progress)

    def _modal_loss(self, pred, gamma, v, alpha, beta, labels, snr_weight):
        npc = self.npc_loss(pred, labels)
        snr = self.snr_loss(pred, labels, Fs=self.fs)
        unc = self._uncertainty_loss(gamma, v, alpha, beta, labels)
        return (
            self.lambda_npc_modal * npc
            + self.lambda_snr_modal * snr_weight * snr
            + self.lambda_unc_modal * unc
        )

    def _fusion_loss(self, pred, labels, snr_weight):
        npc = self.npc_loss(pred, labels)
        snr = self.snr_loss(pred, labels, Fs=self.fs)
        return self.lambda_npc_fusion * npc + self.lambda_snr_fusion * snr_weight * snr

    def _uncertainty_loss(self, gamma, v, alpha, beta, labels):
        labels = labels.unsqueeze(-1)
        gamma = gamma.to(torch.float32)
        v = torch.clamp(v.to(torch.float32), min=1e-6)
        alpha = torch.clamp(alpha.to(torch.float32), min=1.0 + 1e-6)
        beta = torch.clamp(beta.to(torch.float32), min=1e-6)

        two_beta = 2.0 * beta
        error = labels - gamma
        nll = (
            0.5 * torch.log(torch.pi / v)
            - alpha * torch.log(two_beta * (1.0 + v))
            + (alpha + 0.5) * torch.log(v * error.pow(2) + two_beta * (1.0 + v))
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )
        evidence = 2.0 * v + alpha
        reg = torch.abs(error) * evidence
        return nll.mean() + self.uncertainty_tau * reg.mean()
