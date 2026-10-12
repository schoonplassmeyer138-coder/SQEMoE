"""Composite loss for the RGB-RF SQGMoE rPPG model."""

from typing import Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson


def _expert_mse_loss(
    expert_outputs_list: Sequence[Sequence[torch.Tensor]],
    expert_index: int,
    reference_features: torch.Tensor,
) -> torch.Tensor:
    """Average one expert's MSE to the detached pre-MoE fused features."""
    if not expert_outputs_list:
        raise ValueError("expert_outputs_list must contain at least one layer.")

    reference = reference_features.detach()
    losses: List[torch.Tensor] = []
    for layer_index, layer_outputs in enumerate(expert_outputs_list):
        if len(layer_outputs) <= expert_index:
            raise ValueError(
                "Layer {} does not contain expert index {}.".format(
                    layer_index, expert_index
                )
            )
        expert_output = layer_outputs[expert_index]
        if expert_output.shape != reference.shape:
            raise ValueError(
                "Expert output {} and fused reference {} must have the same shape."
                .format(tuple(expert_output.shape), tuple(reference.shape))
            )
        losses.append(F.mse_loss(expert_output, reference))
    return torch.stack(losses).mean()


def noise_suppression_loss(
    expert_outputs_list: Sequence[Sequence[torch.Tensor]],
    fused_features: torch.Tensor,
) -> torch.Tensor:
    """Constrain each layer's NRE output (expert 1) to the fused reference."""
    return _expert_mse_loss(expert_outputs_list, 1, fused_features)


def refinement_loss(
    expert_outputs_list: Sequence[Sequence[torch.Tensor]],
    fused_features: torch.Tensor,
) -> torch.Tensor:
    """Constrain each layer's CRE output (expert 2) to the fused reference."""
    return _expert_mse_loss(expert_outputs_list, 2, fused_features)


def gate_balance_loss(
    gate_weights_list: Sequence[torch.Tensor], num_experts: int = 3
) -> torch.Tensor:
    """Penalize imbalanced average expert usage across batch and time."""
    if not gate_weights_list:
        raise ValueError("gate_weights_list must contain at least one layer.")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive.")

    losses: List[torch.Tensor] = []
    target = 1.0 / num_experts
    for layer_index, gate_weights in enumerate(gate_weights_list):
        if gate_weights.ndim != 3 or gate_weights.shape[-1] != num_experts:
            raise ValueError(
                "Layer {} gate weights must be [B,T,{}], got {}.".format(
                    layer_index, num_experts, tuple(gate_weights.shape)
                )
            )
        average_usage = gate_weights.mean(dim=(0, 1))
        losses.append((average_usage - target).square().sum())
    return torch.stack(losses).mean()


class SQGMoELoss(nn.Module):
    """Fused and single-modality PPG losses plus CogMoE regularization.

    The expected ``outputs`` mapping is returned by
    ``SQGMoE.forward(..., return_details=True)``.
    """

    def __init__(
        self,
        noise_weight: float = 0.75,
        refinement_weight: float = 0.6,
        gate_weight_init: float = 1.0,
        gate_weight_max: float = 0.2,
        gate_weight_decay: float = 0.05,
        num_experts: int = 3,
        rgb_ppg_weight: float = 1.0,
        rf_ppg_weight: float = 1.0,
    ):
        super().__init__()
        self.waveform_loss = Neg_Pearson()
        self.noise_weight = float(noise_weight)
        self.refinement_weight = float(refinement_weight)
        self.gate_weight_init = float(gate_weight_init)
        self.gate_weight_max = float(gate_weight_max)
        self.gate_weight_decay = float(gate_weight_decay)
        self.num_experts = int(num_experts)
        self.rgb_ppg_weight = float(rgb_ppg_weight)
        self.rf_ppg_weight = float(rf_ppg_weight)

    def get_gate_weight(self, epoch: int) -> float:
        """Use the adaptive gate coefficient from the reference CogMoE."""
        decayed = self.gate_weight_init / (
            1.0 + self.gate_weight_decay * max(int(epoch), 0)
        )
        return min(self.gate_weight_max, decayed)

    def forward(
        self,
        outputs: Mapping[str, object],
        labels: torch.Tensor,
        epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        required = {
            "pred",
            "fused_features",
            "gate_weights_list",
            "expert_outputs_list",
            "quality_details",
        }
        missing = required.difference(outputs.keys())
        if missing:
            raise ValueError(
                "SQGMoELoss missing model outputs: {}.".format(sorted(missing))
            )

        prediction = outputs["pred"]
        fused_features = outputs["fused_features"]
        gate_weights_list = outputs["gate_weights_list"]
        expert_outputs_list = outputs["expert_outputs_list"]
        if not isinstance(prediction, torch.Tensor) or not isinstance(
            fused_features, torch.Tensor
        ):
            raise TypeError("pred and fused_features must be tensors.")

        waveform = self.waveform_loss(prediction, labels)
        # 监督加窗、DFT 前的 RGB/RF 投影波形；保留到线性头和 encoder 的梯度。
        modal_ppg = outputs["quality_details"]["projected_signals"]  # [B,2,T]
        rgb_ppg = self.waveform_loss(modal_ppg[:, 0, :], labels)
        rf_ppg = self.waveform_loss(modal_ppg[:, 1, :], labels)
        noise = noise_suppression_loss(expert_outputs_list, fused_features)
        refinement = refinement_loss(expert_outputs_list, fused_features)
        gate_balance = gate_balance_loss(gate_weights_list, self.num_experts)
        gate_weight = self.get_gate_weight(epoch)

        total = (
            waveform
            + self.noise_weight * noise
            + self.refinement_weight * refinement
            + gate_weight * gate_balance
            + self.rgb_ppg_weight * rgb_ppg
            + self.rf_ppg_weight * rf_ppg
        )
        return {
            "total": total,
            "waveform": waveform.detach(),
            "rgb_ppg": rgb_ppg.detach(),
            "rf_ppg": rf_ppg.detach(),
            "noise": noise.detach(),
            "refinement": refinement.detach(),
            "gate_balance": gate_balance.detach(),
            "gate_weight": total.new_tensor(gate_weight),
        }
