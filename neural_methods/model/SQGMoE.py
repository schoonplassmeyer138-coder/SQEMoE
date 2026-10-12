"""CogMoE-style RGB-RF model for waveform-based rPPG estimation.

The two modality encoders deliberately reuse the implementations already in
this repository:

* RGB: RhythmFormer without ``ConvBlockLast`` or PhysNet without ``ConvBlock10``.
* RF: FusionPhys ``RFEncoder``.

Both branches are converted to ``[B, T, D]``.  Cross-modal attention then
produces one fused sequence, which is processed by the same three-expert
quality-guided Transformer design used by CogMoE.  Unlike the original
classification model, the final head predicts one rPPG value per frame.
"""

from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from neural_methods.model.FusionPhys import RFEncoder
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.model.RhythmFormer import RhythmFormer


def _to_btchw(x: torch.Tensor) -> torch.Tensor:
    """Convert RGB input to the shared ``[B, T, C, H, W]`` layout."""
    if x.ndim != 5:
        raise ValueError(
            "RGB input must be [B,C,T,H,W] or [B,T,C,H,W], got {}.".format(
                tuple(x.shape)
            )
        )
    if x.shape[2] == 3:
        return x.contiguous()
    if x.shape[1] == 3:
        return x.permute(0, 2, 1, 3, 4).contiguous()
    raise ValueError(
        "Cannot find the three-channel RGB axis in input shape {}.".format(
            tuple(x.shape)
        )
    )


def _format_rf(x: torch.Tensor) -> torch.Tensor:
    """Convert RF input to the ``[B, C_rf, L]`` layout used by RFEncoder."""
    if x.ndim == 3:
        return x.contiguous()
    if x.ndim == 4 and x.shape[-1] == 1:
        return x.squeeze(-1).contiguous()
    if x.ndim == 4:
        batch, channels, window, length = x.shape
        return x.reshape(batch, channels * window, length).contiguous()
    raise ValueError(
        "RF input must be [B,C,L] or a flattenable 4-D tensor, got {}.".format(
            tuple(x.shape)
        )
    )


class RhythmFormerEncoder(nn.Module):
    """RhythmFormer feature extractor with its final waveform head removed."""

    def __init__(
        self,
        d_model: int = 64,
    ):
        super().__init__()
        rgb_head_dim = 16
        if d_model % rgb_head_dim != 0:
            raise ValueError("RhythmFormer d_model must be divisible by 16.")

        self.d_model = d_model
        # 保留 RhythmFormer 原始 head_dim=16，不与后续融合模块的 nhead 绑定。
        self.backbone = RhythmFormer(
            dim=d_model,
            in_chans=64,
            head_dim=rgb_head_dim,
            embed_dim=[d_model, d_model, d_model],
        )
        # The wrapper stops before this module. Replacing it also ensures that
        # no unused prediction-head parameters enter the optimizer/state dict.
        self.backbone.ConvBlockLast = nn.Identity()

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        # EquiPleth DataLoader 默认给出 [B,3,T,H,W]；RhythmFormer 使用 [B,T,3,H,W]。
        # [B,3,T,H,W] 或 [B,T,3,H,W] -> [B,T,3,H,W]
        rgb = _to_btchw(rgb)
        batch, frames, channels, _, _ = rgb.shape
        if channels != 3:
            raise ValueError("RhythmFormerEncoder expects three RGB channels.")
        if frames < 2:
            raise ValueError("RhythmFormer's Fusion_Stem requires at least 2 frames.")

        # This follows RhythmFormer.forward exactly up to ConvBlockLast.
        # Fusion_Stem 把时间维临时并入 batch，只下采样空间维：
        # [B,T,3,H,W] -> [B*T,64,H/4,W/4]
        x = self.backbone.Fusion_Stem(rgb)
        _, stem_channels, stem_h, stem_w = x.shape

        # 恢复被并入 batch 的时间维：
        # [B*T,64,H/4,W/4] -> [B,T,64,H/4,W/4]
        x = x.reshape(batch, frames, stem_channels, stem_h, stem_w)

        # 把通道维移到 Conv3d 要求的位置：
        # [B,T,64,H/4,W/4] -> [B,64,T,H/4,W/4]
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        # patch_embedding 的 kernel/stride=(1,4,4)，只继续下采样空间维：
        # [B,64,T,H/4,W/4] -> [B,D,T,H/16,W/16]
        x = self.backbone.patch_embedding(x)

        # 每个 RhythmFormer stage 内部会先下采样再恢复时间维；空间尺寸保持不变：
        # [B,D,T,H/16,W/16] -> [B,D,T',H/16,W/16]
        # 当 T 能被最大时间尺度整除时（EquiPleth 的 288 能被 8 整除），T'=T。
        for stage in self.backbone.stages:
            x = stage(x)

        # 对空间 patch 做平均，再交换时间维和特征维：
        # [B,D,T',H/16,W/16] -> [B,D,T'] -> [B,T',D]
        # 这就是 RhythmFormer 原 1x1 Conv1d 波形预测头之前的特征。
        x = x.mean(dim=(-2, -1)).transpose(1, 2).contiguous()
        # 最终 RGB encoder 输出：[B,T,D]
        return x


class PhysNetEncoder(nn.Module):
    """Extract PhysNet's pre-head features as ``[B, T, D]``."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.backbone = PhysNet_padding_Encoder_Decoder_MAX()
        # 不保留原波形预测头；RGB 的 PPG 监督使用质量估计器中的线性头。
        self.backbone.ConvBlock10 = nn.Identity()
        self.projection = nn.Identity() if d_model == 64 else nn.Linear(64, d_model)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb = _to_btchw(rgb)
        frames = rgb.shape[1]
        x = rgb.permute(0, 2, 1, 3, 4).contiguous()

        # 与 PhysNet.forward 中 ConvBlock10 之前的特征提取顺序一致。
        x = self.backbone.ConvBlock1(x)
        x = self.backbone.MaxpoolSpa(x)
        x = self.backbone.ConvBlock2(x)
        x = self.backbone.ConvBlock3(x)
        x = self.backbone.MaxpoolSpaTem(x)
        x = self.backbone.ConvBlock4(x)
        x = self.backbone.ConvBlock5(x)
        x = self.backbone.MaxpoolSpaTem(x)
        x = self.backbone.ConvBlock6(x)
        x = self.backbone.ConvBlock7(x)
        x = self.backbone.MaxpoolSpa(x)
        x = self.backbone.ConvBlock8(x)
        x = self.backbone.ConvBlock9(x)
        x = self.backbone.upsample(x)
        x = self.backbone.upsample2(x)

        # 使用当前 clip 的帧数，替代原 poolspa 固定的输出长度（128）。
        # [B,64,T',H',W'] -> [B,64,T,1,1] -> [B,T,64]
        x = F.adaptive_avg_pool3d(x, (frames, 1, 1))
        x = x.squeeze(-1).squeeze(-1).transpose(1, 2).contiguous()
        return self.projection(x)


class FusionPhysRFEncoder(nn.Module):
    """Adapter around FusionPhys.RFEncoder that returns ``[B, T, D]``."""

    def __init__(self, in_ch: int = 10, d_model: int = 64):
        super().__init__()
        self.in_ch = in_ch
        self.encoder = RFEncoder(in_ch=in_ch, dim=d_model)

    def forward(self, rf: torch.Tensor) -> torch.Tensor:
        rf = _format_rf(rf)
        if rf.shape[1] != self.in_ch:
            raise ValueError(
                "RFEncoder expects {} channels, got {}.".format(
                    self.in_ch, rf.shape[1]
                )
            )
        # RFEncoder: [B,C_rf,L] -> [B,D,L/4,1].
        x = self.encoder(rf).squeeze(-1).transpose(1, 2).contiguous()
        return x


class DFTSNRQualityEstimator(nn.Module):
    """Project each modality to 1-D and estimate its quality using DFT-PSD SNR.

    Each modality owns an independent ``Linear(D, 1)`` projection. The two
    resulting temporal signals are converted to one-sided PSDs. A differentiable
    soft peak is located inside the heart-rate band, and SNR is computed as the
    peak-neighborhood power divided by all remaining valid-band power. Each
    modality's linear SNR is independently mapped to a bounded absolute quality
    score using a configurable SNR midpoint.
    """

    def __init__(
        self,
        d_model: int,
        num_modalities: int = 2,
        fs: float = 30.0,
        hr_f_min: float = 0.7,
        hr_f_max: float = 3.0,
        spectrum_f_min: float = 0.4,
        spectrum_f_max: float = 4.0,
        peak_sigma_hz: float = 0.1,
        snr_midpoint_db: float = 7.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        if not 0.0 <= spectrum_f_min < hr_f_min < hr_f_max <= spectrum_f_max:
            raise ValueError(
                "Expected spectrum_f_min < hr_f_min < hr_f_max <= spectrum_f_max."
            )
        self.d_model = int(d_model)
        self.num_modalities = int(num_modalities)
        self.fs = float(fs)
        self.hr_f_min = float(hr_f_min)
        self.hr_f_max = float(hr_f_max)
        self.spectrum_f_min = float(spectrum_f_min)
        self.spectrum_f_max = float(spectrum_f_max)
        self.peak_sigma_hz = float(peak_sigma_hz)
        # 将配置的 SNR 中点从 dB 转为线性功率比，用于质量映射。
        self.snr_midpoint_db = float(snr_midpoint_db)
        self.snr_midpoint = 10.0 ** (self.snr_midpoint_db / 10.0)
        self.eps = float(eps)
        if self.peak_sigma_hz <= 0.0:
            raise ValueError("peak_sigma_hz must be positive.")
        # RGB 和 RF 分别使用自己的 D -> 1 投影，不共享参数。
        self.signal_heads = nn.ModuleList(
            [nn.Linear(d_model, 1) for _ in range(num_modalities)]
        )

    def forward(
        self, modal_features: torch.Tensor, return_details: bool = False
    ):
        if modal_features.ndim != 4:
            raise ValueError(
                "modal_features must be [B,T,M,D], got {}.".format(
                    tuple(modal_features.shape)
                )
            )
        _, frames, modalities, feature_dim = modal_features.shape
        if modalities != self.num_modalities or feature_dim != self.d_model:
            raise ValueError(
                "Expected modal features [B,T,{},{}], got {}.".format(
                    self.num_modalities, self.d_model, tuple(modal_features.shape)
                )
            )
        input_dtype = modal_features.dtype

        # 两个独立线性层：[B,T,M,D] -> RGB/RF 各 [B,T,1] -> [B,M,T]。
        projected_signals = torch.stack(
            [
                head(modal_features[:, :, modality_index, :]).squeeze(-1)
                for modality_index, head in enumerate(self.signal_heads)
            ],
            dim=1,
        )

        # 去直流并加 Hann 窗，减少有限长度 DFT 的频谱泄漏。
        # FFT 在混合精度下统一使用 float32：[B,M,T] -> [B,M,F]。
        signals = projected_signals.float()
        signals = signals - signals.mean(dim=-1, keepdim=True)
        window = torch.hann_window(
            frames, periodic=True, device=signals.device, dtype=signals.dtype
        )
        windowed_signals = signals * window
        spectrum = torch.fft.rfft(windowed_signals, dim=-1)

        # Periodogram 单边 PSD：[B,M,F]；缩放不改变 SNR，但保留标准 PSD 定义。
        window_energy = window.square().sum().clamp_min(self.eps)
        psd = spectrum.abs().square() / (self.fs * window_energy)
        one_sided_scale = torch.ones(
            psd.shape[-1], device=psd.device, dtype=psd.dtype
        )
        if frames % 2 == 0:
            one_sided_scale[1:-1] = 2.0
        else:
            one_sided_scale[1:] = 2.0
        psd = psd * one_sided_scale
        freqs = torch.fft.rfftfreq(frames, d=1.0 / self.fs).to(psd.device)

        spectrum_mask = (freqs >= self.spectrum_f_min) & (freqs <= self.spectrum_f_max)
        heart_mask = (freqs >= self.hr_f_min) & (freqs <= self.hr_f_max)

        # 在心率频带内用 log-PSD softmax 做可微的主峰定位：
        # [B,M,F] -> [B,M,F_hr] -> peak_frequency [B,M]。
        heart_psd = psd[..., heart_mask]
        heart_freqs = freqs[heart_mask]
        log_heart_psd = torch.log(heart_psd + self.eps)
        peak_weights = F.softmax(log_heart_psd, dim=-1)
        peak_frequency = (peak_weights * heart_freqs).sum(dim=-1)

        # 以软峰值为中心，只在心率频带内构造高斯峰值窗；使用 Hz 定义窗宽，不使用bin来定义，更加适合不同clip长度的实验
        # [F_hr] - [B,M,1] -> [B,M,F_hr]。
        frequency_distance = (
            heart_freqs.view(1, 1, -1) - peak_frequency.unsqueeze(-1)
        )
        peak_window = torch.exp(
            -0.5 * (frequency_distance / self.peak_sigma_hz).square()
        )

        # 只把主峰附近功率视为有效信号，其余 0.4-4.0 Hz 功率均视为噪声：
        # SNR = P_peak / (P_valid_band - P_peak)。
        valid_band_power = psd[..., spectrum_mask].sum(dim=-1)
        signal_power = (heart_psd * peak_window).sum(dim=-1)
        noise_power = (valid_band_power - signal_power).clamp_min(self.eps)
        snr = signal_power / noise_power.clamp_min(self.eps)
        peak_energy_ratio = signal_power / valid_band_power.clamp_min(self.eps)

        # RGB/RF 分别使用 q=SNR/(SNR+c)，位于 [0,1]，在 SNR=c 时等于 0.5。
        quality = (
            snr / (snr + self.snr_midpoint)
        ).clamp(0.0, 1.0).to(dtype=input_dtype)

        # 直接使用线性 SNR 的实验方式（保留，便于切换）。
        # quality = snr.to(dtype=input_dtype)
        if not return_details:
            return quality
        return {
            "quality": quality,
            "projected_signals": projected_signals,
            "frequencies": freqs,
            "psd": psd,
            "heart_frequencies": heart_freqs,
            "heart_psd": heart_psd,
            "peak_weights": peak_weights,
            "peak_frequency_hz": peak_frequency,
            "peak_frequency_bpm": peak_frequency * 60.0,
            "peak_window": peak_window,
            "signal_power": signal_power,
            "noise_power": noise_power,
            "snr": snr,
            "snr_db": 10.0 * torch.log10(snr.clamp_min(self.eps)),
            "snr_midpoint_db": psd.new_tensor(self.snr_midpoint_db),
            "peak_energy_ratio": peak_energy_ratio,
        }


class CrossModalFusion(nn.Module):
    """CogMoE cross-modal attention applied independently at every frame."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, modality_embeddings: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(modality_embeddings) < 2:
            raise ValueError("CrossModalFusion requires at least two modalities.")
        reference_shape = modality_embeddings[0].shape
        if any(embedding.shape != reference_shape for embedding in modality_embeddings):
            raise ValueError("All modality embeddings must have the same [B,T,D] shape.")

        batch, frames, d_model = reference_shape
        # [B,T,M,D] -> [B*T,M,D]: attention is along the modality axis.
        tokens = torch.stack(list(modality_embeddings), dim=2)
        tokens = tokens.reshape(batch * frames, len(modality_embeddings), d_model)
        fused, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        fused = fused.mean(dim=1).reshape(batch, frames, d_model)
        return self.norm(fused)


class CrossModalAttentionForCRE(nn.Module):
    """Embedding-level contextual attention used by CogMoE's CRE expert."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        context = x if context is None else context
        attended, _ = self.attn(x, context, context, need_weights=False)
        return self.norm(x + attended)


class HighFidelityExpert(nn.Module):
    """Lightweight FFN expert for relatively clean modality features."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class NoiseResilientExpert(nn.Module):
    """Normalized residual FFN expert for noisy inputs."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class ContextualRefinementExpert(nn.Module):
    """Attention-based expert for contextually refining fused features."""

    def __init__(
        self, d_model: int, d_ff: int, nhead: int, dropout: float = 0.1
    ):
        super().__init__()
        self.cross_attn = CrossModalAttentionForCRE(d_model, nhead, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        refined = self.cross_attn(x)
        return refined + self.ffn(self.norm(refined))


class DynamicPathwayGating(nn.Module):
    """Route every fused frame using its feature and RGB/RF quality vector."""

    def __init__(self, d_model: int, quality_dim: int, num_experts: int = 3):
        super().__init__()
        self.quality_dim = quality_dim
        self.gate = nn.Linear(d_model + quality_dim, num_experts)

    def forward(self, features: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        if quality.ndim != 2 or quality.shape[1] != self.quality_dim:
            raise ValueError(
                "quality must be [B,{}], got {}.".format(
                    self.quality_dim, tuple(quality.shape)
                )
            )
        quality = quality.unsqueeze(1).expand(-1, features.shape[1], -1)
        logits = self.gate(torch.cat([features, quality], dim=-1))
        return F.softmax(logits, dim=-1)


class MoEFeedForward(nn.Module):
    """CogMoE's dense weighted mixture of HFE, NRE and CRE experts."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        nhead: int,
        quality_dim: int = 2,
        num_experts: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_experts != 3:
            raise ValueError("SQGMoE uses exactly three experts: HFE, NRE and CRE.")
        self.experts = nn.ModuleList(
            [
                HighFidelityExpert(d_model, d_ff, dropout),
                NoiseResilientExpert(d_model, d_ff, dropout),
                ContextualRefinementExpert(d_model, d_ff, nhead, dropout),
            ]
        )
        self.gating = DynamicPathwayGating(d_model, quality_dim, num_experts)

    def forward(
        self, x: torch.Tensor, quality: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        gate_weights = self.gating(x, quality)
        expert_outputs = [expert(x) for expert in self.experts]
        stacked = torch.stack(expert_outputs, dim=-1)
        output = (stacked * gate_weights.unsqueeze(2)).sum(dim=-1)
        return output, gate_weights, expert_outputs


class SQGMoELayer(nn.Module):
    """Self-attention followed by a quality-guided MoE feed-forward block."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_ff: int,
        num_experts: int,
        quality_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.moe_ffn = MoEFeedForward(
            d_model, d_ff, nhead, quality_dim, num_experts, dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, quality: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        attended, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attended))
        moe_output, gate_weights, expert_outputs = self.moe_ffn(x, quality)
        x = self.norm2(x + self.dropout(moe_output))
        return x, gate_weights, expert_outputs


class SQGMoETransformer(nn.Module):
    """Stack of CogMoE-style Transformer layers."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_ff: int,
        num_layers: int,
        num_experts: int,
        quality_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SQGMoELayer(
                    d_model,
                    nhead,
                    d_ff,
                    num_experts,
                    quality_dim,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, quality: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[List[torch.Tensor]]]:
        gate_weights_list = []
        expert_outputs_list = []
        for layer in self.layers:
            x, gate_weights, expert_outputs = layer(x, quality)
            gate_weights_list.append(gate_weights)
            expert_outputs_list.append(expert_outputs)
        return self.norm(x), gate_weights_list, expert_outputs_list


class WaveformPredictionHead(nn.Module):
    """Map every fused temporal token to one rPPG waveform sample."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x).squeeze(-1)


class SQGMoE(nn.Module):
    """RGB-RF CogMoE adapted to waveform-based rPPG prediction."""

    # EquiplethLoader 将 (I/Q=2, range_window=5, T) 展平为 (10, T)。
    EQUIPLETH_RF_IN_CH = 10

    def __init__(
        self,
        dim: int = 64,
        d_ff: int = 128,
        num_layers: int = 2,
        num_experts: int = 3,
        nhead: int = 4,
        dropout: float = 0.1,
        quality_fs: float = 30.0,
        quality_hr_range: Tuple[float, float] = (0.7, 3.0),
        quality_spectrum_range: Tuple[float, float] = (0.4, 4.0),
        quality_peak_sigma_hz: float = 0.1,
        quality_snr_midpoint_db: float = 7.5,
        rgb_encoder: str = "RhythmFormer",
    ):
        super().__init__()
        if dim % nhead != 0:
            raise ValueError("dim must be divisible by nhead.")

        self.num_modalities = 2
        rgb_encoders = {
            "RhythmFormer": RhythmFormerEncoder,
            "PhysNet": PhysNetEncoder,
        }
        if rgb_encoder not in rgb_encoders:
            raise ValueError("rgb_encoder must be 'RhythmFormer' or 'PhysNet'.")
        RGBEncoder = rgb_encoders[rgb_encoder]
        self.rgb_encoder = RGBEncoder(d_model=dim)
        self.rf_encoder = FusionPhysRFEncoder(
            in_ch=self.EQUIPLETH_RF_IN_CH,
            d_model=dim,
        )
        self.quality_estimator = DFTSNRQualityEstimator(
            d_model=dim,
            num_modalities=self.num_modalities,
            fs=quality_fs,
            hr_f_min=quality_hr_range[0],
            hr_f_max=quality_hr_range[1],
            spectrum_f_min=quality_spectrum_range[0],
            spectrum_f_max=quality_spectrum_range[1],
            peak_sigma_hz=quality_peak_sigma_hz,
            snr_midpoint_db=quality_snr_midpoint_db,
        )
        self.fusion = CrossModalFusion(dim, nhead, dropout)
        self.transformer = SQGMoETransformer(
            dim,
            nhead,
            d_ff,
            num_layers,
            num_experts,
            self.num_modalities,
            dropout,
        )
        self.predictor = WaveformPredictionHead(dim, dropout)

    def forward(
        self,
        rgb: torch.Tensor,
        rf: torch.Tensor,
        return_details: bool = False,
    ):
        rgb_features = self.rgb_encoder(rgb)
        rf_features = self.rf_encoder(rf)
        modal_features = torch.stack([rgb_features, rf_features], dim=2)

        # [B,T,2,D] -> 两个 Linear(D,1) -> DFT -> PSD -> SNR -> [B,2]。
        quality_result = self.quality_estimator(
            modal_features, return_details=return_details
        )
        if return_details:
            quality_details = quality_result
            quality = quality_details["quality"]
        else:
            quality_details = None
            quality = quality_result

        fused_features = self.fusion([rgb_features, rf_features])
        transformed, gate_weights_list, expert_outputs_list = self.transformer(
            fused_features, quality
        )
        prediction = self.predictor(transformed)

        if not return_details:
            return prediction
        return {
            "pred": prediction,
            "rgb_features": rgb_features,
            "rf_features": rf_features,
            "fused_features": fused_features,
            "quality": quality,
            "quality_details": quality_details,
            "gate_weights_list": gate_weights_list,
            "expert_outputs_list": expert_outputs_list,
        }


def build_model(**kwargs) -> SQGMoE:
    """Construct SQGMoE from keyword arguments."""
    return SQGMoE(**kwargs)
