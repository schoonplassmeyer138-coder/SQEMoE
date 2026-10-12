import torch


class SNRLoss_dB_Signals(torch.nn.Module):
    def __init__(self, N=1024, pulse_band=None):
        super(SNRLoss_dB_Signals, self).__init__()
        self.N = N
        if pulse_band is None:
            pulse_band = [45 / 60.0, 150 / 60.0]
        self.register_buffer("pulse_band", torch.tensor(pulse_band, dtype=torch.float32))

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, Fs=30):
        device = outputs.device
        pulse_band = self.pulse_band.to(device)
        if not outputs.is_cuda:
            torch.backends.mkl.is_available()
        n_samp = outputs.shape[-1]
        wind_sz = int(self.N / 256)

        f = torch.linspace(0, Fs / 2, int(self.N / 2) + 1, dtype=torch.float32).to(device)

        min_idx = torch.argmin(torch.abs(f - pulse_band[0]))
        max_idx = torch.argmin(torch.abs(f - pulse_band[1]))
        min_idx_int = int(min_idx.item())
        max_idx_int = int(max_idx.item())

        outputs = outputs.view(-1, n_samp)
        targets = targets.view(-1, n_samp)

        y = torch.fft.rfft(targets, n=self.N, dim=1, norm="forward")
        y2 = torch.abs(y) ** 2
        hr_idxs = torch.argmax(y2[:, min_idx_int:max_idx_int], axis=1) + min_idx_int

        x = torch.fft.rfft(outputs, n=self.N, dim=1, norm="forward")
        p1 = torch.abs(x) ** 2

        losses = []
        for count, ref_idx in enumerate(hr_idxs):
            ref_idx = int(ref_idx.item())
            base_start = max(min_idx_int, ref_idx - wind_sz)
            base_end = min(max_idx_int, ref_idx + wind_sz + 1)
            pulse_freq_amp = torch.sum(p1[count, base_start:base_end])
            other_avrg = torch.sum(p1[count, min_idx_int:base_start]) + torch.sum(p1[count, base_end:max_idx_int])
            losses.append(-10 * torch.log10((pulse_freq_amp + 1e-7) / (other_avrg + 1e-7)))

        return torch.stack(losses).mean()


class SNRLossOnPreComputedAndWindowedFFT_base(torch.nn.Module):
    def __init__(self, start_idx, window_fraction=0.02, device=torch.device("cpu")):
        super(SNRLossOnPreComputedAndWindowedFFT_base, self).__init__()
        self.start_idx = start_idx
        self.window_fraction = window_fraction
        self.device = device

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        assert outputs.shape == targets.shape, "The predicted output and the target labels have different shapes"
        if not outputs.is_cuda:
            torch.backends.mkl.is_available()

        window_size_base_harmonic = int(self.window_fraction * targets.shape[1]) + 1

        y2 = torch.abs(targets) ** 2
        hr_idxs = torch.argmax(y2, axis=1)

        x2 = torch.abs(outputs) ** 2

        losses = []
        for count, ref_idx in enumerate(hr_idxs):
            ref_idx = int(ref_idx.item())
            base_start = max(0, ref_idx - window_size_base_harmonic)
            base_end = min(x2.shape[1], ref_idx + window_size_base_harmonic + 1)
            pulse_freq_amp = torch.sum(x2[count, base_start:base_end])
            other_avrg = torch.sum(x2[count, :base_start]) + torch.sum(x2[count, base_end:])
            losses.append(-10 * torch.log10((pulse_freq_amp + 1e-7) / (other_avrg + 1e-7)))

        return torch.stack(losses).mean()
