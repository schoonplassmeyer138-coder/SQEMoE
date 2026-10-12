import torch


class Neg_Pearson2(torch.nn.Module):
    """Squared negative Pearson correlation loss for rPPG waveforms."""

    def __init__(self):
        super(Neg_Pearson2, self).__init__()
        self.epsilon = 1e-2

    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calculate squared negative Pearson loss over each sample's temporal axis.

        Args:
            preds: Model predictions with shape [batch, temporal].
            labels: Ground-truth signals with shape [batch, temporal].

        Returns:
            Mean squared negative Pearson loss over the batch.
        """
        loss = 0
        for i in range(preds.shape[0]):
            x = normalize_signal(preds[i])
            y = normalize_signal(labels[i])

            sum_x = torch.sum(x)
            sum_y = torch.sum(y)
            sum_xy = torch.sum(x * y)
            sum_x2 = torch.sum(torch.pow(x, 2))
            sum_y2 = torch.sum(torch.pow(y, 2))
            n = preds.shape[1]
            pearson = (n * sum_xy - sum_x * sum_y) / torch.sqrt(
                (n * sum_x2 - torch.pow(sum_x, 2) + self.epsilon)
                * (n * sum_y2 - torch.pow(sum_y, 2) + self.epsilon)
            )

            loss += (1 - pearson) ** 2

        return loss / preds.shape[0]


def normalize_signal(sig: torch.Tensor) -> torch.Tensor:
    return (sig - torch.mean(sig)) / (torch.std(sig) + 1.0e-6)
