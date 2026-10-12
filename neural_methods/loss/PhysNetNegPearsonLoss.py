import torch
from torch import nn


class Neg_Pearson(nn.Module):
    """
    Negative Pearson correlation loss over each sample's temporal axis.

    The class name is kept for compatibility with the original PhysNet trainer.
    """

    def __init__(self, epsilon=1e-6):
        super(Neg_Pearson, self).__init__()
        self.epsilon = epsilon

    def forward(self, preds, labels):
        loss = 0
        for i in range(preds.shape[0]):
            x = preds[i]
            y = labels[i]
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
            loss += 1 - pearson

        return loss / preds.shape[0]
