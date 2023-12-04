import torch
import torch.nn as nn
import torch.nn.functional as F


class LambdaMARTLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super(LambdaMARTLoss, self).__init__()
        self.sigma = sigma

    def forward(self, scores, target):
        """
        Compute LambdaMART loss.

        Parameters:
            - scores: Predicted scores for the items in each list.
            - target: Ground truth ranking order for the lists.

        Returns:
            - loss: LambdaMART loss.
        """
        # Calculate the pairwise differences in scores
        pairwise_diff = scores[:, None, :] - scores[:, :, None]

        # Calculate the target differences
        target_diff = target[:, None, :] - target[:, :, None]

        # Compute the LambdaMART loss
        loss = torch.log(1 + torch.exp(-self.sigma * pairwise_diff * target_diff))

        # Sum over all pairs and average over all lists
        loss = loss.sum() / (2 * target.numel())

        return loss
