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


class ListNetLoss(nn.Module):
    def __init__(self):
        super(ListNetLoss, self).__init__()

    def forward(self, predictions, ground_truth_probs):
        """
        Compute ListNet loss.

        Args:
            predictions (torch.Tensor): Predicted probabilities for ranking order.
            ground_truth_probs (torch.Tensor): Ground truth probabilities for the correct ranking order.

        Returns:
            torch.Tensor: ListNet loss.
        """
        return -torch.sum(ground_truth_probs * torch.log(predictions))


class AdaRankLoss(nn.Module):
    def __init__(self):
        super(AdaRankLoss, self).__init__()

    def forward(self, predicted_scores, ground_truth_labels, weights):
        """
        Compute List-wise AdaRank loss.

        Args:
            predicted_scores (torch.Tensor): Predicted relevance scores.
            ground_truth_labels (torch.Tensor): Ground truth relevance labels.
            weights (torch.Tensor): Weights for individual documents.

        Returns:
            torch.Tensor: List-wise AdaRank loss.
        """
        return torch.sum(weights * torch.relu(1.0 - (ground_truth_labels - predicted_scores)))


'''loss_fn = LambdaMARTLoss()
a = torch.tensor([[2.0, 1.3, 7.7, 4, 9, 0.3]])
b = torch.tensor([[2.0, 1.3, 7.7, 4, 9, 0.3]])
w = torch.tensor([[1, 1, 1, 1, 1, 1]])
# b = torch.flip(b, dims=(0,1))
print(loss_fn(a, b))
loss_fn = ListNetLoss()
print(loss_fn(a, b))
loss_fn = AdaRankLoss()
print(loss_fn(a, b, w))'''

