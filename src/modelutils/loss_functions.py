import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import ndcg_score
import torch.nn.functional as f


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


class LambdaLoss(nn.Module):
    def __init__(self):
        super(LambdaLoss, self).__init__()

    def forward(self, pred_scores, true_scores):
        # Sort the scores
        _, pred_sorted_indices = torch.sort(pred_scores, dim=1, descending=True)
        _, true_sorted_indices = torch.sort(true_scores, dim=1, descending=True)

        # Compute the rank difference
        pred_sorted_ranks = torch.argsort(pred_sorted_indices)
        true_sorted_ranks = torch.argsort(true_sorted_indices)
        rank_diff = pred_sorted_ranks - true_sorted_ranks

        # Compute the absolute difference in true scores
        abs_diff_true_scores = torch.abs(true_scores[:, None] - true_scores)

        # Compute the lambda loss
        lambda_loss = torch.sum(rank_diff * abs_diff_true_scores)

        return lambda_loss


class GPTListNetLoss(nn.Module):
    def __init__(self):
        super(GPTListNetLoss, self).__init__()

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


class MyListNetLoss(nn.Module):
    def __init__(self):
        super(MyListNetLoss, self).__init__()

    def forward(self, predictions, ground_truth_probs):
        """
        Compute ListNet loss.

        Args:
            predictions (torch.Tensor): Predicted probabilities for ranking order.
            ground_truth_probs (torch.Tensor): Ground truth probabilities for the correct ranking order.

        Returns:
            torch.Tensor: ListNet loss.
        """
        preds = torch.tensor([])
        for j in range(len(predictions[0])):
            prob_of_top1 = torch.exp(predictions[0][j]) / torch.sum(torch.exp(predictions)).reshape(-1, 1)
            preds = torch.cat((preds.to('cuda'), prob_of_top1.to('cuda')), 1).to('cuda')
        # top1_true_position = torch.argmax(ground_truth_probs)
        loss = torch.nn.CrossEntropyLoss()
        output = loss(preds, ground_truth_probs.to('cuda'))
        return output


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


class LambdaLossLTR(nn.Module):
    def __init__(self):
        super(LambdaLossLTR, self).__init__()

    def forward(self, y_pred, y_true, eps=1e-10, weighing_scheme=None, k=None,
                sigma=1., mu=10.,
                reduction="sum", reduction_log="binary"):
        """
        LambdaLoss framework for LTR losses implementations, introduced in "The LambdaLoss Framework for Ranking Metric Optimization".
        Contains implementations of different weighing schemes corresponding to e.g. LambdaRank or RankNet.
        :param y_pred: predictions from the model, shape [batch_size, slate_length]
        :param y_true: ground truth labels, shape [batch_size, slate_length]
        :param eps: epsilon value, used for numerical stability
        :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
        :param weighing_scheme: a string corresponding to a name of one of the weighing schemes
        :param k: rank at which the loss is truncated
        :param sigma: score difference weight used in the sigmoid function
        :param mu: optional weight used in NDCGLoss2++ weighing scheme
        :param reduction: losses reduction method, could be either a sum or a mean
        :param reduction_log: logarithm variant used prior to masking and loss reduction, either binary or natural
        :return: loss value, a torch.Tensor
        """
        device = y_pred.device
        y_pred = y_pred.clone()
        y_true = y_true.clone()

        # y_pred[padded_mask] = float("-inf")
        # y_true[padded_mask] = float("-inf")

        # Here we sort the true and predicted relevancy scores.
        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

        # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)

        if weighing_scheme != "ndcgLoss1_scheme":
            padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

        ndcg_at_k_mask = torch.zeros((y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
        ndcg_at_k_mask[:k, :k] = 1

        # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
        true_sorted_by_preds.clamp_(min=0.)
        y_true_sorted.clamp_(min=0.)

        # Here we find the gains, discounts and ideal DCGs per slate.
        pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
        D = torch.log2(1. + pos_idxs.float())[None, :]
        maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(min=eps)
        G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

        # Here we apply appropriate weighing scheme - ndcgLoss1, ndcgLoss2, ndcgLoss2++ or no weights (=1.0)
        if weighing_scheme is None:
            weights = 1.
        else:
            weights = globals()[weighing_scheme](G, D, mu, true_sorted_by_preds)  # type: ignore

        # We are clamping the array entries to maintain correct backprop (log(0) and division by 0)
        scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
        scores_diffs.masked_fill(torch.isnan(scores_diffs), 0.)
        weighted_probas = (torch.sigmoid(sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)
        if reduction_log == "natural":
            losses = torch.log(weighted_probas)
        elif reduction_log == "binary":
            losses = torch.log2(weighted_probas)
        else:
            raise ValueError("Reduction logarithm base can be either natural or binary")

        if reduction == "sum":
            loss = -torch.sum(losses[padded_pairs_mask & ndcg_at_k_mask])
        elif reduction == "mean":
            loss = -torch.mean(losses[padded_pairs_mask & ndcg_at_k_mask])
        else:
            raise ValueError("Reduction method can be either sum or mean")

        return loss


    def ndcgLoss1_scheme(self, G, D, *args):
        return (G / D)[:, :, None]

    def ndcgLoss2_scheme(self, G, D, *args):
        pos_idxs = torch.arange(1, G.shape[1] + 1, device=G.device)
        delta_idxs = torch.abs(pos_idxs[:, None] - pos_idxs[None, :])
        deltas = torch.abs(torch.pow(torch.abs(D[0, delta_idxs - 1]), -1.) - torch.pow(torch.abs(D[0, delta_idxs]), -1.))
        deltas.diagonal().zero_()

        return deltas[None, :, :] * torch.abs(G[:, :, None] - G[:, None, :])

    def lambdaRank_scheme(self, G, D, *args):
        return torch.abs(torch.pow(D[:, :, None], -1.) - torch.pow(D[:, None, :], -1.)) * torch.abs(
            G[:, :, None] - G[:, None, :])

    def ndcgLoss2PP_scheme(self, G, D, *args):
        return args[0] * ndcgLoss2_scheme(G, D) + lambdaRank_scheme(G, D)

    def rankNet_scheme(self, G, D, *args):
        return 1.

    def rankNetWeightedByGTDiff_scheme(self, G, D, *args):
        return torch.abs(args[1][:, :, None] - args[1][:, None, :])

    def rankNetWeightedByGTDiffPowed_scheme(self, G, D, *args):
        return torch.abs(torch.pow(args[1][:, :, None], 2) - torch.pow(args[1][:, None, :], 2))

class ListOneLoss(nn.Module):
    def __init__(self, M=1):
        super(ListOneLoss, self).__init__()
        self.M = M

    def forward(self, y_pred, y_true):
        pred_max = f.softmax(y_pred/self.M, dim=0) + 1e-9
        true_max = f.softmax(-y_true/self.M, dim=0)  # need to reverse the sign
        pred_log = torch.log(pred_max)
        return torch.mean(-torch.sum(true_max*pred_log))


class ListAllLoss(nn.Module):
    def __init__(self, M=0.5):
        super(ListAllLoss, self).__init__()
        self.M = M

    def forward(self, y_pred, y_label):
        pred_max = f.softmax(y_pred/self.M, dim=1) + 1e-9
        pred_log = torch.log(pred_max)
        return torch.mean(-torch.sum(y_label*pred_log))


'''loss_fn = LambdaLossLTR()
a = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0]])
b = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0]])
# b = torch.flip(b, dims=(0,1))
print(loss_fn(a, b))
print(ndcg_score(a, b))
'''