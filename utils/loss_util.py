import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class bce_loss(nn.Module):
    def __init__(self, weights=None):
        super(bce_loss, self).__init__()
        self.weights = weights

    def forward(self, pred, label):
        if self.weights is not None:
            fore_weights = torch.tensor(self.weights[0])
            back_weights = self.weights[1]

            fore_weights = fore_weights.to('cuda')
            back_weights = back_weights.to('cuda')
            weights = label * back_weights + (1.0 - label) * fore_weights
        else:
            weights = torch.ones(label.shape)
            weights = weights.to('cuda')

        loss = F.binary_cross_entropy_with_logits(pred, label, weights, reduction='none')
        return loss


class NTXentLoss_atom(nn.Module):
    def __init__(self, t=0.1):
        super(NTXentLoss_atom, self).__init__()
        self.T = t
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss(ignore_index=-1)

    def forward(self, out, out_mask, labels):
        out = nn.functional.normalize(out, dim=-1)
        out_mask = nn.functional.normalize(out_mask, dim=-1)

        logits = torch.matmul(out_mask, out.permute(0, 2, 1))
        logits /= self.T

        softmaxs = self.softmax(logits)
        loss = self.criterion(softmaxs.transpose(1, 2), labels)

        return loss, logits


class NTXentLoss(torch.nn.Module):

    def __init__(self, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to('cuda')

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, N, C)
        # v shape: (N, N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to('cuda').long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)
