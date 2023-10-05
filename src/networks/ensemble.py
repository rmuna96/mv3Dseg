import torch
import torch.nn as nn

from utils import sliding_window_inference


class EnsembleModel(nn.Module):
    """
        Given a list of models this class implements the ensembling of
        these models by staking them together and taking the maximum
        predictions along each detected class.
    """
    def __init__(self, models, outchannels, roi_size=(160, 160, 160), sw_batch_size=4):
        super().__init__()
        self.models = models
        self.outchannels = outchannels
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size

    def forward(self, x):
        nfold = len(self.models)
        labels = [torch.empty_like(x).expand(-1, nfold, -1, -1, -1) for _ in range(self.outchannels)]

        for m, model in enumerate(self.models):
            out = sliding_window_inference(x, self.roi_size, self.sw_batch_size, model)
            for l, label in enumerate(labels):

                index = torch.ones_like(x) * m
                labels[l] = label.scatter(1, index, out.narrow(1, l, 1))

        out_e = torch.empty_like(x).expand(-1, self.outchannels, -1, -1, -1)

        for l, label in enumerate(labels):
            index = torch.ones_like(x) * l
            ensemble = torch.max(label, dim=1, keepdim=True)
            out_e = out_e.scatter(1, index, ensemble)

        return out_e
