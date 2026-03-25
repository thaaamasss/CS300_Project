import torch
import torch.nn as nn


class SISAEnsemble(nn.Module):

    def __init__(self, shard_models, shard_indices=None,
                 shard_checkpoints=None, num_slices=1):
        super(SISAEnsemble, self).__init__()

        self.shard_models = nn.ModuleList(shard_models)
        self.shard_indices = shard_indices
        self.shard_checkpoints = shard_checkpoints   # None if num_slices=1
        self.num_slices = num_slices

    def forward(self, x):
        outputs_sum = None
        for model in self.shard_models:
            outputs = model(x)
            if outputs_sum is None:
                outputs_sum = outputs
            else:
                outputs_sum += outputs
        return outputs_sum / len(self.shard_models)