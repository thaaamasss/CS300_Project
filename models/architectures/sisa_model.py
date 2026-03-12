import torch
import torch.nn as nn


class SISAEnsemble(nn.Module):

    def __init__(self, shard_models, shard_indices=None):
        """
        Args:
            shard_models  : list of trained CNNModel instances (one per shard)
            shard_indices : list of lists — shard_indices[i] contains the
                            original dataset indices that shard i was trained on.
                            Required for true SISA unlearning (identify affected shards).
                            Set by sisa_training automatically.
        """
        super(SISAEnsemble, self).__init__()

        self.shard_models = nn.ModuleList(shard_models)

        # Store shard assignments so sisa_unlearning can identify affected shards.
        # Not an nn.Parameter — just metadata. Survives deepcopy correctly.
        self.shard_indices = shard_indices

    def forward(self, x):

        outputs_sum = None

        for model in self.shard_models:

            outputs = model(x)

            if outputs_sum is None:
                outputs_sum = outputs
            else:
                outputs_sum += outputs

        outputs_avg = outputs_sum / len(self.shard_models)

        return outputs_avg
