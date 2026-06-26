"""Bucketed weighted sampler for Stage 2 task-type mixing."""

import torch
from torch.utils.data import Sampler


class BucketedDistributedSampler(Sampler):
    def __init__(self, bucket_lengths, bucket_offsets, weights, num_microbatches,
                 num_replicas, rank, seed=42):
        assert len(bucket_lengths) == len(bucket_offsets) == len(weights)
        self.bucket_lengths = list(bucket_lengths)
        self.bucket_offsets = list(bucket_offsets)
        self.weights = torch.tensor(list(weights), dtype=torch.double)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.num_per_rank = num_microbatches // num_replicas
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_per_rank

    def _gen(self, *offsets):
        s = self.seed + self.epoch * 1_000_003 + self.rank * 97
        for o in offsets:
            s = s * 1_000_003 + int(o)
        g = torch.Generator()
        g.manual_seed(s & 0x7FFFFFFFFFFFFFFF)
        return g

    def _permutation(self, bucket_idx, cycle):
        return torch.randperm(self.bucket_lengths[bucket_idx],
                              generator=self._gen(bucket_idx, cycle)).tolist()

    def __iter__(self):
        bg = self._gen(-1)   # bucket-choice RNG, distinct from permutation seeds
        B = len(self.bucket_lengths)
        cycle = [0] * B
        perm = [self._permutation(b, 0) for b in range(B)]
        cursor = [0] * B
        for _ in range(self.num_per_rank):
            b = int(torch.multinomial(self.weights, 1, generator=bg).item())
            if cursor[b] >= len(perm[b]):
                cycle[b] += 1
                perm[b] = self._permutation(b, cycle[b])
                cursor[b] = 0
            yield self.bucket_offsets[b] + perm[b][cursor[b]]
            cursor[b] += 1
