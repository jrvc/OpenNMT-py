### Taken from NVIDIA/NeMo.
### NVIDIA/NeMo is licenced under Apache 2.0
#https://github.com/NVIDIA/NeMo/blob/master/nemo/collections/asr/parts/spectr_augment.py
# This file is thus under the Apache 2.0 licence, 
# see: https://github.com/NVIDIA/NeMo/blob/master/LICENSE

import random

import torch
import torch.nn as nn


class SpecAugment(nn.Module):
    """
    Zeroes out(cuts) random continuous horisontal or
    vertical segments of the spectrogram as described in
    SpecAugment (https://arxiv.org/abs/1904.08779).
    params:
    freq_masks - how many frequency segments should be cut
    time_masks - how many time segments should be cut
    freq_width - maximum number of frequencies to be cut in one segment
    time_width - maximum number of time steps to be cut in one segment
    """

    def __init__(
        self, freq_masks=2, time_masks=2, freq_width=27, time_width=100, rng=None,
    ):
        super(SpecAugment, self).__init__()

        self._rng = random.Random() if rng is None else rng

        self.freq_masks = freq_masks
        self.time_masks = time_masks

        self.freq_width = freq_width
        self.time_width = time_width

    @torch.no_grad()
    def forward(self, x):
        if self.training:
            return x
        sh = x.shape

        mask = torch.zeros(x.shape).byte()

        for idx in range(sh[0]):
            for i in range(self.freq_masks):
                x_left = int(self._rng.uniform(0, sh[1] - self.freq_width))

                w = int(self._rng.uniform(0, self.freq_width))

                mask[idx, x_left : x_left + w, :] = 1

            for i in range(self.time_masks):
                y_left = int(self._rng.uniform(0, sh[2] - self.time_width))

                w = int(self._rng.uniform(0, self.time_width))

                mask[idx, :, y_left : y_left + w] = 1

        x = x.masked_fill(mask.type(torch.bool).to(device=x.device), 0)

        return x

