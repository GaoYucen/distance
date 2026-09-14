"""R4C execution fix: pairwise IQE-maxmean alpha broadcasting only.

The frozen scientific protocol, data, optimizer, seeds, budgets and selection rule are
unchanged. The first R4C attempt stopped before IQE training because the R4B
full-matrix reduction reshaped alpha for a [seed,n,n] output, while R4C pairwise
outputs are [seed,batch]. This wrapper fixes only that shape and delegates all
other code to r4c_realroads.py.
"""
from __future__ import annotations
import argparse
import torch
import scripts.r4c_realroads as base


class PairSeedBatchFixed(base.PairSeedBatch):
    def forward(self, ids: torch.Tensor, return_components=False):
        x = self.table[:, ids[:, 0], :]
        y = self.table[:, ids[:, 1], :]
        c = base.components(x, y, self.mode)
        if self.mode == 'IQE-maxmean':
            alpha = self.raw_alpha.sigmoid()[:, None]
            out = torch.lerp(c.mean(-1), c.max(-1).values, alpha)
        elif self.mode == 'IQE-sum':
            out = c.sum(-1)
        else:
            out = c.max(-1).values.relu()
        out = out * self.calibration[:, None]
        return (out, c) if return_components else out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train', 'replay'])
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    base.PairSeedBatch = PairSeedBatchFixed
    if args.action == 'train':
        base.train_all(args.device)
    else:
        base.replay_all()


if __name__ == '__main__':
    main()
