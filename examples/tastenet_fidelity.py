"""
Fidelity check: TasteNet-MNL on Swissmetro under the original paper's protocol.

Han et al. (2022, Table 12) report a test NLL per observation of 0.645 for the
best TasteNet run and 0.652 (sd 0.008) over 100 runs; MNL-C gives 0.698.
This script reproduces their setup as found in the official code
(github.com/deborahmit/TasteNet-MNL):

- sample: all 10,692 observations after dropping AGE = 6, PURPOSE = 9 and unknown choices;
  PURPOSE 5-8 recoded to 1-4;
- utilities with travel time, headway and seats, cost fixed at -1 (willingness-to-pay space),
  times and costs divided by 100, cost zero for GA holders;
- tastes from 17 characteristic dummies via Linear(17, 110) -> ReLU -> Linear(110, 8),
  time and headway coefficients kept negative by -exp(-raw);
- split 70/15/15 by observation (sklearn train_test_split, random_state 8 then 9);
- Adam, learning rate 1e-3, batch size 32, at most 100 epochs, best-dev checkpoint.

    python examples/tastenet_fidelity.py [--seeds 5]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn

import dcmbench.data
from dcmbench.engines.torch_module import TorchModuleFitter, Training

DUMMIES = {"MALE": (1,), "AGE": (2, 3, 4, 5), "INCOME": (2, 3, 4), "FIRST": (1,), "WHO": (2, 3),
           "PURPOSE_RECODED": (2, 3, 4), "LUGGAGE": (1, 3), "GA": (1,)}
Z = [f"{c}_{v}" for c, values in DUMMIES.items() for v in values]
COLUMNS = ["TRAIN_TT", "TRAIN_HE", "TRAIN_CO", "SM_TT", "SM_HE", "SM_SEATS", "SM_CO", "CAR_TT", "CAR_CO", "GA"] + Z


class OriginalTasteNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(len(Z), 110), nn.ReLU(), nn.Linear(110, 8))

    def forward(self, x, available):
        raw = self.net(torch.stack([x[c] for c in Z], dim=1))
        negative = -torch.exp(-raw[:, :5])  # TT train, TT sm, TT car, HE train, HE sm
        seats, asc_train, asc_sm = raw[:, 5], raw[:, 6], raw[:, 7]
        paid = 1 - x["GA"]
        v_train = asc_train + negative[:, 0] * x["TRAIN_TT"] / 100 + negative[:, 3] * x["TRAIN_HE"] / 100 \
            - x["TRAIN_CO"] * paid / 100
        v_sm = asc_sm + negative[:, 1] * x["SM_TT"] / 100 + negative[:, 4] * x["SM_HE"] / 100 \
            + seats * x["SM_SEATS"] - x["SM_CO"] * paid / 100
        v_car = negative[:, 2] * x["CAR_TT"] / 100 - x["CAR_CO"] / 100
        return torch.stack([v_train, v_sm, v_car], dim=1)


def paper_sample():
    dataset = dcmbench.data.load("swissmetro")
    dataset = dataset.filter("AGE != 6 and PURPOSE != 9", label="drop unknown age and 'other' purpose (TasteNet paper)")
    dataset = dataset.derive("PURPOSE 5-8 recoded to 1-4", PURPOSE_RECODED=lambda f: f.PURPOSE.where(f.PURPOSE <= 4, f.PURPOSE - 4))
    source = {"PURPOSE_RECODED": "PURPOSE_RECODED"}
    return dataset.derive("characteristic dummies", **{
        f"{c}_{v}": (lambda f, c=c, v=v: (f[source.get(c, c)] == v).astype(int)) for c, values in DUMMIES.items() for v in values
    })


def nll_per_observation(model, frame, problem):
    p = model.predict_probabilities(frame).to_numpy()
    chosen = problem.choices(frame).map({a: i for i, a in enumerate(problem.alternatives)}).to_numpy()
    return float(-np.log(p[np.arange(len(frame)), chosen]).mean()), float((p.argmax(axis=1) == chosen).mean())


def main():
    parser = argparse.ArgumentParser(description="TasteNet-MNL fidelity check on Swissmetro")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    dataset = paper_sample()
    frame, problem = dataset.frame, dataset.problem
    train, rest = train_test_split(frame, test_size=0.3, random_state=8)
    dev, test = train_test_split(rest, test_size=0.5, random_state=9)
    print(f"sample {len(frame)} (paper: 10,692); train/dev/test {len(train)}/{len(dev)}/{len(test)} (paper: 7,484/1,604/1,604)")

    training = Training(optimizer="adam", learning_rate=1e-3, batch_size=32, max_epochs=100, patience=5)
    results = []
    for seed in range(args.seeds):
        fitter = TorchModuleFitter(OriginalTasteNet, problem, COLUMNS, training=training, seed=seed, name="TasteNet (original)")
        model = fitter.fit(train, dev)
        test_nll, test_acc = nll_per_observation(model, test, problem)
        results.append((test_nll, test_acc))
        print(f"seed {seed}: test NLL {test_nll:.3f}, accuracy {test_acc:.3f}, best epoch {model.report['best_epoch']}", flush=True)
    nll = np.array([r[0] for r in results])
    print(f"test NLL: best {nll.min():.3f}, mean {nll.mean():.3f} (sd {nll.std():.3f}) over {len(nll)} seeds; "
          f"paper: best 0.645, mean 0.652 (sd 0.008) over 100 runs")


if __name__ == "__main__":
    main()
