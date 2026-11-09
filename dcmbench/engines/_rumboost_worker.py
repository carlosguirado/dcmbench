"""
Worker process for RUMBoost.

LightGBM (inside RUMBoost) and PyTorch each load their own OpenMP runtime; on
macOS, using both in one process crashes or deadlocks. Each fitted RUMBoost
model therefore lives in its own subprocess, which imports only numpy,
pandas, lightgbm and rumboost (lightgbm first). It is started as
``python -c "from dcmbench.engines._rumboost_worker import main; main()"``,
so the parent's script is not re-imported.

Protocol: pickled ``(command, payload)`` requests on stdin, pickled
``(status, value)`` replies on a private copy of stdout. The real stdout is
redirected to stderr, so library prints cannot corrupt the protocol.
"""

from __future__ import annotations

import os
import pickle
import sys
import traceback


def main() -> None:
    replies = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)  # anything printed from here on goes to stderr
    requests = sys.stdin.buffer

    import lightgbm  # before rumboost, which imports torch
    import numpy as np
    import pandas as pd
    from rumboost.rumboost import rum_train

    booster = None

    def dataset(columns, label):
        frame = pd.DataFrame(columns)  # dict of 1-D arrays: column by column (RUMBoost reshapes with order="A")
        return lightgbm.Dataset(frame, label=np.zeros(len(frame)) if label is None else label, free_raw_data=False)

    def reply(status, value):
        pickle.dump((status, value), replies, protocol=pickle.HIGHEST_PROTOCOL)
        replies.flush()

    while True:
        try:
            command, payload = pickle.load(requests)
        except EOFError:
            return
        try:
            if command == "fit":
                np.random.seed(payload["seed"])
                train = dataset(payload["train"], payload["train_label"])
                valid = None if payload["valid"] is None else [dataset(payload["valid"], payload["valid_label"])]
                booster = rum_train(train, payload["specification"], valid_sets=valid)
                reply("ok", int(booster.best_iteration))
            elif command == "predict":
                probabilities = booster.predict(dataset(payload, None), num_iteration=booster.best_iteration)
                reply("ok", np.asarray(probabilities, dtype=float))
            elif command == "close":
                reply("ok", None)
                return
            else:
                reply("error", f"unknown command {command!r}")
        except Exception:  # sent back and re-raised in the parent
            reply("error", traceback.format_exc())
