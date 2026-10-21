"""
Fidelity check: RUMBoost on the full London LPMC data under the original paper's protocol.

Salvadé and Hillel (arXiv:2401.11954, Table 4) report a test cross-entropy of
0.6737 for RUMBoost-GBUV and 0.7085 for MNL, training on survey years 1-2
(54,766 trips) and testing on year 3 (26,320 trips). Variables, monotone
constraints and hyperparameters (learning rate 0.1, depth-1 trees, 1,300
rounds) follow the paper and its example notebook.

    PYTHONPATH=third_party/rumboost python examples/rumboost_fidelity.py
"""

from __future__ import annotations

import time

import numpy as np

import dcmbench.data
from dcmbench.engines.rumboost import RumboostFitter

COMMON = ["age", "female", "day_of_week", "start_time_linear", "car_ownership", "driving_license",
          "purpose_B", "purpose_HBE", "purpose_HBO", "purpose_HBW", "purpose_NHBO",
          "fueltype_Average", "fueltype_Diesel", "fueltype_Hybrid", "fueltype_Petrol", "distance_km"]
VARIABLES = {
    "walk": ["dur_walking"] + COMMON,
    "cycle": ["dur_cycling"] + COMMON,
    "pt": ["dur_pt_access", "dur_pt_bus", "dur_pt_rail", "dur_pt_int_waiting", "dur_pt_int_walking",
           "pt_n_interchanges", "cost_transit"] + COMMON,
    "drive": ["dur_driving", "cost_driving_fuel", "cost_driving_con_charge", "driving_traffic_percent"] + COMMON,
}
DECREASING = ["dur_walking", "dur_cycling", "dur_pt_access", "dur_pt_bus", "dur_pt_rail", "dur_pt_int_waiting",
              "dur_pt_int_walking", "cost_transit", "dur_driving", "cost_driving_fuel", "cost_driving_con_charge",
              "distance_km"]


def full_lpmc():
    dataset = dcmbench.data.load("lpmc")
    purposes = {f"purpose_{p}": (lambda f, p=p: (f.purpose == p).astype(int)) for p in ("B", "HBE", "HBO", "HBW", "NHBO")}
    fuels = {f"fueltype_{k}": (lambda f, k=k: f.fueltype.str.startswith(k).astype(int))
             for k in ("Average", "Diesel", "Hybrid", "Petrol")}
    return dataset.derive("purpose and fuel type dummies; distance in km",
                          distance_km=lambda f: f.distance / 1000, **purposes, **fuels)


def main():
    dataset = full_lpmc()
    frame, problem = dataset.frame, dataset.problem
    train, test = frame[frame.survey_year < 3], frame[frame.survey_year == 3]
    print(f"train {len(train)} (paper: 54,766), test {len(test)} (paper: 26,320)", flush=True)
    fitter = RumboostFitter(problem, VARIABLES, {c: -1 for c in DECREASING}, learning_rate=0.1, max_depth=1,
                            num_iterations=1300, validation_fraction=0.0)
    start = time.time()
    model = fitter(train)
    p = model.predict_probabilities(test).to_numpy()
    chosen = problem.choices(test).map({a: i for i, a in enumerate(problem.alternatives)}).to_numpy()
    ce = float(-np.log(p[np.arange(len(test)), chosen]).mean())
    print(f"test cross-entropy {ce:.4f} (paper: RUMBoost-GBUV 0.6737, MNL 0.7085); "
          f"accuracy {(p.argmax(axis=1) == chosen).mean():.3f}; {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
