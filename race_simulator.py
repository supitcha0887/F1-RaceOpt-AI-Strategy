# race_simulator.py
"""
Full race simulation 20 drivers ด้วยโมเดล RandomForest
รองรับการเลือก race และนักแข่งจากภายนอก
"""

import pickle
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

PIT_LOSS = 22.0

DRIVER_PACE = {
    "VER": 0.0,  "PER": 0.30, "ALO": 0.70,  "HAM": 0.90,
    "SAI": 1.00, "RUS": 1.05, "LEC": 0.80,  "STR": 1.40,
    "NOR": 1.20, "PIA": 1.30, "GAS": 1.60,  "OCO": 1.60,
    "BOT": 1.80, "ZHO": 1.90, "HUL": 2.00,  "MAG": 2.10,
    "ALB": 2.20, "SAR": 2.40, "TSU": 2.10,  "DEV": 2.50,
}

DEFAULT_STRATEGIES = {
    "VER": ("MEDIUM", "SOFT", 25), "PER": ("SOFT", "MEDIUM", 18),
    "ALO": ("SOFT", "MEDIUM", 16), "HAM": ("SOFT", "MEDIUM", 14),
    "SAI": ("SOFT", "MEDIUM", 15), "RUS": ("SOFT", "MEDIUM", 15),
    "LEC": ("SOFT", "MEDIUM", 13), "STR": ("SOFT", "MEDIUM", 12),
    "NOR": ("SOFT", "MEDIUM", 17), "PIA": ("SOFT", "MEDIUM", 18),
    "GAS": ("SOFT", "MEDIUM", 18), "OCO": ("SOFT", "MEDIUM", 18),
    "BOT": ("SOFT", "MEDIUM", 19), "ZHO": ("SOFT", "MEDIUM", 19),
    "HUL": ("SOFT", "MEDIUM", 20), "MAG": ("SOFT", "MEDIUM", 20),
    "ALB": ("SOFT", "MEDIUM", 17), "SAR": ("SOFT", "MEDIUM", 17),
    "TSU": ("SOFT", "MEDIUM", 16), "DEV": ("SOFT", "MEDIUM", 16),
}


@dataclass
class DriverStrategy:
    code: str
    first_compound: str
    second_compound: str
    pit_lap: int
    pace_offset: float


@dataclass
class DriverResult:
    code: str
    total_time: float
    laps: List[float]
    strategy: DriverStrategy
    rank: int = 0


class LapPredictor:
    def __init__(self, model_path: str = "model.pkl"):
        with open(model_path, "rb") as f:
            obj = pickle.load(f)
        self.model = obj["model"]
        self.features: List[str] = obj["features"]

    def predict_lap(self, feature_dict: Dict) -> float:
        row = {f: 0.0 for f in self.features}
        for k, v in feature_dict.items():
            if k in row:
                row[k] = v
        X = pd.DataFrame([row], columns=self.features)
        return float(self.model.predict(X)[0])


def _compound_one_hot(comp: str) -> Dict[str, int]:
    comp = comp.upper()
    return {
        "Compound_SOFT":   1 if comp == "SOFT"   else 0,
        "Compound_MEDIUM": 1 if comp == "MEDIUM" else 0,
        "Compound_HARD":   1 if comp == "HARD"   else 0,
    }


def simulate_driver(
    predictor: LapPredictor,
    strategy: DriverStrategy,
    total_laps: int,
) -> DriverResult:
    laps = []
    stint_number = 1
    pit_stops = 0
    current_compound = strategy.first_compound

    for lap in range(1, total_laps + 1):
        if lap == strategy.pit_lap:
            pit_stops += 1
            stint_number = 2
            current_compound = strategy.second_compound
            tyre_life = 1
            is_in_lap = 1
            pit_penalty = PIT_LOSS
        else:
            tyre_life = lap if stint_number == 1 else lap - strategy.pit_lap + 1
            is_in_lap = 0
            pit_penalty = 0.0

        feat = {
            "LapNumber":      lap,
            "TyreLife":       tyre_life,
            "FuelEst":        (total_laps - lap) / total_laps,
            "StintNumber":    stint_number,
            "StintLap":       tyre_life,
            "PitStopsSoFar":  pit_stops,
            "IsInLap":        is_in_lap,
            "IsOutLap":       0,
            "TrackStatus_1":  1,
            "Position":       1,
        }
        feat.update(_compound_one_hot(current_compound))

        base_lap = predictor.predict_lap(feat)
        lap_time = base_lap + strategy.pace_offset + np.random.normal(0, 0.12)
        lap_time += pit_penalty
        laps.append(lap_time)

    return DriverResult(
        code=strategy.code,
        total_time=float(np.sum(laps)),
        laps=laps,
        strategy=strategy,
    )


def simulate_full_race(
    model_path: str = "model.pkl",
    total_laps: int = 57,
    global_offset: float = 0.0,
) -> List[DriverResult]:
    predictor = LapPredictor(model_path)

    strategies = []
    for code, (c1, c2, pit) in DEFAULT_STRATEGIES.items():
        strategies.append(DriverStrategy(
            code=code,
            first_compound=c1,
            second_compound=c2,
            pit_lap=pit,
            pace_offset=DRIVER_PACE.get(code, 1.5) + global_offset,
        ))

    results = []
    for st in strategies:
        results.append(simulate_driver(predictor, st, total_laps))

    results.sort(key=lambda r: r.total_time)
    for i, r in enumerate(results, start=1):
        r.rank = i

    return results


def compute_win_probabilities(results: List[DriverResult]) -> Dict[str, float]:
    times = np.array([r.total_time for r in results])
    scores = -times
    exp = np.exp(scores - scores.max())
    probs = exp / exp.sum()
    return {r.code: float(p) for r, p in zip(results, probs)}