# race_simulator.py
"""
Full race simulation 20 drivers ด้วยโมเดล RandomForest ที่เทรนจาก FastF1

แนวคิด:
- ใช้โมเดลเดียว (เทรนจาก VER) เป็น "base pace"
- แต่ละนักขับมี pace_offset ต่างกัน (ช้ากว่า/เร็วกว่าเล็กน้อย)
- กลยุทธ์พื้นฐาน: 2 stint (Medium -> Soft หรือ Soft -> Medium)
- คำนวณ lap time ทีละ lap + pit time + รวมเวลา
- ได้ผลลัพธ์: total_time ต่อ driver + รายละเอียด lap-by-lap

ใช้ร่วมกับ model.pkl ที่เซฟจาก train_model_advanced.py
"""

import pickle
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict


# คงที่สำหรับการจำลอง
PIT_LOSS = 22.0  # เวลาที่เสียตอนเข้าพิท (วินาที) แบบคร่าว ๆ
TOTAL_LAPS_DEFAULT = 57  # Bahrain 2023 VER = 57 laps โดยประมาณ


# pace_offset ต่อ driver (สมมติจากฟอร์มโดยรวม / ใช้ปรับความเร็ว)
DRIVER_PACE = {
    "VER": 0.0,
    "PER": 0.30,
    "ALO": 0.70,
    "HAM": 0.90,
    "SAI": 1.00,
    "RUS": 1.05,
    "LEC": 0.80,
    "STR": 1.40,
    "NOR": 1.20,
    "PIA": 1.30,
    "GAS": 1.60,
    "OCO": 1.60,
    "BOT": 1.80,
    "ZHO": 1.90,
    "HUL": 2.00,
    "MAG": 2.10,
    "ALB": 2.20,
    "SAR": 2.40,
    "TSU": 2.10,
    "DEV": 2.50,  # ตัวอย่าง grid ปี 2023
}


# กลยุทธ์เริ่มต้นต่อ driver: (สามารถไปปรับ UI ให้เปลี่ยนกลยุทธ์ได้ภายหลัง)
DEFAULT_STRATEGIES = {
    "VER": ("MEDIUM", "SOFT", 25),
    "PER": ("SOFT", "MEDIUM", 18),
    "ALO": ("SOFT", "MEDIUM", 16),
    "HAM": ("SOFT", "MEDIUM", 14),
    "SAI": ("SOFT", "MEDIUM", 15),
    "RUS": ("SOFT", "MEDIUM", 15),
    "LEC": ("SOFT", "MEDIUM", 13),
    "STR": ("SOFT", "MEDIUM", 12),
    "NOR": ("SOFT", "MEDIUM", 17),
    "PIA": ("SOFT", "MEDIUM", 18),
    "GAS": ("SOFT", "MEDIUM", 18),
    "OCO": ("SOFT", "MEDIUM", 18),
    "BOT": ("SOFT", "MEDIUM", 19),
    "ZHO": ("SOFT", "MEDIUM", 19),
    "HUL": ("SOFT", "MEDIUM", 20),
    "MAG": ("SOFT", "MEDIUM", 20),
    "ALB": ("SOFT", "MEDIUM", 17),
    "SAR": ("SOFT", "MEDIUM", 17),
    "TSU": ("SOFT", "MEDIUM", 16),
    "DEV": ("SOFT", "MEDIUM", 16),
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
    rank: int = 0  # ให้ตอนหลัง


class LapPredictor:
    """โหลดโมเดลจาก model.pkl และให้ฟังก์ชัน predict lap time"""

    def __init__(self, model_path: str = "model.pkl"):
        with open(model_path, "rb") as f:
            obj = pickle.load(f)
        self.model = obj["model"]
        self.features: List[str] = obj["features"]

    def predict_lap(self, feature_dict: Dict) -> float:
        """รับ dict ของฟีเจอร์ 1 lap และคืนค่า lap time ที่ทำนาย (วินาที)"""
        # ให้แน่ใจว่ามีทุก feature; ถ้าไม่มีให้ใส่ 0
        row = {f: 0.0 for f in self.features}
        for k, v in feature_dict.items():
            if k in row:
                row[k] = v

        X = pd.DataFrame([row], columns=self.features)
        pred = float(self.model.predict(X)[0])
        return pred


def build_strategy_objects() -> List[DriverStrategy]:
    strategies = []
    for code, (c1, c2, pit) in DEFAULT_STRATEGIES.items():
        pace = DRIVER_PACE.get(code, 1.5)
        strategies.append(
            DriverStrategy(
                code=code,
                first_compound=c1,
                second_compound=c2,
                pit_lap=pit,
                pace_offset=pace,
            )
        )
    return strategies


def _compound_one_hot(comp: str) -> Dict[str, int]:
    comp = comp.upper()
    return {
        "Compound_SOFT": 1 if comp == "SOFT" else 0,
        "Compound_MEDIUM": 1 if comp == "MEDIUM" else 0,
        "Compound_HARD": 1 if comp == "HARD" else 0,
    }


def simulate_driver(
    predictor: LapPredictor,
    strategy: DriverStrategy,
    total_laps: int = TOTAL_LAPS_DEFAULT,
) -> DriverResult:
    laps = []
    stint_number = 1
    pit_stops = 0
    current_compound = strategy.first_compound

    for lap in range(1, total_laps + 1):
        if lap == strategy.pit_lap:
            # lap นี้ถือเป็น in-lap + pit time
            pit_stops += 1
            stint_number = 2
            current_compound = strategy.second_compound
            tyre_life = 1  # เริ่ม stint ใหม่หลัง pit
            is_in_lap = 1
            is_out_lap = 0
            pit_penalty = PIT_LOSS
        else:
            # normal lap
            tyre_life = lap if stint_number == 1 else lap - strategy.pit_lap + 1
            is_in_lap = 0
            is_out_lap = 0
            pit_penalty = 0.0

        fuel_est = (total_laps - lap) / total_laps
        stint_lap = tyre_life

        feat = {
            "LapNumber": lap,
            "TyreLife": tyre_life,
            "FuelEst": fuel_est,
            "StintNumber": stint_number,
            "StintLap": stint_lap,
            "PitStopsSoFar": pit_stops,
            "IsInLap": is_in_lap,
            "IsOutLap": is_out_lap,
            "TrackStatus_1": 1,  # สมมติ green flag ทั้ง race
            "Position": 1,       # ใช้เป็น dummy
        }
        feat.update(_compound_one_hot(current_compound))

        base_lap = predictor.predict_lap(feat)

        # เติม pace offset และ noise เล็กน้อย
        lap_time = base_lap + strategy.pace_offset + np.random.normal(0, 0.12)
        lap_time += pit_penalty

        laps.append(lap_time)

    total_time = float(np.sum(laps))
    return DriverResult(
        code=strategy.code,
        total_time=total_time,
        laps=laps,
        strategy=strategy,
    )


def simulate_full_race(
    model_path: str = "model.pkl",
    total_laps: int = TOTAL_LAPS_DEFAULT,
) -> List[DriverResult]:
    predictor = LapPredictor(model_path)
    strategies = build_strategy_objects()

    results: List[DriverResult] = []
    for st in strategies:
        res = simulate_driver(predictor, st, total_laps=total_laps)
        results.append(res)

    # Ranking
    results.sort(key=lambda r: r.total_time)
    for i, r in enumerate(results, start=1):
        r.rank = i

    return results


def compute_win_probabilities(results: List[DriverResult]) -> Dict[str, float]:
    """สร้างความน่าจะเป็นแบบ softmax จาก total_time"""
    times = np.array([r.total_time for r in results])
    # invert: เวลาน้อย = ดีกว่า
    scores = -times
    exp = np.exp(scores - scores.max())
    probs = exp / exp.sum()
    return {r.code: float(p) for r, p in zip(results, probs)}


if __name__ == "__main__":
    # ทดสอบรันตรง ๆ
    res = simulate_full_race()
    for r in res[:5]:
        print(r.rank, r.code, f"{r.total_time:.3f}")
    probs = compute_win_probabilities(res)
    print("Win Probabilities (top 5):")
    for r in res[:5]:
        print(r.code, f"{probs[r.code]*100:.1f}%")
