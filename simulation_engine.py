# simulation_engine.py
"""
Engine สำหรับจำลองกลยุทธ์ pit stop โดยใช้โมเดลที่เทรนแล้ว
"""

import numpy as np
import pandas as pd


def _prepare_single_lap_features(
    base_row: pd.Series,
    feature_cols,
    compound: str,
    stint_lap: int,
    pit_stops_so_far: int,
):
    """
    เอา base_row ของ lap นั้นๆ มาแก้ค่าบางฟีเจอร์:
    - compound (Soft/Medium/Hard)
    - StintLap
    - PitStopsSoFar

    แล้วคืนค่าเป็น vector ที่เรียงตาม feature_cols
    """
    row = base_row.copy()

    # Reset one-hot ของ Compound ทั้งหมด
    for c in row.index:
        if c.startswith("Compound_"):
            row[c] = 0

    # ตั้งค่ายางใหม่
    comp_key = f"Compound_{compound.upper()}"
    if comp_key in row.index:
        row[comp_key] = 1

    # ปรับค่า stint lap / pit count
    if "StintLap" in row.index:
        row["StintLap"] = stint_lap
    if "PitStopsSoFar" in row.index:
        row["PitStopsSoFar"] = pit_stops_so_far

    # สร้าง vector ตามลำดับ feature_cols
    feat = []
    for col in feature_cols:
        feat.append(row.get(col, 0.0))
    return np.array(feat, dtype=float)


def simulate_one_stop_strategy(
    data: pd.DataFrame,
    feature_cols,
    model,
    start_compound: str,
    second_compound: str,
    pit_lap: int,
):
    """
    จำลองกลยุทธ์ 1-stop เช่น:
    - Start: MEDIUM → PIt lap 20 → SOFT

    data: DataFrame จาก data_pipeline (มีฟีเจอร์จริงของแต่ละ lap)
    feature_cols: list ของฟีเจอร์ที่โมเดลใช้
    model: RandomForestRegressor ที่เทรนแล้ว
    pit_lap: lap ที่เข้าพิท (ย้ายชุดยาง)
    """

    max_lap = int(data["LapNumber"].max())
    total_time = 0.0

    stint_lap = 0
    pit_stops = 0
    current_compound = start_compound.upper()

    for lap in range(1, max_lap + 1):
        base_row = data.loc[data["LapNumber"] == lap].iloc[0]

        # เช็คว่าจะเปลี่ยนยางที่ lap นี้ไหม
        if lap == pit_lap:
            pit_stops += 1
            current_compound = second_compound.upper()
            stint_lap = 1  # รอบแรกของ stint ใหม่
        else:
            stint_lap += 1

        feat = _prepare_single_lap_features(
            base_row, feature_cols, current_compound, stint_lap, pit_stops
        )

        lap_time_pred = model.predict(feat.reshape(1, -1))[0]
        total_time += lap_time_pred

    return total_time
