# data_pipeline.py
"""
ดึงข้อมูล Lap จาก FastF1 และเตรียม Feature สำหรับใช้เทรน AI และ Simulation
"""

import fastf1
import pandas as pd
import numpy as np


def load_race_laps(year=2023, gp="Bahrain", driver="VER"):
    """
    โหลดข้อมูลการแข่งขันจริงจาก FastF1
    และสร้างฟีเจอร์ระดับกลาง-สูง (FuelEstimate, StintLap, PitStopsSoFar ฯลฯ)

    Return:
        data: DataFrame พร้อมฟีเจอร์ที่พร้อมใช้กับโมเดล
        meta: dict รวมข้อมูล meta เช่น total_laps, driver, year, gp
    """
    fastf1.Cache.enable_cache("cache")

    session = fastf1.get_session(year, gp, "R")
    session.load()

    # เลือกนักขับ
    # ใช้ code โดยตรงปลอดภัยกว่าใน fastf1 เวอร์ชันใหม่
    laps = session.laps.pick_driver(driver).reset_index(drop=True)

    # แปลง LapTime / SectorTime เป็นวินาที
    laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()
    laps["Sector1Sec"] = laps["Sector1Time"].dt.total_seconds()
    laps["Sector2Sec"] = laps["Sector2Time"].dt.total_seconds()
    laps["Sector3Sec"] = laps["Sector3Time"].dt.total_seconds()

    # กรองแถวที่ไม่มี LapTime หรือ LapNumber ออกก่อน
    laps = laps[~laps["LapTimeSec"].isna() & ~laps["LapNumber"].isna()].copy()
    laps = laps.reset_index(drop=True)

    if laps.empty:
        raise ValueError("ไม่พบ lap ของ driver นี้ใน session ที่เลือก (DataFrame ว่าง)")

    # ตอนนี้ค่อยคำนวณจำนวน lap สูงสุด
    total_laps = int(laps["LapNumber"].max())

    # FuelEstimate แบบง่าย: สมมติว่า lap แรกน้ำมันเต็ม, lap สุดท้ายเกือบหมด
    laps["FuelEst"] = (total_laps - laps["LapNumber"]) / total_laps

    # ถ้ามีคอลัมน์ Stint ให้ใช้เลย (FastF1 ส่วนใหญ่มี)
    if "Stint" in laps.columns:
        laps["StintNumber"] = (
            laps["Stint"].fillna(method="ffill").fillna(1).astype(int)
        )
    else:
        # ถ้าไม่มี ให้ถือว่าไม่มีการเปลี่ยนยาง => Stint เดียว
        laps["StintNumber"] = 1

    # สร้าง StintLap = นับ lap ใหม่เมื่อเปลี่ยน stint
    laps["StintLap"] = laps.groupby("StintNumber").cumcount() + 1

    # จำนวนครั้งที่เข้าพิทก่อนหน้านี้ ~ StintNumber - 1
    laps["PitStopsSoFar"] = laps["StintNumber"] - 1

    # Flag outlap / inlap จาก PitOutTime / PitInTime
    laps["IsOutLap"] = laps["PitOutTime"].notna().astype(int)
    laps["IsInLap"] = laps["PitInTime"].notna().astype(int)

    # TrackStatus อาจมี NaN
    if "TrackStatus" not in laps.columns:
        laps["TrackStatus"] = 1
    laps["TrackStatus"] = laps["TrackStatus"].fillna(1).astype(int)

    # เผื่อ TyreLife ไม่มีค่า → ใส่ 0 แทน
    if "TyreLife" in laps.columns:
        laps["TyreLife"] = laps["TyreLife"].fillna(0)
    else:
        laps["TyreLife"] = 0

    # เผื่อ Position ไม่มีค่า → ใส่ 1 แทน (ถือว่าอยู่อันดับ 1)
    if "Position" in laps.columns:
        laps["Position"] = laps["Position"].fillna(method="ffill").fillna(1)
    else:
        laps["Position"] = 1

    # ตัดฟีเจอร์ที่น่าจะใช้
    base_cols = [
        "LapNumber",
        "LapTimeSec",
        "Compound",
        "TyreLife",
        "TrackStatus",
        "Position",
        "Sector1Sec",
        "Sector2Sec",
        "Sector3Sec",
        "FuelEst",
        "StintNumber",
        "StintLap",
        "PitStopsSoFar",
        "IsOutLap",
        "IsInLap",
    ]

    # เผื่อบางคอลัมน์ไม่มีใน laps → เอาเฉพาะที่มีจริง
    base_cols = [c for c in base_cols if c in laps.columns]

    data = laps[base_cols].copy()

    # One-hot encoding Compound + TrackStatus
    data = pd.get_dummies(
        data,
        columns=[col for col in ["Compound", "TrackStatus"] if col in data.columns],
        drop_first=False,  # เก็บครบทุกค่า
    )

    meta = {
        "year": year,
        "gp": gp,
        "driver": driver,
        "total_laps": total_laps,
    }

    return data, meta
