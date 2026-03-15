# data_pipeline.py
"""
ดึงข้อมูล Lap จาก FastF1 และเตรียม Feature สำหรับใช้เทรน AI และ Simulation
รองรับหลาย Race / หลายนักแข่ง
"""

import os
import pickle       
import fastf1
import pandas as pd
import numpy as np


AVAILABLE_RACES = {
    "2023_Bahrain":     {"year": 2023, "gp": "Bahrain",      "label": "Bahrain GP 2023",      "laps": 57},
    "2023_Saudi":       {"year": 2023, "gp": "Saudi Arabia",  "label": "Saudi Arabia GP 2023",  "laps": 50},
    "2023_Australia":   {"year": 2023, "gp": "Australia",    "label": "Australia GP 2023",    "laps": 58},
    "2023_Monaco":      {"year": 2023, "gp": "Monaco",       "label": "Monaco GP 2023",       "laps": 78},
    "2023_Silverstone": {"year": 2023, "gp": "British",      "label": "British GP 2023",      "laps": 52},
    "2023_Monza":       {"year": 2023, "gp": "Italian",      "label": "Italian GP 2023",      "laps": 51},
    "2022_Bahrain":     {"year": 2022, "gp": "Bahrain",      "label": "Bahrain GP 2022",      "laps": 57},
    "2022_Monaco":      {"year": 2022, "gp": "Monaco",       "label": "Monaco GP 2022",       "laps": 64},
}

AVAILABLE_DRIVERS = [
    "VER", "PER", "HAM", "RUS", "LEC", "SAI",
    "ALO", "STR", "NOR", "PIA", "GAS", "OCO",
    "BOT", "ZHO", "HUL", "MAG", "ALB", "TSU",
]


def load_race_laps(year=2023, gp="Bahrain", driver="VER"):
    """
    โหลดข้อมูลจาก FastF1 พร้อม disk cache
    ครั้งแรก: ดึงจาก FastF1 แล้วบันทึก cache
    ครั้งต่อไป: โหลด cache ทันที (เร็วมาก)
    """
    # ── Disk cache ──────────────────────────────────
    cache_key  = f"{year}_{gp}_{driver}".replace(" ", "_")
    cache_path = os.path.join("cache", f"{cache_key}.pkl")
    os.makedirs("cache", exist_ok=True)

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)        # ← โหลด cache ทันที

    # ── ดึงจาก FastF1 (ครั้งแรก) ──────────────────
    cache_dir = os.path.join("cache", f"{year}_{gp.replace(' ', '_')}")
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    session = fastf1.get_session(year, gp, "R")
    session.load()

    laps = session.laps.pick_driver(driver).reset_index(drop=True)

    laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()
    laps["Sector1Sec"] = laps["Sector1Time"].dt.total_seconds()
    laps["Sector2Sec"] = laps["Sector2Time"].dt.total_seconds()
    laps["Sector3Sec"] = laps["Sector3Time"].dt.total_seconds()

    laps = laps[~laps["LapTimeSec"].isna() & ~laps["LapNumber"].isna()].copy()
    laps = laps.reset_index(drop=True)

    if laps.empty:
        raise ValueError(f"ไม่พบ lap ของ {driver} ใน {gp} {year}")

    total_laps = int(laps["LapNumber"].max())

    laps["FuelEst"] = (total_laps - laps["LapNumber"]) / total_laps

    if "Stint" in laps.columns:
        laps["StintNumber"] = laps["Stint"].ffill().fillna(1).astype(int)
    else:
        laps["StintNumber"] = 1

    laps["StintLap"]      = laps.groupby("StintNumber").cumcount() + 1
    laps["PitStopsSoFar"] = laps["StintNumber"] - 1
    laps["IsOutLap"]      = laps["PitOutTime"].notna().astype(int)
    laps["IsInLap"]       = laps["PitInTime"].notna().astype(int)

    if "TrackStatus" not in laps.columns:
        laps["TrackStatus"] = 1
    laps["TrackStatus"] = laps["TrackStatus"].fillna(1).astype(int)

    laps["TyreLife"] = laps["TyreLife"].fillna(0) if "TyreLife" in laps.columns else 0

    if "Position" in laps.columns:
        laps["Position"] = laps["Position"].ffill().fillna(1)
    else:
        laps["Position"] = 1

    base_cols = [
        "LapNumber", "LapTimeSec", "Compound", "TyreLife", "TrackStatus",
        "Position", "Sector1Sec", "Sector2Sec", "Sector3Sec",
        "FuelEst", "StintNumber", "StintLap", "PitStopsSoFar",
        "IsOutLap", "IsInLap",
    ]
    base_cols = [c for c in base_cols if c in laps.columns]
    data = laps[base_cols].copy()

    data = pd.get_dummies(
        data,
        columns=[col for col in ["Compound", "TrackStatus"] if col in data.columns],
        drop_first=False,
    )

    meta = {"year": year, "gp": gp, "driver": driver, "total_laps": total_laps}

    # ── บันทึก cache ────────────────────────────────
    with open(cache_path, "wb") as f:
        pickle.dump((data, meta), f)     # ← indent ถูกต้องแล้ว

    return data, meta


def load_multi_race_laps(race_driver_list):
    all_data = []
    all_meta = []

    for year, gp, driver in race_driver_list:
        try:
            print(f"  โหลด {driver} @ {gp} {year}...")
            data, meta = load_race_laps(year, gp, driver)
            data["DriverCode"] = driver
            data["RaceYear"]   = year
            data["GP_Label"]   = gp
            all_data.append(data)
            all_meta.append(meta)
            print(f"    ✓ {len(data)} laps")
        except Exception as e:
            print(f"    ✗ ข้าม {driver} @ {gp} {year}: {e}")

    if not all_data:
        raise ValueError("ไม่สามารถโหลดข้อมูลได้เลย")

    combined = pd.concat(all_data, ignore_index=True)
    combined = pd.get_dummies(combined, columns=["DriverCode", "GP_Label"], drop_first=False)

    return combined, all_meta