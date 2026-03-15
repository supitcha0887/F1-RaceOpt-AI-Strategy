# strategy_optimizer.py
"""
Grid Search กลยุทธ์ที่ดีกว่าของจริง และอธิบายว่า parameter ไหนทำให้เร็วขึ้น
"""

import numpy as np
import pandas as pd
from itertools import product


PIT_LOSS = 22.0  # วินาทีที่เสียตอนเข้าพิท


def _compound_one_hot(comp: str) -> dict:
    comp = comp.upper()
    return {
        "Compound_SOFT":   1 if comp == "SOFT"   else 0,
        "Compound_MEDIUM": 1 if comp == "MEDIUM" else 0,
        "Compound_HARD":   1 if comp == "HARD"   else 0,
    }


def simulate_strategy(
    model,
    feature_cols: list,
    total_laps: int,
    first_compound: str,
    second_compound: str,
    pit_lap: int,
    pace_offset: float = 0.0,
    num_stops: int = 1,
    second_pit_lap: int = None,
    third_compound: str = None,
) -> float:
    """
    จำลอง race time สำหรับกลยุทธ์หนึ่งๆ
    รองรับ 1-stop และ 2-stop
    """
    total_time = 0.0
    stint = 1
    pit_stops = 0
    compound = first_compound.upper()

    for lap in range(1, total_laps + 1):
        pit_penalty = 0.0

        # เช็ค pit stop
        if lap == pit_lap:
            pit_stops += 1
            stint = 2
            compound = second_compound.upper()
            tyre_life = 1
            pit_penalty = PIT_LOSS
        elif num_stops == 2 and second_pit_lap and lap == second_pit_lap:
            pit_stops += 1
            stint = 3
            compound = (third_compound or "HARD").upper()
            tyre_life = 1
            pit_penalty = PIT_LOSS
        else:
            if stint == 1:
                tyre_life = lap
            elif stint == 2:
                tyre_life = lap - pit_lap + 1
            else:
                tyre_life = lap - second_pit_lap + 1

        fuel_est = (total_laps - lap) / total_laps

        feat = {f: 0.0 for f in feature_cols}
        feat.update({
            "LapNumber":     lap,
            "TyreLife":      tyre_life,
            "FuelEst":       fuel_est,
            "StintNumber":   stint,
            "StintLap":      tyre_life,
            "PitStopsSoFar": pit_stops,
            "IsInLap":       1 if pit_penalty > 0 else 0,
            "IsOutLap":      0,
            "TrackStatus_1": 1,
            "Position":      1,
        })
        feat.update(_compound_one_hot(compound))

        X = pd.DataFrame([feat], columns=feature_cols)
        lap_time = float(model.predict(X)[0]) + pace_offset + pit_penalty
        total_time += lap_time

    return total_time


def calibrate_pace_offset(
    model,
    feature_cols: list,
    real_lap_times: list,
    actual_data: pd.DataFrame,
    total_laps: int,
    baseline_strategy: dict,
) -> float:
    """
    หา pace_offset ที่ทำให้เวลาจำลองใกล้เคียงเวลาจริงมากที่สุด
    ค้นหาแบบ binary search ในช่วง [-5, +5] วินาที/lap
    """
    real_total = sum(real_lap_times)

    best_offset = 0.0
    best_diff = float("inf")

    for offset in np.arange(-5.0, 5.1, 0.1):
        sim_total = simulate_strategy(
            model, feature_cols, total_laps,
            baseline_strategy["first_compound"],
            baseline_strategy["second_compound"],
            baseline_strategy["pit_lap"],
            pace_offset=round(offset, 2),
        )
        diff = abs(sim_total - real_total)
        if diff < best_diff:
            best_diff = diff
            best_offset = round(offset, 2)

    return best_offset


def grid_search_strategies(
    model,
    feature_cols: list,
    real_lap_times: list,
    total_laps: int,
    pace_offset: float = 0.0,
) -> list:
    """
    ลอง compound + pit_lap ทุก combination
    คืนค่าผลลัพธ์ทุก combo เรียงจากเร็วสุด
    """
    real_total = sum(real_lap_times)

    compounds = ["SOFT", "MEDIUM", "HARD"]
    # pit_lap ช่วง 20%-55% ของ race
    pit_laps  = list(range(
        max(5, int(total_laps * 0.20)),
        min(total_laps - 5, int(total_laps * 0.55)) + 1,
        2,  # step 2 laps เพื่อประหยัดเวลา
    ))

    results = []

    # 1-stop
    for c1, c2, pit in product(compounds, compounds, pit_laps):
        if c1 == c2:
            continue
        sim = simulate_strategy(
            model, feature_cols, total_laps,
            c1, c2, pit,
            pace_offset=pace_offset,
            num_stops=1,
        )
        delta = sim - real_total
        results.append({
            "num_stops":       1,
            "first_compound":  c1,
            "second_compound": c2,
            "third_compound":  None,
            "pit_lap":         pit,
            "second_pit_lap":  None,
            "sim_total":       sim,
            "real_total":      real_total,
            "delta_sec":       delta,
            "delta_pct":       delta / real_total * 100,
            "faster":          delta < 0,
        })

    # 2-stop (เฉพาะ combo ที่น่าสนใจ)
    two_stop_combos = [
        ("SOFT", "MEDIUM", "HARD"),
        ("SOFT", "HARD",   "MEDIUM"),
        ("MEDIUM", "SOFT", "HARD"),
        ("MEDIUM", "HARD", "SOFT"),
    ]
    pit1_options = [int(total_laps * 0.25), int(total_laps * 0.30)]
    pit2_options = [int(total_laps * 0.55), int(total_laps * 0.60)]

    for (c1, c2, c3), p1, p2 in product(two_stop_combos, pit1_options, pit2_options):
        if p1 >= p2:
            continue
        sim = simulate_strategy(
            model, feature_cols, total_laps,
            c1, c2, p1,
            pace_offset=pace_offset,
            num_stops=2,
            second_pit_lap=p2,
            third_compound=c3,
        )
        delta = sim - real_total
        results.append({
            "num_stops":       2,
            "first_compound":  c1,
            "second_compound": c2,
            "third_compound":  c3,
            "pit_lap":         p1,
            "second_pit_lap":  p2,
            "sim_total":       sim,
            "real_total":      real_total,
            "delta_sec":       delta,
            "delta_pct":       delta / real_total * 100,
            "faster":          delta < 0,
        })

    results.sort(key=lambda r: r["sim_total"])
    return results


def explain_parameters(best: dict, baseline: dict, real_total: float) -> list:
    """
    อธิบายว่า parameter ไหนที่เปลี่ยนไปจาก baseline และทำให้เร็วขึ้น
    """
    explanations = []

    # เปรียบ compound
    if best["first_compound"] != baseline.get("first_compound"):
        explanations.append({
            "param":   "ยางเริ่มต้น (Start Compound)",
            "from":    baseline.get("first_compound", "?"),
            "to":      best["first_compound"],
            "reason":  _compound_reason(baseline.get("first_compound"), best["first_compound"], "start"),
            "impact":  "high",
        })

    if best["second_compound"] != baseline.get("second_compound"):
        explanations.append({
            "param":   "ยางหลังพิท (Second Compound)",
            "from":    baseline.get("second_compound", "?"),
            "to":      best["second_compound"],
            "reason":  _compound_reason(baseline.get("second_compound"), best["second_compound"], "second"),
            "impact":  "high",
        })

    if best["third_compound"] and best["third_compound"] != baseline.get("third_compound"):
        explanations.append({
            "param":   "ยาง Stint ที่ 3 (Third Compound)",
            "from":    baseline.get("third_compound", "ไม่มี"),
            "to":      best["third_compound"],
            "reason":  "เพิ่ม stint ที่ 3 เพื่อใช้ยางที่ทนทานกว่าในช่วงท้าย",
            "impact":  "medium",
        })

    # เปรียบ pit lap
    pit_diff = best["pit_lap"] - baseline.get("pit_lap", best["pit_lap"])
    if abs(pit_diff) >= 2:
        direction = "ช้าลง" if pit_diff > 0 else "เร็วขึ้น"
        tactic    = "Overcut" if pit_diff > 0 else "Undercut"
        explanations.append({
            "param":   f"รอบที่เข้าพิท (Pit Lap: {baseline.get('pit_lap','?')} → {best['pit_lap']})",
            "from":    f"Lap {baseline.get('pit_lap', '?')}",
            "to":      f"Lap {best['pit_lap']}",
            "reason":  f"เลื่อนพิท{direction} {abs(pit_diff)} laps → ใช้กลยุทธ์ {tactic} เพื่อใช้ยางที่มี grip ดีกว่าในช่วงวิกฤต",
            "impact":  "high",
        })

    # เปรียบจำนวนครั้งพิท
    base_stops = baseline.get("num_stops", 1)
    if best["num_stops"] != base_stops:
        explanations.append({
            "param":   f"จำนวนครั้งพิท ({base_stops}-stop → {best['num_stops']}-stop)",
            "from":    f"{base_stops}-stop",
            "to":      f"{best['num_stops']}-stop",
            "reason":  "การเพิ่มจำนวนพิทช่วยให้ใช้ยางที่มี grip สูงขึ้นได้ตลอด race แม้เสียเวลาพิทเพิ่ม",
            "impact":  "medium",
        })

    return explanations


def _compound_reason(from_c, to_c, position):
    mapping = {
        ("MEDIUM", "SOFT",   "start"):  "Soft มี grip สูงกว่าในช่วงต้น race ช่วยให้ได้เวลาที่ดีกว่าก่อนที่ยางจะเสื่อม",
        ("SOFT",   "MEDIUM", "start"):  "Medium ทนทานกว่าในช่วงต้น ช่วยให้ยืด stint ได้นานขึ้น",
        ("MEDIUM", "HARD",   "start"):  "Hard ทนทานมากที่สุด เหมาะกับการยืด stint ยาวในสนามที่ tyre deg สูง",
        ("SOFT",   "MEDIUM", "second"): "Medium ในช่วงหลังช่วยยืด stint ท้าย race ได้นานกว่า Soft ที่เสื่อมเร็ว",
        ("MEDIUM", "SOFT",   "second"): "Soft ในช่วงท้ายช่วยให้ push pace ได้สูงสุดในช่วงโค้งสุดท้าย",
        ("HARD",   "SOFT",   "second"): "Soft ในช่วงท้ายให้ grip สูงสุด เหมาะกับการ attack ในช่วงท้าย race",
    }
    return mapping.get((from_c, to_c, position),
                       f"เปลี่ยนจาก {from_c} → {to_c} เพื่อ balance ระหว่าง pace และ durability")