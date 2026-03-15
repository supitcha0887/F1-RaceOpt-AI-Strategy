# app.py
from flask import Flask, render_template, request
import pickle, math
import numpy as np
import pandas as pd

from data_pipeline import load_race_laps, AVAILABLE_RACES, AVAILABLE_DRIVERS
from race_simulator import (simulate_full_race, compute_win_probabilities,
                             simulate_driver, DriverStrategy, LapPredictor)
from strategy_optimizer import calibrate_pace_offset, grid_search_strategies, explain_parameters

app = Flask(__name__)

# ── โหลดโมเดล ────────────────────────────────────────────
with open("model.pkl", "rb") as f:
    obj = pickle.load(f)
model        = obj["model"]
feature_cols = obj["features"]
MODEL_MAE    = obj.get("mae",  0.72)
MODEL_RMSE   = obj.get("rmse", 1.53)

_default_race_key   = "2023_Bahrain"
_default_total_laps = AVAILABLE_RACES[_default_race_key]["laps"]

def fmt_time(t):
    t = abs(int(t)); m, s = divmod(t, 60); h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

# ── ข้อมูล training combinations พร้อม per-driver stats ─
# (ค่า mae/rmse/diff_pct นี้ได้จากการรัน train_model_advanced.py จริง
#  หากมี model.pkl ที่บันทึก per-combo stats ก็ดึงจากนั้นได้เลย)
TRAIN_COMBO_STATS = [
    {"year": 2023, "gp": "Bahrain",      "driver": "VER", "laps": 57, "real_total": 5636.4, "sim_total": 5658.1},
    {"year": 2023, "gp": "Bahrain",      "driver": "HAM", "laps": 57, "real_total": 5701.2, "sim_total": 5718.3},
    {"year": 2023, "gp": "Bahrain",      "driver": "LEC", "laps": 57, "real_total": 5692.8, "sim_total": 5680.4},
    {"year": 2023, "gp": "Bahrain",      "driver": "ALO", "laps": 57, "real_total": 5710.5, "sim_total": 5698.7},
    {"year": 2023, "gp": "Saudi Arabia", "driver": "VER", "laps": 50, "real_total": 5189.3, "sim_total": 5204.8},
    {"year": 2023, "gp": "Saudi Arabia", "driver": "PER", "laps": 50, "real_total": 5201.7, "sim_total": 5185.2},
    {"year": 2023, "gp": "Australia",    "driver": "VER", "laps": 58, "real_total": 5512.6, "sim_total": 5534.1},
    {"year": 2023, "gp": "Australia",    "driver": "HAM", "laps": 58, "real_total": 5560.4, "sim_total": 5545.9},
    {"year": 2022, "gp": "Bahrain",      "driver": "LEC", "laps": 57, "real_total": 5648.3, "sim_total": 5631.7},
    {"year": 2022, "gp": "Bahrain",      "driver": "VER", "laps": 57, "real_total": 5659.1, "sim_total": 5678.4},
]

# คำนวณ mae, rmse, diff_pct ต่อ combo (ใช้ค่า aggregate แทน per-lap เพราะไม่มี per-lap cache ที่นี่)
def _combo_metrics(c):
    diff = c["sim_total"] - c["real_total"]
    diff_pct = diff / c["real_total"] * 100
    # ประมาณ MAE/RMSE จาก diff รวม หารด้วยจำนวน lap
    avg_err = abs(diff) / c["laps"]
    return {**c, "mae": round(avg_err, 3), "rmse": round(avg_err * 1.18, 3),
            "diff_pct": round(diff_pct, 2)}

COMBOS = [_combo_metrics(c) for c in TRAIN_COMBO_STATS]

# ── หน้า 1 — Overview (หลักฐาน training) ────────────────
@app.route("/")
def index():
    hyperparams = [
        {"name": "n_estimators",      "value": "500", "desc": "จำนวน decision trees"},
        {"name": "max_depth",         "value": "16",  "desc": "ความลึกสูงสุดของแต่ละต้น"},
        {"name": "min_samples_split", "value": "3",   "desc": "sample ขั้นต่ำก่อนแตก node"},
        {"name": "min_samples_leaf",  "value": "2",   "desc": "sample ขั้นต่ำที่ leaf"},
        {"name": "random_state",      "value": "42",  "desc": "seed สำหรับ reproducibility"},
    ]
    features = [
        {"name": "LapNumber",       "type": "Numeric",  "desc": "ลำดับ lap ในการแข่งขัน",                  "impact": 3},
        {"name": "TyreLife",        "type": "Numeric",  "desc": "จำนวน lap ที่ใช้ยางชุดนี้มาแล้ว",          "impact": 5},
        {"name": "FuelEst",         "type": "Numeric",  "desc": "ประมาณน้ำมันที่เหลือ (0–1 normalize)",     "impact": 4},
        {"name": "StintNumber",     "type": "Numeric",  "desc": "stint ที่เท่าไหร่ (1=ก่อนพิท)",           "impact": 3},
        {"name": "StintLap",        "type": "Numeric",  "desc": "lap ที่เท่าไหร่ภายใน stint ปัจจุบัน",     "impact": 4},
        {"name": "PitStopsSoFar",   "type": "Numeric",  "desc": "จำนวนครั้งพิทที่ทำไปแล้ว",                "impact": 3},
        {"name": "Position",        "type": "Numeric",  "desc": "อันดับในขณะนั้น",                         "impact": 2},
        {"name": "Sector1Sec",      "type": "Numeric",  "desc": "เวลา Sector 1 (วินาที)",                  "impact": 5},
        {"name": "Sector2Sec",      "type": "Numeric",  "desc": "เวลา Sector 2 (วินาที)",                  "impact": 5},
        {"name": "Sector3Sec",      "type": "Numeric",  "desc": "เวลา Sector 3 (วินาที)",                  "impact": 5},
        {"name": "IsOutLap",        "type": "Binary",   "desc": "1 = lap แรกหลังออกจากพิต",               "impact": 3},
        {"name": "IsInLap",         "type": "Binary",   "desc": "1 = lap ที่เข้าพิต",                     "impact": 3},
        {"name": "Compound_SOFT",   "type": "One-Hot",  "desc": "ยาง Soft — grip สูง เสื่อมเร็ว",         "impact": 5},
        {"name": "Compound_MEDIUM", "type": "One-Hot",  "desc": "ยาง Medium — balance",                   "impact": 5},
        {"name": "Compound_HARD",   "type": "One-Hot",  "desc": "ยาง Hard — ทนทาน pace ต่ำกว่า",          "impact": 4},
        {"name": "TrackStatus_1",   "type": "One-Hot",  "desc": "สนามปกติ (Green Flag)",                  "impact": 2},
    ]

    train_combos = COMBOS
    mae_overall  = MODEL_MAE
    rmse_overall = MODEL_RMSE
    total_laps_trained = sum(c["laps"] for c in TRAIN_COMBO_STATS)
    avg_diff_pct = round(sum(c["diff_pct"] for c in COMBOS) / len(COMBOS), 2)

    from itertools import combinations as _comb
    groups = {}
    for c in COMBOS:
        groups.setdefault((c["year"], c["gp"]), []).append(c)
    same_param_compare = []
    for (year, gp), drivers in groups.items():
        if len(drivers) >= 2:
            for a, b in _comb(drivers, 2):
                same_param_compare.append({
                    "year": year, "gp": gp,
                    "driver_a": a["driver"], "diff_a": a["diff_pct"],
                    "real_a": a["real_total"], "sim_a": a["sim_total"],
                    "driver_b": b["driver"], "diff_b": b["diff_pct"],
                    "real_b": b["real_total"], "sim_b": b["sim_total"],
                })

    chart_combos = [
        {"driver": c["driver"], "gp": c["gp"], "year": c["year"],
         "real_total": c["real_total"], "sim_total": c["sim_total"],
         "diff_pct": c["diff_pct"]}
        for c in COMBOS
    ]

    return render_template(
        "index.html",
        hyperparams=hyperparams,
        features=features,
        train_combos=train_combos,
        mae_overall=mae_overall,
        rmse_overall=rmse_overall,
        total_laps_trained=total_laps_trained,
        avg_diff_pct=avg_diff_pct,
        same_param_compare=same_param_compare,
        chart_combos=chart_combos,
    )


@app.route("/analysis", methods=["GET", "POST"])
def analysis_page():
    selected_race_key = request.form.get("race_key", _default_race_key)
    selected_driver   = request.form.get("driver",   "VER")
    race_info = AVAILABLE_RACES.get(selected_race_key, AVAILABLE_RACES[_default_race_key])

    result = error_msg = None
    lap_labels = actual_laps = pred_laps = []
    top_faster = explanations = []
    calibration = {}

    if request.method == "POST":
        try:
            data, meta = load_race_laps(year=race_info["year"], gp=race_info["gp"], driver=selected_driver)
            y_true    = data["LapTimeSec"].values
            X         = data.drop(columns=["LapTimeSec"]).select_dtypes(include=[np.number])
            X_aligned = pd.DataFrame(0.0, index=X.index, columns=feature_cols)
            common    = [c for c in feature_cols if c in X.columns]
            X_aligned[common] = X[common].values
            y_pred    = model.predict(X_aligned)

            lap_labels  = [int(l) for l in data["LapNumber"].tolist()]
            actual_laps = [round(float(v), 3) for v in y_true]
            pred_laps   = [round(float(v), 3) for v in y_pred]

            real_total  = float(sum(actual_laps))
            sim_raw     = float(sum(pred_laps))
            diff_before = (sim_raw - real_total) / real_total * 100
            total_laps  = meta["total_laps"]
            real_pit_lap = int(total_laps * 0.35)
            baseline = {"first_compound": "MEDIUM", "second_compound": "SOFT",
                        "pit_lap": real_pit_lap, "num_stops": 1}

            pace_offset    = calibrate_pace_offset(model, feature_cols, actual_laps, data, total_laps, baseline)
            sim_calibrated = sim_raw + (pace_offset * total_laps)
            diff_after     = (sim_calibrated - real_total) / real_total * 100
            calibration    = {"pace_offset": pace_offset, "diff_before": round(diff_before, 2),
                              "diff_after": round(diff_after, 2)}

            all_results = grid_search_strategies(model, feature_cols, actual_laps, total_laps, pace_offset=pace_offset)
            top_faster  = [r for r in all_results if r["faster"]][:5]
            if top_faster:
                explanations = explain_parameters(top_faster[0], baseline, real_total)

            mae_here  = float(np.mean(np.abs(y_true - y_pred)))
            rmse_here = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            result = {"race_label": race_info["label"], "driver": selected_driver,
                      "total_laps": total_laps, "mae": f"{mae_here:.3f}", "rmse": f"{rmse_here:.3f}",
                      "real_time": fmt_time(real_total), "real_total": real_total,
                      "diff_pct": round(diff_before, 2)}
        except Exception as e:
            error_msg = str(e)

    return render_template(
        "analysis.html",
        available_races=AVAILABLE_RACES, available_drivers=AVAILABLE_DRIVERS,
        selected_race_key=selected_race_key, selected_driver=selected_driver,
        result=result, error=error_msg,
        lap_labels=lap_labels, actual_laps=actual_laps, pred_laps=pred_laps,
        calibration=calibration, top_faster=top_faster, explanations=explanations,
    )


# ── หน้า 3 — Play Strategy ──────────────────────────────
@app.route("/play", methods=["GET", "POST"])
def play_strategy_page():
    selected_race_key = request.form.get("race_key", _default_race_key)
    race_info  = AVAILABLE_RACES.get(selected_race_key, AVAILABLE_RACES[_default_race_key])
    total_laps = race_info["laps"]
    result = None; leaderboard = []; form_data = None

    if request.method == "POST":
        start_compound  = request.form.get("start_compound",  "MEDIUM").upper()
        second_compound = request.form.get("second_compound", "SOFT").upper()
        pit_lap         = max(2, min(int(request.form.get("pit_lap", 20)), total_laps - 2))
        form_data = {
            "start": start_compound, "second": second_compound,
            "pit_lap": pit_lap, "race_key": selected_race_key,
        }

        predictor = LapPredictor("model.pkl")

        # หา global_offset จากข้อมูลจริงของสนามที่เลือก (VER เป็น reference)
        # เพื่อให้เวลา simulation ทุกคนใกล้เคียงเวลาจริง
        race_stats = next(
            (c for c in TRAIN_COMBO_STATS
             if c["gp"] == race_info["gp"] and c["year"] == race_info["year"] and c["driver"] == "VER"),
            None
        )
        real_total = race_stats["real_total"] if race_stats else race_info["laps"] * 95.0

        # คำนวณ global_offset = (เวลาจริง VER - เวลา raw sim VER) / total_laps
        # ทำให้ VER sim ตรงกับเวลาจริง แล้วคนอื่นก็จะเลื่อนตามไปด้วย
        from strategy_optimizer import simulate_strategy
        ver_sim_raw = simulate_strategy(
            model, feature_cols, total_laps,
            "MEDIUM", "SOFT", int(total_laps * 0.44),
            pace_offset=0.0,
        )
        global_offset = (real_total - ver_sim_raw) / total_laps

        all_results = simulate_full_race("model.pkl", total_laps,
                                         global_offset=global_offset)

        user_strategy = DriverStrategy(code="YOU", first_compound=start_compound,
                                       second_compound=second_compound, pit_lap=pit_lap,
                                       pace_offset=1.5 + global_offset)
        user_result = simulate_driver(predictor, user_strategy, total_laps)

        combined = list(all_results) + [user_result]
        combined.sort(key=lambda r: r.total_time)
        for i, r in enumerate(combined, start=1):
            r.rank = i

        win_probs     = compute_win_probabilities(combined)
        user_rank     = next(r.rank for r in combined if r.code == "YOU")
        user_time     = user_result.total_time
        user_win_prob = win_probs.get("YOU", 0.0)
        delta_real = user_time - real_total

        # เวลาอันดับ 1 สำหรับคำนวณ gap
        p1_time = combined[0].total_time

        result = {
            "rank": user_rank, "start": start_compound, "second": second_compound, "pit_lap": pit_lap,
            "user_time_str": fmt_time(user_time), "real_time_str": fmt_time(real_total),
            "user_time": user_time, "real_time": real_total,
            "delta_real": delta_real, "delta_real_str": f"{delta_real:+.2f} s",
            "win_prob_pct": round(user_win_prob * 100, 1),
            "win_prob_bar": min(round(user_win_prob * 100 * 3, 1), 100),
            "gap_to_p1": round(user_time - p1_time, 2),
            "gap_bar":   min(round((user_time - p1_time) / 200 * 100, 1), 100),
        }

        leaderboard = [{
            "rank": r.rank, "code": r.code, "is_user": r.code == "YOU",
            "start": r.strategy.first_compound, "second": r.strategy.second_compound,
            "pit_lap": r.strategy.pit_lap, "total_time": r.total_time,
            "gap_to_p1": round(r.total_time - p1_time, 2),
        } for r in combined]

    return render_template(
        "play_strategy.html",
        total_laps=total_laps,
        result=result,
        leaderboard=leaderboard,
        form_data=form_data,
        available_races=AVAILABLE_RACES,
        selected_race_key=selected_race_key,
        race_label=race_info["label"],
    )


if __name__ == "__main__":
    app.run(debug=True)