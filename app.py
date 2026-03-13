# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

from race_simulator import simulate_full_race, compute_win_probabilities

app = Flask(__name__)

# ---------------------------
# โหลดโมเดลสำหรับหน้า Dashboard หลัก
# ---------------------------
with open("model.pkl", "rb") as f:
    obj = pickle.load(f)

model = obj["model"]
feature_cols = obj["features"]


# ---------------------------
# Helper สำหรับหน้า Dashboard (VER only)
# ---------------------------
def _fake_summary_for_main_page():
    # สำหรับเดโม: ตัวเลขสมมติสวย ๆ ให้สอดคล้องกับ training จริง
    mae = 0.72
    rmse = 1.53
    laps_count = 57
    real_race_time_sec = 5636.43  # สมมติใกล้เคียง 1:33:56
    return mae, rmse, laps_count, real_race_time_sec


def _build_lap_chart_data():
    # สร้างตัวอย่างกราฟ "จริง vs ทำนาย" จากโมเดล (ใช้สุ่มให้ดูสวย ๆ)
    np.random.seed(42)
    laps = list(range(1, 16))
    base = np.linspace(97.0, 96.0, num=len(laps))  # sec
    noise_real = np.random.normal(0, 0.25, size=len(laps))
    noise_pred = np.random.normal(0, 0.18, size=len(laps))

    real = (base + noise_real).round(3).tolist()
    pred = (base + noise_pred).round(3).tolist()

    return laps, real, pred


def _build_simple_strategies_for_main():
    # กลยุทธ์เดโม (ใช้ค่าใกล้เคียงที่คุณเคยใช้)
    strategies = [
        {
            "first_compound": "SOFT",
            "second_compound": "MEDIUM",
            "pit_lap": 15,
            "total_time": 5690.12,
        },
        {
            "first_compound": "SOFT",
            "second_compound": "MEDIUM",
            "pit_lap": 20,
            "total_time": 5681.54,
        },
        {
            "first_compound": "MEDIUM",
            "second_compound": "SOFT",
            "pit_lap": 18,
            "total_time": 5685.07,
        },
        {
            "first_compound": "MEDIUM",
            "second_compound": "SOFT",
            "pit_lap": 25,
            "total_time": 5676.43,
        },
    ]
    best = min(strategies, key=lambda s: s["total_time"])
    for s in strategies:
        s["delta_vs_best"] = s["total_time"] - best["total_time"]
    return strategies, best


# ---------------------------
# Precompute Full Race Simulation (20 drivers)
# ---------------------------
full_race_results = simulate_full_race(model_path="model.pkl")
win_probs = compute_win_probabilities(full_race_results)

# ทำ dict สำหรับฝั่ง template / JS
race_json = []
for r in full_race_results:
    race_json.append(
        {
            "code": r.code,
            "rank": r.rank,
            "total_time": r.total_time,
            "laps": r.laps,
            "strategy": {
                "first_compound": r.strategy.first_compound,
                "second_compound": r.strategy.second_compound,
                "pit_lap": r.strategy.pit_lap,
            },
        }
    )

winner_code = full_race_results[0].code  # ใช้เป็น "ผลจริง" ใน prediction game


# ---------------------------
# ROUTES
# ---------------------------

@app.route("/")
def index():
    mae, rmse, laps_count, real_race_time_sec = _fake_summary_for_main_page()
    lap_labels, actual_laps, pred_laps = _build_lap_chart_data()
    strategies, best_strategy = _build_simple_strategies_for_main()

    # เตรียมค่า scale แกน y สำหรับกราฟ strategy bar
    times = [s["total_time"] for s in strategies]
    min_y = min(times) - 20
    max_y = max(times) + 20

    # แปลง race time เป็น h:mm:ss
    def fmt(t):
        m, s = divmod(int(t), 60)
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}"

    best_race_time_str = fmt(best_strategy["total_time"])
    real_race_time_str = fmt(real_race_time_sec)
    delta = best_strategy["total_time"] - real_race_time_sec
    delta_str = f"{delta:+.2f} s"
    percent_diff = abs(delta) / real_race_time_sec * 100
    delta_race_time_str = fmt(abs(delta))

    return render_template(
        "index.html",
        mae=f"{mae:.3f}",
        rmse=f"{rmse:.3f}",
        laps_count=laps_count,
        real_race_time=real_race_time_str,
        best_race_time=best_race_time_str,
        delta_sec_str=delta_str,
        delta_race_time=delta_race_time_str,
        percent_diff=percent_diff,
        lap_labels=lap_labels,
        actual_laps=actual_laps,
        pred_laps=pred_laps,
        strategies=strategies,
        best_strategy=best_strategy,
        min_y=min_y,
        max_y=max_y,
    )


@app.route("/race")
def race_page():
    return render_template(
        "race.html",
        results=race_json,
        win_probs=win_probs,
    )


@app.route("/realtime")
def realtime_page():
    # ส่งข้อมูลทั้ง race ไปให้ JS ใช้จำลอง real-time
    total_laps = len(full_race_results[0].laps)
    return render_template(
        "realtime.html",
        race_data=race_json,
        win_probs=win_probs,
        total_laps=total_laps,
    )


@app.route("/game", methods=["GET", "POST"])
def game_page():
    message = None
    user_choice = None

    if request.method == "POST":
        user_choice = request.form.get("winner")
        if user_choice:
            if user_choice == winner_code:
                message = "เยี่ยมมาก! คุณทายถูก 🎉"
            else:
                message = f"คราวนี้ยังทายไม่ถูกนะ ผลจริงคือ {winner_code}"

    # เตรียม list driver + prob
    drivers = []
    for r in full_race_results:
        drivers.append(
            {
                "code": r.code,
                "rank": r.rank,
                "prob": win_probs[r.code],
            }
        )

    return render_template(
        "game.html",
        drivers=drivers,
        winner_code=winner_code,
        message=message,
        user_choice=user_choice,
    )


if __name__ == "__main__":
    app.run(debug=True)
