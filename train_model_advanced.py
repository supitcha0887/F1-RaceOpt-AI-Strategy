# train_model_advanced.py
"""
เทรนโมเดล Random Forest ขั้นสูงสำหรับทำนาย Lap Time ของ F1
"""

import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_pipeline import load_race_laps


def train_advanced_model():
    print("=== โหลดข้อมูลจาก FastF1 ===")
    data, meta = load_race_laps(2023, "Bahrain", "VER")

    # แยก X, y
    y = data["LapTimeSec"].values
    X = data.drop(columns=["LapTimeSec"])

    feature_cols = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"จำนวนตัวอย่าง (ทั้งหมด): {len(X)}")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"จำนวนฟีเจอร์: {len(feature_cols)}")

    print("\n=== เทรนโมเดล RandomForest ขั้นสูง ===")
    model = RandomForestRegressor(
        n_estimators=800,
        max_depth=18,
        min_samples_split=2,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"MAE  = {mae:.3f} วินาที")
    print(f"RMSE = {rmse:.3f} วินาที")

    print("\n=== บันทึกโมเดลลงไฟล์ model.pkl ===")
    with open("model.pkl", "wb") as f:
        pickle.dump(
            {
                "model": model,
                "features": feature_cols,
                "meta": meta,
                "mae": mae,
                "rmse": rmse,
            },
            f,
        )

    print("เสร็จสิ้น 🎉")


if __name__ == "__main__":
    train_advanced_model()
