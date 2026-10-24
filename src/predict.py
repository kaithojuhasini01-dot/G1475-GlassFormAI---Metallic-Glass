
import pandas as pd
import joblib

MODEL_PATH = "models/temperature_model.joblib"
DATA_PATH = "data/features/dataset_with_elemental_fractions.csv"

saved = joblib.load(MODEL_PATH)
model = saved["model"]
feature_names = saved["features"]

df = pd.read_csv(DATA_PATH)
X = df.select_dtypes(include="number")
X = X.drop(
    columns=["Tg_[K]", "Tx_[K]", "Tl_[K]"],
    errors="ignore"
)

# Match the exact feature columns used during training
X = X.reindex(columns=feature_names, fill_value=0)
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

predictions = model.predict(X.head(5))

print("\nGLASSFORMAI: TEMPERATURE PREDICTIONS")
for i, values in enumerate(predictions, start=1):
    print(f"\nSample {i}")
    print(f"Predicted Tg: {values[0]:.2f} K")
    print(f"Predicted Tx: {values[1]:.2f} K")
    print(f"Predicted Tl: {values[2]:.2f} K")
