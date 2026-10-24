
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

DATA = "data/cleaned/metallic_glass_cleaned.csv"
TARGETS = ["Tg_[K]", "Tx_[K]", "Tl_[K]"]

df = pd.read_csv(DATA)

# Convert target columns to numeric values
for col in TARGETS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Use elemental composition fractions as model inputs
features = pd.read_csv(
    "data/features/dataset_with_elemental_fractions.csv"
)

X = features.select_dtypes(include="number")
X = X.drop(columns=TARGETS, errors="ignore")
y = df[TARGETS]

# Ensure rows align and remove incomplete records
n = min(len(X), len(y))
data = pd.concat(
    [X.iloc[:n].reset_index(drop=True),
     y.iloc[:n].reset_index(drop=True)],
    axis=1
).dropna()

X = data[X.columns]
y = data[TARGETS]

if len(data) < 10:
    raise ValueError("Not enough complete rows to train safely.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

for i, target in enumerate(TARGETS):
    print(f"{target}:")
    print("  MAE:", round(mean_absolute_error(
        y_test.iloc[:, i], pred[:, i]
    ), 3))
    print("  R2:", round(r2_score(
        y_test.iloc[:, i], pred[:, i]
    ), 3))

os.makedirs("models", exist_ok=True)
joblib.dump(
    {"model": model, "features": list(X.columns)},
    "models/temperature_model.joblib"
)
print("Saved model to models/temperature_model.joblib")
