import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from xgboost import XGBClassifier

# ===============================
# Load data
# ===============================
df = pd.read_csv("aug_train.csv")

# ===============================
# DROP UNWANTED COLUMNS (CRITICAL)
# ===============================
df = df.drop(columns=[
    "enrollee_id",
    "city",
    "city_development_index"
])

# ===============================
# Split X / y
# ===============================
y = df["target"]
X = df.drop(columns=["target"])

# ===============================
# Handle missing values
# ===============================
for col in X.columns:
    if X[col].dtype == "object":
        X[col].fillna(X[col].mode()[0], inplace=True)
    else:
        X[col].fillna(X[col].median(), inplace=True)

# ===============================
# Encode categorical features
# ===============================
encoders = {}
cat_cols = X.select_dtypes(include="object").columns

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# ===============================
# Train / Validation split
# ===============================
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# XGBoost tuning
# ===============================
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

param_dist = {
    "n_estimators": [200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

search = RandomizedSearchCV(
    xgb,
    param_distributions=param_dist,
    n_iter=15,
    scoring="f1",
    cv=3,
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)
best_model = search.best_estimator_

# ===============================
# Validation
# ===============================
val_preds = best_model.predict(X_val)
print(classification_report(y_val, val_preds))
print("Validation F1:", f1_score(y_val, val_preds))

# ===============================
# Train on full data
# ===============================
best_model.fit(X, y)

# ===============================
# Save artifacts
# ===============================
joblib.dump(best_model, "xgboost_fraud_model.pkl")
joblib.dump(encoders, "label_encoders.pkl")

print("✅ Model retrained with APP features only")
