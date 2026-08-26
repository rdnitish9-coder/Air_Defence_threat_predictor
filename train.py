import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE

# 1. Load Dataset
df = pd.read_csv('ML_project.csv')

# LEAKAGE FIX: Saare target columns, labels, aur ID fields ko strictly drop karo
drop_columns = [
    'Target_Class',
    'Target_Label',
    'Object_ID',
    'Latitude',
    'Longitude',
    'Heading_deg',
    'Pitch_deg',
    'Roll_deg',
    'Yaw_deg',
    'Electronic_Jamming'
]

df_clean = df.drop(columns=drop_columns, errors='ignore')

# Features (X) aur Target (y) alag karo
X = df_clean.copy()
y = df['Target_Class'] # Original target column

# 2. Label Encoding for Categorical Columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

target_le = LabelEncoder()
y_encoded = target_le.fit_transform(y)

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 4. SCALING FIX: MinMaxScaler use kar rahe hain taaki 0 ka matlab actual zero ho
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 5. Handle Class Imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# 6. Advanced XGBoost Model + Calibration
base_model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

calibrated_model = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv=5)
calibrated_model.fit(X_train_res, y_train_res)

# 7. Save Artifacts securely
feature_columns = list(X.columns)
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(calibrated_model, 'calibrated_model.pkl')
joblib.dump(encoders, 'encoders.pkl')
joblib.dump(target_le, 'target_le.pkl')
joblib.dump(feature_columns, 'feature_columns.pkl')

print(f"✅ Success! Model trained on {X.shape[1]} clean, leakage-free features.")