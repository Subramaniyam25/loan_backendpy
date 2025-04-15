from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Suppress all warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Load and preprocess dataset
df = pd.read_csv(r'./loan_eligibility_dataset.csv')
df.ffill(inplace=True)

# Label encode categorical columns
label_encoders = {}
categorical_cols = ['Gender', 'Maritial_Status', 'Education', 'Employment_Status', 'Residential_Status']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode target
df['Loan_Status'] = df['Loan_Status'].apply(lambda x: 1 if x == 'Approved' else 0)

# Feature engineering
df['Debt_to_Income'] = (df['Existing_EMI'] + df['Rent_Amount']) / df['Total_Income']

# Train/test split
X = df.drop(columns=['Loan_Id', 'Loan_Status'])
y = df['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Sample weight for imbalance
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Hyperparameter tuning for XGBoost
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 4],
    'learning_rate': [0.01, 0.05]
}

xgb = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=6,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train, sample_weight=sample_weights)
best_model = search.best_estimator_
best_model.fit(X_train, y_train)

# Train Random Forest
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Prediction function
def predict_loan_eligibility(manual_input):
    input_df = pd.DataFrame([manual_input])

    # Apply label encoding
    for col, le in label_encoders.items():
        input_df[col] = le.transform([manual_input[col]])

    input_df['Debt_to_Income'] = (input_df['Existing_EMI'] + input_df['Rent_Amount']) / input_df['Total_Income']

    rf_prob = model_rf.predict_proba(input_df)[:, 1][0]
    custom_threshold=0.45
    b = "Approved" if rf_prob >= custom_threshold else "Rejected"
    print('rfa prediction probability', rf_prob, 'status', b)

    xgb_prob = best_model.predict_proba(input_df)[:, 1][0]
    custom_threshold = 0.36
    status = "Approved" if ((xgb_prob >= custom_threshold) and rf_prob >= 0.3) else "Rejected"

    reasons = []
    if status == "Rejected":
        if manual_input["Cibil_Score"] < 650:
            reasons.append("Low CIBIL Score (< 650) – indicates weak credit history.")

        dti = input_df['Debt_to_Income'].values[0]
        if dti > 0.5:
            reasons.append(f"High Debt-to-Income Ratio ({dti:.2f}) – should be less than 0.5.")

        if manual_input["Total_Income"] < 25000:
            reasons.append(f"Low Income (₹{manual_input['Total_Income']}) – minimum ₹25,000 recommended.")

        if 'Employment_Type' in manual_input and manual_input['Employment_Type'].lower() in ['unemployed', 'student']:
            reasons.append(f"Employment Type: {manual_input['Employment_Type']} – may not be seen as stable.")

        if not reasons:
            fallback = []
            if manual_input["Dependents"] >= 4:
                fallback.append("High number of dependents – increases financial burden.")
            if manual_input["Age"] >= 50:
                fallback.append("Due to Higher Age Reason")
            if manual_input["Education"].lower() == "not graduate":
                fallback.append("Education level is below graduation – may reduce perceived stability.")
            if manual_input["Loan_Amount"] > 250000 and manual_input["Loan_Term"] <= 36:
                fallback.append("High loan amount with short term – leads to high monthly EMI.")

            if not fallback:
                fallback.append("Model flagged this profile as risky due to combined features.")
            reasons.extend(fallback)

    return {
    "status": str(status),
    "probability": float(round(xgb_prob, 2)),
    "rf_probability": float(round(rf_prob, 2)),
    "reasons": [str(reason) for reason in reasons]
}

# Flask endpoint
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print("Received data from React:", data)

    try:
        manual_input = {
            'Gender': data['Gender'],
            'Maritial_Status': data['Maritial_Status'],
            'Dependents': int(data['Dependents']),
            'Age': int(data['Age']),
            'Education': data['Education'],
            'Employment_Status': data['Employment_Status'],
            'Total_Income': int(data['Total_Income']),
            'Existing_EMI': int(data['Existing_EMI']),
            'Residential_Status': data['Residential_Status'],
            'Rent_Amount': int(data['Rent_Amount']),
            'Cibil_Score': int(data['Cibil_Score']),
            'Loan_Amount': int(data['Loan_Amount']),
            'Loan_Term': int(data['Loan_Term']),
        }
        result = predict_loan_eligibility(manual_input)
        return jsonify(result)
    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({"error": str(e)}), 500

# Run app
if __name__ == '__main__':
    app.run(debug=True)
