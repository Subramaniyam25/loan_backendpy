from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
import pickle  # Import pickle here

# Suppress all warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Load the models (Random Forest & XGBoost)
with open('rf_model.pkl', 'rb') as rf_file:
    model_rf = pickle.load(rf_file)

with open('xgb_model.pkl', 'rb') as xgb_file:
    best_model = pickle.load(xgb_file)

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

# Function to calculate loan EMI
def calculate_loan_emi(loan_amount, interest_rate, loan_term):
    R = interest_rate / (12 * 100)
    emi = (loan_amount * R * (1 + R) ** loan_term) / ((1 + R) ** loan_term - 1)
    return emi

# Function to determine the interest rate based on CIBIL score
def get_interest_rate(cibil_score):
    if cibil_score < 600:
        return 15  # Very high risk
    elif 600 <= cibil_score <= 649:
        return 12  # High risk
    elif 650 <= cibil_score <= 749:
        return 9   # Moderate risk
    elif cibil_score >= 750:
        return 7   # Low risk

# Prediction function
def predict_loan_eligibility(manual_input):
    input_df = pd.DataFrame([manual_input])

    # Apply label encoding
    for col, le in label_encoders.items():
        input_df[col] = le.transform([manual_input[col]])

    input_df['Debt_to_Income'] = (input_df['Existing_EMI'] + input_df['Rent_Amount']) / input_df['Total_Income']

    # Predict using Random Forest
    rf_prob = model_rf.predict_proba(input_df)[:, 1][0]
    custom_threshold = 0.45
    rf_status = "Approved" if rf_prob >= custom_threshold else "Rejected"

    # Predict using XGBoost
    xgb_prob = best_model.predict_proba(input_df)[:, 1][0]
    custom_threshold = 0.36
    status = "Approved" if ((xgb_prob >= custom_threshold) and rf_prob >= 0.3) else "Rejected"

    reasons = []
    if status == "Rejected":
        if manual_input["Cibil_Score"] < 600:
            reasons.append("Low CIBIL Score (< 600) – indicates weak credit history.")
        dti = input_df['Debt_to_Income'].values[0]
        if dti > 0.5:
            reasons.append(f"High Debt-to-Income Ratio ({dti:.2f}) – should be less than 0.5.")
        if manual_input["Total_Income"] < 15000:
            reasons.append(f"Low Income (₹{manual_input['Total_Income']}) – minimum ₹15,000 recommended.")
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
    print("Received data from React:", data)  # Log incoming data

    student = 0
    if(data['Employment_Status']=="Student"):
        student = 1
    print('Student status',student)

    try:
        # Ensuring correct conversion of the data types
        manual_input = {
            'Gender': data['Gender'],
            'Maritial_Status': data['Maritial_Status'],
            'Dependents': int(data['Dependents']),
            'Age': int(data['Age']),
            'Education': data['Education'],
            'Employment_Status': data['Employment_Status'].replace('Student', 'Not Employed'),
            'Total_Income': int(data['Total_Income']),
            'Existing_EMI': int(data['Existing_EMI']),
            'Residential_Status': data['Residential_Status'],
            'Rent_Amount': int(data['Rent_Amount']),
            'Cibil_Score': int(data['Cibil_Score']),
            'Loan_Amount': int(data['Loan_Amount']),
            'Loan_Term': int(data['Loan_Term']),
        }

        # Log the processed manual_input data to ensure it's correctly parsed
        print("Processed manual_input:", manual_input)

        # Initialize an empty list to store reasons
        reasons = []

        # Condition checks before passing to model (without returning 400 errors)
        if manual_input['Age'] > 55:
            reasons.append("Age is greater than 55, which might affect eligibility.")
        if manual_input['Employment_Status'].lower() == 'not employed' and student != 1:
            reasons.append("Employment status is 'Not Employed', which is risky for approval.")
        if manual_input['Cibil_Score'] < 600:
            reasons.append("CIBIL Score is below 600, which affects loan eligibility.")

        # Proceed with EMI calculation and further checks
        interest_rate = get_interest_rate(manual_input['Cibil_Score'])
        loan_emi = calculate_loan_emi(manual_input['Loan_Amount'], interest_rate, manual_input['Loan_Term'])
        print(f"Calculated loan EMI: ₹{loan_emi:.2f} for Loan Amount: ₹{manual_input['Loan_Amount']} with interest rate: {interest_rate}%")

        # Subtract Existing EMI, Rent Amount, and Loan EMI from Total Income
        remaining_income = manual_input['Total_Income'] - (manual_input['Existing_EMI'] + manual_input['Rent_Amount'] + loan_emi)
        print(f"Remaining Income after deductions: ₹{remaining_income:.2f}")

        # Ensure remaining income is greater than ₹8,000
        if remaining_income <= 1:
            reasons.append("You are Unable to Pay EMI due to low income")

        # If reasons list is not empty, reject the application with reasons
        if reasons:
            return jsonify({
                "status": "Rejected",
                "reasons": reasons
            })

        # If all conditions are passed, predict loan eligibility
        result = predict_loan_eligibility(manual_input)
        return jsonify(result)

    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({"error": str(e)}), 500

# Run app
if __name__ == '__main__':
    app.run(debug=True)
