import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="AEGIS Tactical Defense System",
    page_icon="🛡️",
    layout="wide"
)

# ---------------------------------------------------------
# PATH FIX & ARTIFACT LOADING
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
    model = joblib.load(os.path.join(BASE_DIR, 'calibrated_model.pkl'))
    encoders = joblib.load(os.path.join(BASE_DIR, 'encoders.pkl'))
    target_le = joblib.load(os.path.join(BASE_DIR, 'target_le.pkl'))
    feature_columns = joblib.load(os.path.join(BASE_DIR, 'feature_columns.pkl'))
    return scaler, model, encoders, target_le, feature_columns


try:
    scaler, model, encoders, target_le, feature_columns = load_artifacts()
except Exception as e:
    st.error(f"Error loading models or artifacts: {e}")
    st.stop()

# ---------------------------------------------------------
# UI LAYOUT & DYNAMIC INPUT FORM
# ---------------------------------------------------------
st.title("🛡️ AEGIS Calibrated Tactical Air Defense System")
st.write(f"Model loaded successfully with **{len(feature_columns)} clean features** (Leakage-Free).")

input_data = {}

with st.form("threat_form"):
    st.subheader("Target Signature Parameters")

    # 3 columns layout for dynamic features rendering
    cols = st.columns(3)

    for idx, col_name in enumerate(feature_columns):
        current_col = cols[idx % 3]

        if col_name in encoders:
            options = list(encoders[col_name].classes_)
            selected_val = current_col.selectbox(f"{col_name}", options)
            input_data[col_name] = encoders[col_name].transform([selected_val])[0]
        else:
            current_col.number_input(f"{col_name}", value=0.0, key=f"inp_{col_name}")
            # Note: Storing input values safely via st.session_state or direct mapping

    # For form elements safely capturing values via session state in newer streamlit or direct assignment:
    for col_name in feature_columns:
        if col_name not in encoders:
            input_data[col_name] = st.session_state.get(f"inp_{col_name}", 0.0)

    submit = st.form_submit_button("🚨 EXECUTE CALIBRATED THREAT ANALYSIS", use_container_width=True)

# ---------------------------------------------------------
# INFERENCE & PREDICTION LOGIC
# ---------------------------------------------------------
if submit:
    # Sanity Guard: Check if all numerical inputs are zero
    numeric_vals = [val for key, val in input_data.items() if key not in encoders]
    if all(v == 0.0 for v in numeric_vals):
        st.error("🚨 **CRITICAL WARNING:** All numerical inputs are zero! Please provide valid telemetry data.")
    else:
        input_df = pd.DataFrame([input_data])[feature_columns]

        # Scale inputs using MinMaxScaler
        scaled_input = scaler.transform(input_df)

        # Predict probabilities
        probabilities = model.predict_proba(scaled_input)[0]
        best_idx = np.argmax(probabilities)
        confidence = probabilities[best_idx] * 100
        predicted_target = target_le.inverse_transform([best_idx])[0]

        st.markdown("---")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.success(f"### Identified Target: **{predicted_target}**")
            st.metric(label="Calibrated Classification Confidence", value=f"{confidence:.2f}%")
            st.progress(float(probabilities[best_idx]))

        with res_col2:
            st.write("#### Class Probabilities Breakdown")
            classes = target_le.classes_
            prob_df = pd.DataFrame({
                'Target Class': classes,
                'Probability (%)': np.round(probabilities * 100, 2)
            }).sort_values(by='Probability (%)', ascending=False)

            st.dataframe(prob_df, hide_index=True, use_container_width=True)