import json
import joblib
import pandas as pd
import streamlit as st
import tensorflow as tf

# Page setup
st.set_page_config(
    page_title="Campus Placement Prediction",
    page_icon="🎓",
    layout="centered"
)

# Load model + preprocessor only once
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("artifacts/preprocessor.pkl")
    model = tf.keras.models.load_model("artifacts/dnn_model.keras")
    with open("artifacts/meta.json", "r") as f:
        meta = json.load(f)
    return preprocessor, model, meta

preprocessor, model, meta = load_artifacts()

st.title("🎓 Campus Placement Prediction System")
st.write("This application predicts whether a student will be placed based on academic and skill-related attributes.")

# ---------- INPUT FORM ----------
with st.form("placement_form"):

    st.subheader("Personal & Academic Details")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["M", "F"])
        ssc_b = st.selectbox("SSC Board", ["Central", "Others"])
        hsc_b = st.selectbox("HSC Board", ["Central", "Others"])
        workex = st.selectbox("Work Experience", ["Yes", "No"])

    with col2:
        hsc_s = st.selectbox("HSC Stream", ["Science", "Commerce", "Arts"])
        degree_t = st.selectbox("Degree Type", ["Sci&Tech", "Comm&Mgmt", "Others"])
        specialisation = st.selectbox("MBA Specialisation", ["Mkt&HR", "Mkt&Fin"])

    st.subheader("Scores (in percentage)")

    ssc_p = st.number_input("SSC Percentage", 0.0, 100.0, 70.0)
    hsc_p = st.number_input("HSC Percentage", 0.0, 100.0, 70.0)
    degree_p = st.number_input("Degree Percentage", 0.0, 100.0, 65.0)
    etest_p = st.number_input("Employability Test Percentage", 0.0, 100.0, 75.0)
    mba_p = st.number_input("MBA Percentage", 0.0, 100.0, 62.0)

    submit = st.form_submit_button("Predict Placement")

# ---------- PREDICTION ----------
if submit:
    input_df = pd.DataFrame([{
        "gender": gender,
        "ssc_p": ssc_p,
        "ssc_b": ssc_b,
        "hsc_p": hsc_p,
        "hsc_b": hsc_b,
        "hsc_s": hsc_s,
        "degree_p": degree_p,
        "degree_t": degree_t,
        "workex": workex,
        "etest_p": etest_p,
        "specialisation": specialisation,
        "mba_p": mba_p
    }])

    processed_input = preprocessor.transform(input_df)
    probability = float(model.predict(processed_input)[0][0])

    threshold = meta["threshold"]
    result = "Placed" if probability >= threshold else "Not Placed"

    st.subheader("Prediction Result")
    st.metric("Placement Probability", f"{probability*100:.2f}%")

    if result == "Placed":
        st.success("✅ Student is likely to be PLACED")
    else:
        st.error("❌ Student is likely to be NOT PLACED")
