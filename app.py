import streamlit as st
import pandas as pd
import requests

# FastAPI prediction endpoint URL
FASTAPI_PREDICT_URL = "https://eastcoast-fastapi.onrender.com/predict"

st.title("East Coast Fever Detection")
st.write("Upload your cow health data or enter features manually to get prediction results.")

# Choose input method
input_method = st.radio("Select input method:", ["Upload CSV", "Manual Input"])

if input_method == "Upload CSV":
    st.subheader("Upload Cow Health Data (CSV)")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Read CSV into DataFrame
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(df.head())
        
        # Check if the CSV has exactly 4 columns corresponding to the expected features
        if df.shape[1] != 4:
            st.error("Error: CSV file must have exactly 4 columns corresponding to the features: Tick, Cape, Cattle, and Bio5.")
        else:
            # Let user decide whether to use the teacher model
            use_teacher_model = st.checkbox("Use Teacher Model", value=False)
            if st.button("Predict for All Rows"):
                results = []
                for idx, row in df.iterrows():
                    features = row.tolist()  # Only 4 features: tick, cape, cattle, bio5
                    payload = {"features": features, "use_teacher_model": use_teacher_model}
                    try:
                        response = requests.post(FASTAPI_PREDICT_URL, json=payload)
                        if response.status_code == 200:
                            results.append(response.json())
                        else:
                            st.error(f"Row {idx} Error: {response.text}")
                    except requests.exceptions.ConnectionError as e:
                        st.error(f"Connection Error: Unable to connect to the FastAPI server. Please ensure the server is running.")
                if results:
                    st.subheader("Prediction Results")
                    st.write(pd.DataFrame(results))

elif input_method == "Manual Input":
    st.subheader("Enter Features Manually")
    # Create clearly labeled input fields for each feature
    tick = st.number_input("Tick", value=0.0)
    cape = st.number_input("Cape", value=0.0)
    cattle = st.number_input("Cattle", value=0.0)
    bio5 = st.number_input("Bio5", value=0.0)
    
    use_teacher_model = st.checkbox("Use Teacher Model", value=False)
    
    if st.button("Predict"):
        # Assemble the features in the order the model expects
        features = [tick, cape, cattle, bio5]  # Only 4 features
        payload = {"features": features, "use_teacher_model": use_teacher_model}
        try:
            response = requests.post(FASTAPI_PREDICT_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success("Prediction Successful!")
                st.write("**Prediction:**", result.get("prediction", "N/A"))
                st.write("**Confidence:**", result.get("confidence", "N/A"))
                st.write("**Model used:**", result.get("model_used", "N/A"))
            else:
                st.error(f"Error: {response.text}")
        except requests.exceptions.ConnectionError as e:
            st.error(f"Connection Error: Unable to connect to the FastAPI server. Please ensure the server is running.")
