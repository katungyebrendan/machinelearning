import streamlit as st
import pandas as pd
import requests

# FastAPI prediction endpoint URL
FASTAPI_PREDICT_URL = "http://127.0.0.1:8000/predict"

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
        
        # Check if the CSV has exactly 5 columns corresponding to the expected features
        if df.shape[1] != 5:
            st.error("Error: CSV file must have exactly 5 columns corresponding to the features: Tick, Cape, Cattle, Bio5, and Cluster.")
        else:
            # Let user decide whether to use the teacher model
            use_teacher_model = st.checkbox("Use Teacher Model", value=False)
            if st.button("Predict for All Rows"):
                results = []
                for idx, row in df.iterrows():
                    features = row.tolist()
                    payload = {"features": features, "use_teacher_model": use_teacher_model}
                    response = requests.post(FASTAPI_PREDICT_URL, json=payload)
                    if response.status_code == 200:
                        results.append(response.json())
                    else:
                        st.error(f"Row {idx} Error: {response.text}")
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
    cluster = st.number_input("Cluster", value=0.0)  # If the cluster is generated automatically, you might omit this field
    
    use_teacher_model = st.checkbox("Use Teacher Model", value=False)
    
    if st.button("Predict"):
        # Assemble the features in the order the model expects
        features = [tick, cape, cattle, bio5, cluster]
        payload = {"features": features, "use_teacher_model": use_teacher_model}
        response = requests.post(FASTAPI_PREDICT_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success("Prediction Successful!")
            st.write("**Prediction:**", result.get("prediction", "N/A"))
            st.write("**Confidence:**", result.get("confidence", "N/A"))
            st.write("**Model used:**", result.get("model_used", "N/A"))
        else:
            st.error(f"Error: {response.text}")
