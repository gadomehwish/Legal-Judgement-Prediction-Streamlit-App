import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
import numpy as np

# Set page configuration
st.set_page_config(page_title="Legal Judgment Predictor", page_icon="⚖️", layout="centered")

# === Load & train and Prepare Data and Model (Silent) ===

@st.cache_resource(show_spinner=False)
def load_and_train_model(data_path):
    if not os.path.exists(data_path):
        st.error(f"Error: File not found at {data_path}")
        st.stop()

    df = pd.read_excel(data_path)

    text_columns = ['case_facts', 'evidence', 'precedents', 'prior_convictions']
    target_columns = ['legal_issue', 'suggested_punishment', 'relevant_articles', 'judgment', 'reasoning']

    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna('')

    df[target_columns] = df[target_columns].astype(str)

    vectorizer = TfidfVectorizer()
    available_text_cols = [col for col in text_columns if col in df.columns]
    X_text = vectorizer.fit_transform(df[available_text_cols].apply(lambda x: ' '.join(x.astype(str)), axis=1))
    X_text_df = pd.DataFrame(X_text.toarray(), columns=vectorizer.get_feature_names_out())

    X = X_text_df
    y = df[target_columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultiOutputClassifier(RandomForestClassifier(random_state=42))
    model.fit(X_train, y_train)

    # Save model (optional)
    joblib.dump(model, 'legal_judgment_prediction_model.pkl')

    return model, vectorizer, available_text_cols, X_train.columns

DATA_PATH = r"C:\Users\Mehwish\Desktop\Legal Recommendation App\DATASET.xlsx"
model, vectorizer, available_text_cols, model_features = load_and_train_model(DATA_PATH)

# === Streamlit UI ===

st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #0288d1;
        color: white;
        height: 40px;
        width: 100%;
        font-size: 18px;
        font-weight: bold;
        border-radius:8px;
        transition: background-color 0.3s ease;
        margin-top: 20px;
    }
    .stButton>button:hover {
        background-color: #0277bd;
        color: white;
    }
    .block-container {
        max-width: 700px;
        padding-top: 30px;
        padding-bottom: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚖️ Legal Judgment Predictor")

st.markdown(
    """
    Enter case details below and click 'Predict Judgment' to get legal recommendations based on the trained model.
    """
)

# Input text areas
case_facts = st.text_area("Case Facts", height=150, max_chars=5000)
evidence = st.text_area("Evidence", height=150, max_chars=5000)
precedents = st.text_area("Precedents", height=150, max_chars=5000)
prior_convictions = st.text_area("Prior Convictions", height=150, max_chars=5000)

if st.button("Predict Judgment"):
    with st.spinner("Predicting..."):
        try:
            input_data = {
                'case_facts': [case_facts],
                'evidence': [evidence],
                'precedents': [precedents],
                'prior_convictions': [prior_convictions]
            }
            input_df = pd.DataFrame(input_data)
            input_df.fillna('', inplace=True)

            combined_text = input_df[available_text_cols].apply(lambda x: ' '.join(x.astype(str)), axis=1)
            input_text_vectorized = vectorizer.transform(combined_text)
            input_text_df = pd.DataFrame(input_text_vectorized.toarray(), columns=vectorizer.get_feature_names_out())

            X_input = pd.DataFrame(index=[0])
            X_input = pd.concat([X_input, input_text_df], axis=1)

            missing_cols = set(model_features) - set(X_input.columns)
            for col in missing_cols:
                X_input[col] = 0
            X_input = X_input[model_features]

            prediction = model.predict(X_input)[0]

            st.markdown("### Prediction Result:")
            st.success(f"**Predicted Legal Issue:** {prediction[0]}\n\n"
                       f"**Suggested Punishment:** {prediction[1]}\n\n"
                       f"**Relevant Articles:** {prediction[2]}\n\n"
                       f"**Judgment:** {prediction[3]}\n\n"
                       f"**Reasoning:** {prediction[4]}")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
