import streamlit as st
import joblib

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📧 Email Spam Detection System")
st.write("Enter an email/message to check if it is Spam or Not")

message = st.text_area("Email Message")

if st.button("Predict"):
    data = vectorizer.transform([message])
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("🚫 Spam Message")
    else:
        st.success("✅ Not Spam (Ham)")
