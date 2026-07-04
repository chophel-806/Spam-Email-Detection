import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load("models/spam_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Configure the page
st.set_page_config(
    page_title="Spam Email Detection",
    page_icon="📧",
)

# Title
st.title("📧 Spam Email Detection")

st.write("Enter an email message below to check whether it is Spam or Ham.")

# Input box
email = st.text_area("Email Message")

# Predict button
if st.button("Predict"):

    if email.strip() == "":
        st.warning("Please enter an email message.")
    else:
        email_features = vectorizer.transform([email])
        prediction = model.predict(email_features)

        if prediction[0] == 1:
            st.error("🚨 Spam Email")
        else:
            st.success("✅ Ham (Not Spam)")