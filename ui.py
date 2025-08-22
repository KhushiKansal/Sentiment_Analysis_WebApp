# import streamlit as st
# import requests

# st.title("Twitter Sentiment Analysis App")

# # Input box
# user_input = st.text_input("Enter a tweet:")

# # Button to send input to FastAPI
# if st.button("Analyze"):
#     if user_input.strip() != "":
#         response = requests.post("http://127.0.0.1:8000/predict", json={"text": user_input})
#         result = response.json()
#         st.write("Prediction:", result["sentiment"])
#     else:
#         st.warning("Please enter some text before analyzing.")
import streamlit as st
import pandas as pd
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import requests

# Load FastAPI endpoints
API_URL_SINGLE = "http://127.0.0.1:8000/predict"
API_URL_BATCH = "http://127.0.0.1:8000/batch_predict"

st.title("Twitter Sentiment Analysis")

# ---------------- Single text prediction ----------------
st.subheader("Single Text Sentiment")
text_input = st.text_area("Enter text for prediction:")

if st.button("Predict Sentiment"):
    if text_input:
        response = requests.post(API_URL_SINGLE, json={"text": text_input})
        if response.status_code == 200:
            result = response.json()
            st.write("Prediction:", result["sentiment"])
        else:
            st.error("Error contacting API")
    else:
        st.warning("Please enter some text")

# ---------------- CSV batch prediction ----------------
st.subheader("Batch Sentiment Analysis (CSV)")
uploaded_file = st.file_uploader("Upload CSV with 'text' column", type=["csv"])

if uploaded_file is not None:
    if st.button("Predict CSV Sentiments"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(API_URL_BATCH, files={"file": uploaded_file})
        if response.status_code == 200:
            df_result = pd.DataFrame(response.json())
            st.dataframe(df_result)

            # ---------------- WordClouds ----------------
            st.subheader("WordClouds by Sentiment")
            for sentiment in ["positive", "negative", "neutral"]:
                sentiment_text = " ".join(df_result[df_result['sentiment'] == sentiment]['text'].astype(str))
                if sentiment_text:
                    wc = WordCloud(width=800, height=400, background_color='white').generate(sentiment_text)
                    st.write(f"WordCloud for {sentiment} tweets")
                    plt.figure(figsize=(10,5))
                    plt.imshow(wc, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot(plt)
        else:
            st.error("Error contacting API")

