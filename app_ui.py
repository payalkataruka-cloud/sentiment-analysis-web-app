
import streamlit as st
import pandas as pd
from transformers import pipeline

st.set_page_config(page_title="Sentiment Analysis", page_icon="💭")

st.title("💭 Sentiment Analysis")
st.write("Analyze a single review or multiple reviews from a CSV file.")

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

sentiment_analyzer = load_model()

# Single review analysis
st.header("1. Analyze a Single Review")

text = st.text_area("Enter your review:")

if st.button("Analyze Sentiment"):
    if text.strip():
        result = sentiment_analyzer(text)[0]
        st.subheader("Result")
        st.write("**Sentiment:**", result["label"].capitalize())
        st.write("**Confidence:**", f"{result['score'] * 100:.2f}%")
    else:
        st.warning("Please enter a review.")

# Multiple review analysis
st.header("2. Analyze Multiple Reviews")

uploaded_file = st.file_uploader(
    "Upload a CSV file with a 'review' column",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "review" not in df.columns:
        st.error("Your CSV must contain a column named 'review'.")
    elif df.empty:
        st.warning("The CSV file contains no reviews.")
    else:
        df = df.dropna(subset=["review"]).copy()
        df["review"] = df["review"].astype(str)
        df = df[df["review"].str.strip() != ""]

        if df.empty:
            st.warning("No valid reviews were found.")
        else:
            if st.button("Analyze All Reviews"):
                with st.spinner("Analyzing reviews..."):
                    results = sentiment_analyzer(
                        df["review"].tolist(),
                        batch_size=8,
                        truncation=True
                    )

                df["sentiment"] = [
                    result["label"].capitalize()
                    for result in results
                ]
                df["confidence"] = [
                    round(result["score"] * 100, 2)
                    for result in results
                ]


                st.subheader("Analysis Summary")

                total = len(df)
                summary = df["sentiment"].value_counts()

                col1, col2, col3, col4 = st.columns(4)

                col1.metric("Total Reviews", total)
                col2.metric("Positive 😊", int(summary.get("Positive", 0)))
                col3.metric("Neutral 😐", int(summary.get("Neutral", 0)))
                col4.metric("Negative 😞", int(summary.get("Negative", 0)))

                st.subheader("Sentiment Distribution")

                percentages = (
                    summary.reindex(["Positive", "Neutral", "Negative"], fill_value=0)
                    / total * 100
                )

                for sentiment, percentage in percentages.items():
                    st.write(f"**{sentiment}:** {percentage:.2f}%")

                st.bar_chart(summary.reindex(
                    ["Positive", "Neutral", "Negative"], fill_value=0
                ))



                st.subheader("Detailed Results")
                st.dataframe(df, use_container_width=True)

                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Results as CSV",
                    data=csv_data,
                    file_name="sentiment_results.csv",
                    mime="text/csv"
                )

