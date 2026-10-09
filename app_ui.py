
import streamlit as st
import pandas as pd
from transformers import pipeline

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💗",
    layout="wide"
)

# Pastel pink and lavender styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #fff5fa 0%, #f5f0ff 100%);
    color: #40354d;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #60466f !important;
}

p, label, .stMarkdown {
    color: #55465f;
}

.hero {
    background: linear-gradient(120deg, #f6cfe2, #dcd1ff);
    padding: 30px;
    border-radius: 22px;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0 6px 20px #b69cc533;
}

.hero h1 {
    color: #563c68 !important;
    margin-bottom: 8px;
    font-size: 2.3rem;
}

.hero p {
    color: #624c70;
    font-size: 1.05rem;
    margin-bottom: 0;
}

.panel {
    background: #ffffffc9;
    border: 1px solid #eadcf5;
    border-radius: 18px;
    padding: 22px;
    margin: 10px 0 22px 0;
    box-shadow: 0 4px 16px #b69cc51a;
}

.result-card {
    border-radius: 16px;
    padding: 20px;
    margin: 12px 0;
    border: 1px solid #eadcf5;
    background: white;
}

.result-card h3 {
    margin: 0 0 8px 0;
}

.result-card p {
    margin: 4px 0;
}

.metric-card {
    background: white;
    border: 1px solid #eadcf5;
    border-radius: 15px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 3px 12px #b69cc51a;
}

.metric-label {
    font-size: 0.9rem;
    color: #776482;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #60466f;
}

.stTextArea textarea {
    background: #fffaff;
    border: 1px solid #dfc9ee;
    border-radius: 12px;
    color: #40354d;
}

.stFileUploader section {
    background: #fffaff;
    border: 1px dashed #c9afe2;
    border-radius: 14px;
}

.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(100deg, #e9b9d5, #cbbcf5);
    color: #483354;
    border: none;
    border-radius: 12px;
    padding: 0.55rem 1.2rem;
    font-weight: 600;
    transition: 0.2s;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border: 1px solid #a88acb;
    color: #382548;
    transform: translateY(-1px);
}

div[data-testid="stDataFrame"] {
    border: 1px solid #eadcf5;
    border-radius: 12px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# Page header
st.markdown("""
<div class="hero">
    <h1> Sentiment Analysis 😊</h1>
    <p>Discover the emotions behind your words with AI.</p>
</div>
""", unsafe_allow_html=True)

st.write(
    "Analyze one review instantly or upload a CSV file "
    "to understand the sentiment of multiple reviews."
)

# Load the existing sentiment model
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

sentiment_analyzer = load_model()


def sentiment_color(label):
    label = label.lower()
    if label == "positive":
        return "#d9f4e5", "#218051", "😊"
    elif label == "negative":
        return "#ffe0e5", "#a33e56", "😞"
    return "#e9e0ff", "#694c9b", "😐"


# --------------------------------------------------
# 1. SINGLE REVIEW ANALYSIS
# --------------------------------------------------
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.header("✍️ 1. Analyze a Single Review")

text = st.text_area(
    "Enter your review:",
    placeholder="Example: I really enjoyed this product!",
    height=130,
    key="single_review"
)

if st.button("💗 Analyze Sentiment", key="single_analyze"):
    if text.strip():
        with st.spinner("Understanding your review..."):
            result = sentiment_analyzer(text.strip())[0]

        label = result["label"].capitalize()
        confidence = result["score"] * 100
        bg, fg, emoji = sentiment_color(label)

        st.markdown(f"""
        <div class="result-card" style="background:{bg};">
            <h3 style="color:{fg};">{emoji} {label} Sentiment</h3>
            <p style="color:{fg};">
                Confidence score: <b>{confidence:.2f}%</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(min(max(result["score"], 0.0), 1.0))
    else:
        st.warning("Please enter a review first.")

st.markdown("</div>", unsafe_allow_html=True)


# --------------------------------------------------
# 2. MULTIPLE REVIEW ANALYSIS
# --------------------------------------------------
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.header("📂 2. Analyze Multiple Reviews")

st.write("Upload a CSV file containing a column named `review`.")

uploaded_file = st.file_uploader(
    "Choose your CSV file",
    type=["csv"],
    key="review_csv"
)

if uploaded_file is not None:
    try:
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

            elif st.button("✨ Analyze All Reviews", key="bulk_analyze"):
                with st.spinner("Analyzing your reviews..."):
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

                st.subheader("🌸 Analysis Summary")

                total = len(df)
                summary = df["sentiment"].value_counts()
                positive = int(summary.get("Positive", 0))
                neutral = int(summary.get("Neutral", 0))
                negative = int(summary.get("Negative", 0))

                col1, col2, col3, col4 = st.columns(4)

                metrics = [
                    (col1, "Total Reviews", total, "#f5d8e9"),
                    (col2, "Positive 😊", positive, "#d9f4e5"),
                    (col3, "Neutral 😐", neutral, "#e9e0ff"),
                    (col4, "Negative 😞", negative, "#ffe0e5"),
                ]

                for column, title, value, color in metrics:
                    with column:
                        st.markdown(f"""
                        <div class="metric-card" style="background:{color};">
                            <div class="metric-label">{title}</div>
                            <div class="metric-value">{value}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.subheader("📊 Sentiment Distribution")

                order = ["Positive", "Neutral", "Negative"]
                percentages = (
                    summary.reindex(order, fill_value=0) / total * 100
                )

                bar_colors = {
                    "Positive": "#8ed6b0",
                    "Neutral": "#c6b2f3",
                    "Negative": "#f3a9bb"
                }

                for sentiment in order:
                    percentage = percentages[sentiment]
                    st.markdown(f"""
                    <div style="margin:14px 0;">
                        <div style="display:flex;justify-content:space-between;
                                    margin-bottom:5px;">
                            <span>{sentiment}</span>
                            <b>{percentage:.2f}%</b>
                        </div>
                        <div style="background:#eee5f4;border-radius:10px;
                                    height:15px;overflow:hidden;">
                            <div style="width:{percentage}%;
                                        background:{bar_colors[sentiment]};
                                        height:15px;border-radius:10px;">
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.subheader("📋 Detailed Results")
                st.dataframe(df, use_container_width=True)

                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Results as CSV",
                    data=csv_data,
                    file_name="sentiment_results.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"Could not read or analyze this file: {e}")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;padding:20px;color:#8a7497;">
    Made with 💗 using Python, Streamlit and AI
</div>
""", unsafe_allow_html=True)