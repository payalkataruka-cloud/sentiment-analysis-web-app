# Sentiment Analysis Web App using Python

## About the Project

This project analyzes text reviews and classifies their sentiment as Positive, Neutral, or Negative using a pretrained Transformer model from Hugging Face.

The application is built with Python and Streamlit. It supports individual review analysis and bulk analysis through CSV uploads.
# 💗 Sentiment Analysis Web App

**🌐 Live Demo:** [Open Sentiment Analysis App](https://payalkataruka-cloud-sentiment-analysis-web-app-app-ui-y5xjy4.streamlit.app/)

An AI-powered web application that analyzes text reviews and classifies their sentiment as positive, neutral, or negative using Python, Streamlit, and a pretrained NLP model.


## Features

* Analyze individual text reviews.
* Analyze multiple reviews using CSV upload.
* Classify sentiment as Positive, Neutral, or Negative.
* Display confidence scores.
* Show sentiment counts, percentages, and a bar chart.
* Display detailed results.
* Download results as a CSV file.

## Technologies Used

* Python
* Streamlit
* Hugging Face Transformers
* CardiffNLP RoBERTa sentiment model
* Pandas

## How to Run the Project

1. Install Python.

2. Install the required libraries:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Open PowerShell in the project folder.

4. Run the application:

   ```bash
   python -m streamlit run app_ui.py
   ```

5. Open the local URL displayed in the terminal.

## Project Files

* `app_ui.py` — Streamlit web interface.
* `app.py` — Terminal-based sentiment analysis.
* `requirements.txt` — Required Python libraries.
* `sample_review.csv` — Sample reviews for testing.
* `results.csv` — Generated analysis results.

## Model

The application uses `cardiffnlp/twitter-roberta-base-sentiment-latest`, a pretrained model that predicts positive, neutral, and negative sentiment.

## Limitations

Predictions may be inaccurate for sarcasm, mixed opinions, and context-dependent statements. Confidence scores do not guarantee correctness.

## Author

Payal Kataruka
