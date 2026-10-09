from transformers import pipeline
import csv

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
)

results = []

with open("reviews.csv", "r", encoding="utf-8") as file:
    reviews = csv.DictReader(file)

    for row in reviews:
        text = row["review"]
        result = sentiment_analyzer(text)[0]

        results.append({
            "review": text,
            "sentiment": result["label"],
            "confidence": round(result["score"] * 100, 2)
        })

# Save results
with open("results.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["review", "sentiment", "confidence"]
    )

    writer.writeheader()
    writer.writerows(results)

print("Analysis completed!")
print("Results saved in results.csv")

# Sentiment summary
total_reviews = len(results)

positive_reviews = sum(
    1 for result in results
    if result["sentiment"] == "POSITIVE"
)

negative_reviews = sum(
    1 for result in results
    if result["sentiment"] == "NEGATIVE"
)

positive_percentage = (positive_reviews / total_reviews) * 100
negative_percentage = (negative_reviews / total_reviews) * 100

print("\n=== Sentiment Summary ===")
print("Total Reviews:", total_reviews)
print("Positive Reviews:", positive_reviews)
print("Negative Reviews:", negative_reviews)
print("Positive Percentage:", f"{positive_percentage:.2f}%")
print("Negative Percentage:", f"{negative_percentage:.2f}%")
# Analyze a custom review

print("\n=== Custom Sentiment Analysis ===")

text = input("Enter your review: ")

result = sentiment_analyzer(text)[0]

print("\nSentiment:", result["label"])
print("Confidence:", f"{result['score'] * 100:.2f}%")