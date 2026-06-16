from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="ProsusAI/finbert"
)

headlines = [
    "Infosys reports strong quarterly earnings",
    "Reliance Industries faces regulatory investigation",
    "TCS wins major AI transformation contract"
]

results = classifier(headlines)

for headline, result in zip(headlines, results):

    label = result["label"]
    confidence = result["score"]

    if label == "positive":
        sentiment_score = confidence

    elif label == "negative":
        sentiment_score = -confidence

    else:
        sentiment_score = 0

    print("\nHeadline:")
    print(headline)

    print("Sentiment:")
    print(label)

    print("Confidence:")
    print(round(confidence, 4))

    print("Numeric Score:")
    print(round(sentiment_score, 4))