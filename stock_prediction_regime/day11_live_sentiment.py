from transformers import pipeline
import feedparser

classifier = pipeline(
    "text-classification",
    model="ProsusAI/finbert"
)

rss_url = (
    "https://news.google.com/rss/search?"
    "q=Infosys&hl=en-IN&gl=IN&ceid=IN:en"
)

feed = feedparser.parse(rss_url)

print("Articles Found:", len(feed.entries))
print()

headlines = []

for article in feed.entries[:25]:
    headlines.append(article.title)

results = classifier(headlines)

scores = []

for headline, result in zip(headlines, results):

    label = result["label"]
    confidence = result["score"]

    if label == "positive":
        score = confidence

    elif label == "negative":
        score = -confidence

    else:
        score = 0

    scores.append(score)

    print("Headline:")
    print(headline)

    print("Sentiment:", label)
    print("Confidence:", round(confidence, 4))
    print("Score:", round(score, 4))

    print("-" * 80)

overall_sentiment = sum(scores) / len(scores)

print("\nOverall Sentiment Score:")
print(round(overall_sentiment, 4))

if overall_sentiment > 0.25:
    print("Overall Sentiment: POSITIVE")

elif overall_sentiment < -0.25:
    print("Overall Sentiment: NEGATIVE")

else:
    print("Overall Sentiment: NEUTRAL")


print("\nTESTING FINBERT\n")

test_headlines = [
    "Infosys reports record profits and beats expectations",
    "Infosys faces major fraud investigation",
    "Infosys announces quarterly earnings"
]

test_results = classifier(test_headlines)

for headline, result in zip(test_headlines, test_results):

    print(headline)
    print(result)
    print()