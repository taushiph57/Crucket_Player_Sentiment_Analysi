import pandas as pd
from textblob import TextBlob

# Load the cricket comments
data = pd.read_csv("data/cricket_comments.csv")

# Function to find sentiment
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment analysis
data["sentiment"] = data["text"].apply(get_sentiment)

# Display results
print(data)

# Save results
data.to_csv("output/sentiment_results.csv", index=False)

print("\nSentiment analysis completed!")
print("Results saved in output/sentiment_results.csv")

import matplotlib.pyplot as plt

sentiment_counts = data["sentiment"].value_counts()

print("\nSentiment Counts:")
print(sentiment_counts)

sentiment_counts.plot(
    kind="bar",
    color=["green", "red", "blue"]
)

plt.title("Cricket Player Sentiment Analysis")
plt.xlabel("Sentiment")
plt.ylabel("Number of Comments")

plt.tight_layout()
plt.show()

# Player-wise sentiment analysis
player_sentiment = pd.crosstab(
    data["player"],
    data["sentiment"]
)

print("\nPlayer-wise Sentiment:")
print(player_sentiment) 

import matplotlib.pyplot as plt

player_sentiment.plot(kind="bar")

plt.title("Player-wise Sentiment Analysis")
plt.xlabel("Player")
plt.ylabel("Number of Comments")

plt.tight_layout()

plt.savefig("output/player_sentiment_chart.png")
plt.show()
print("\n--- PROJECT SUMMARY ---")

total_comments = len(data)
positive_comments = (data["sentiment"] == "Positive").sum()
negative_comments = (data["sentiment"] == "Negative").sum()
neutral_comments = (data["sentiment"] == "Neutral").sum()

print("Total Comments:", total_comments)
print("Positive Comments:", positive_comments)
print("Negative Comments:", negative_comments)
print("Neutral Comments:", neutral_comments)

print("\nProject completed successfully!")