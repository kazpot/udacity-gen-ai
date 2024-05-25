from datasets import load_dataset

# Load the IMDB dataset, which contains movie reviews
# and sentiment labels (positive or negative)
dataset = load_dataset("imdb")

# Fetch a review from the training set
review_number = 42
sample_review = dataset["train"][review_number]

print(sample_review["text"][:450] + "...")

if sample_review["label"] == 1:
    print("Sentiment: Positive")
else:
    print("Sentiment: Negative")