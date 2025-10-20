# import libraries 
import spacy
import random
import pandas as pd

# load the spacy model
nlp = spacy.load("en_core_web_sm")

# load dataset kagggle amazon reviews 
file_path = "train.ft.txt"

# read first 100000 lines of the dataset
sample_size = 100000
data = []

with open(file_path, "r", encoding="utf-8") as f:
    for i, line in enumerate (f):
        if i >= sample_size:
            break
        parts = line.strip().split(" ", 1)
        if len(parts) == 2:
            label, review = parts
            data.append((label.replace("__label__", ""), review))

# create a dataframe
df = pd.DataFrame(data, columns=["label", "review"])
print (f"Loaded {len(df)} records successfully.")
print(df.head())

# function to perform NLP tasks - rule based sentiment analyzer
def analyze_sentiment(text):
    positive_words = ["good", "great", "excellent", "amazing", "fantastic", "love", "wonderful", "best", "perfect", "satisfied", "happy", "positive", "enjoyed", "pleased"]
    negative_words = ["bad", "terrible", "awful", "worst", "hate", "disappointing", "poor", "negative", "sad", "angry", "frustrated", "unsatisfied", "boring", "dull", "annoying", "problem"]

    text_lower = text.lower()
    score = 0
    for word in positive_words:
        if word in text_lower:
            score += 1
    for word in negative_words:
        if word in text_lower:
            score -= 1

    if score > 0:
        return "positive"
    elif score < 0:
        return "negative"
    else:
        return "neutral"

# apply sentiment analysis on the reviews
sample_reviews = df.sample(1000, random_state=42).reset_index(drop=True)

for i, review in enumerate(sample_reviews["review"], 1):
    doc = nlp(review)
    entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["ORG","PRODUCT"]]
    sentiment = analyze_sentiment(review)

    print (f"Review {i}: {review}")
    print (f"  Sentiment: {sentiment}")
    if entities:
        print (f"  Entities: {entities}")
    else:
        print (f"  Entities: None") 
    print ("-"*80)
    