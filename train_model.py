import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Sample dataset
texts = [
    "I love this!", "This is amazing", "I am happy", "Best experience",
    "I hate this", "This is terrible", "Very bad experience", "Worst ever",
    "Not good", "I dislike this", "Awful", "Bad and disappointing",
    "I enjoy this", "I don't like it", "It's okay", "Could be better"
]
labels = [
    "Positive", "Positive", "Positive", "Positive",
    "Negative", "Negative", "Negative", "Negative",
    "Negative", "Negative", "Negative", "Negative",
    "Positive", "Negative", "Neutral", "Neutral"
]

# Convert text to feature vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train model
model = LogisticRegression()
model.fit(X, labels)

# Save model
with open('model/sentiment_model.pkl', 'wb') as f:
    pickle.dump((vectorizer, model), f)

print("✅ Model trained and saved.")