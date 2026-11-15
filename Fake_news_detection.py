import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Sample dataset (small, for demonstration)
data = {
    "news": [
        "Government launches new education policy",
        "Aliens landed in India yesterday",
        "Scientists discovered a new vaccine",
        "Celebrity secretly married an alien",
        "New technology improves mobile battery",
        "Ghost spotted in parliament building",
        "Central bank announces new interest rate decision",
        "Man claims he lives on the moon",
        "University opens new research centre",
        "Miracle drink makes people live 500 years",
        "City launches new metro line",
        "Time traveller predicts next election winner",
    ],
    "label": ["Real", "Fake"] * 6,
}
df = pd.DataFrame(data)

# Convert text into numbers
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["news"])
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nConfusion matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification report:\n",
      classification_report(y_test, predictions, zero_division=0))

# Predict new news
new_news = ["Aliens found on the moon"]
new_data = vectorizer.transform(new_news)
print("Prediction:", model.predict(new_data)[0])
