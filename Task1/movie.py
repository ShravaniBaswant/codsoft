import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
train_file = "train_data.txt"  

def load_data(file_path):  
    """Loads the dataset from a text file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(" ::: ")
            if len(parts) == 4:
                index, title, genre, plot = parts
                data.append((title, genre, plot))
    return pd.DataFrame(data, columns=["Title", "Genre", "Plot"])

# Load training data
df = load_data(train_file)

# Preprocess text
def clean_text(text):
    """Cleans text by removing special characters and converting to lowercase."""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)  # Remove special characters
    return text

df["Plot"] = df["Plot"].apply(clean_text)

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(df["Plot"], df["Genre"], test_size=0.2, random_state=42)

# Create a TF-IDF model with Naive Bayes
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Predict function
def predict_genre(plot):
    """Predicts the genre of a given movie plot."""
    plot = clean_text(plot)
    return model.predict([plot])[0]

# Example usage
sample_plot = "A young boy discovers he has magical powers and attends a school of wizardry."
print("Predicted Genre:", predict_genre(sample_plot))