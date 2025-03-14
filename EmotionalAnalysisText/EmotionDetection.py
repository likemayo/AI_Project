from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Load and preprocess the dataset
data = pd.read_csv("./data/training.1600000.processed.noemoticon.csv", encoding="latin-1", header=None)
data.columns = ["target", "ids", "date", "flag", "user", "text"]

# Map labels: 0 -> 0 (negative), 2 -> 1 (neutral), 4 -> 2 (positive)
data["target"] = data["target"].map({0: 0, 4: 2})

# Data preprocessing
X = data["text"]
y = data["target"]
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_tfidf, y)

# Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json["text"]
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)
    sentiment = ["negative", "positive"][prediction[0]]
    return jsonify({"sentiment": sentiment})

if __name__ == "__main__":
    app.run(debug=True)
