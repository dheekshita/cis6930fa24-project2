import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from unredactor import evaluate_model

def test_evaluate_model():
    validation_data = pd.DataFrame({
        "context": ["Good", "Bad"],
        "name": [1, 0]
    })
    vectorizer = TfidfVectorizer().fit(validation_data["context"])
    X = vectorizer.transform(validation_data["context"])
    model = LogisticRegression().fit(X, validation_data["name"])

    y_pred = evaluate_model(model, vectorizer, validation_data)

    assert y_pred is not None
    assert len(y_pred) == len(validation_data)
