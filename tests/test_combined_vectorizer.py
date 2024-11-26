import pandas as pd
from unredactor import combined_vectorizer

def test_combined_vectorizer():
    reviews = pd.DataFrame({"context": ["Positive review", "Negative review"]})
    unredactor_data = pd.DataFrame({"context": ["[REDACTED] is good", "[REDACTED] is bad"]})

    vectorizer = combined_vectorizer(reviews, unredactor_data)

    assert vectorizer is not None
    assert hasattr(vectorizer, "transform")
