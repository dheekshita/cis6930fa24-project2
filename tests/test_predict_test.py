import pandas as pd
from pathlib import Path
from unredactor import predict_test

def test_predict_test(tmp_path):
    test_file = tmp_path / "test.tsv"
    output_file = tmp_path / "submission.tsv"

    test_file.write_text("1\t[REDACTED] is nice\n2\t[REDACTED] is bad")

    class testModel:
        def predict(self, X):
            return ["Alice Bright", "Tom Brady"]

    class testVectorizer:
        def transform(self, X):
            return X

    test_model = testModel()
    test_vectorizer = testVectorizer()

    predict_test(test_model, test_vectorizer, test_file, output_file)

    # Assert output
    assert output_file.exists()
    output_data = pd.read_csv(output_file, sep='\t')
    assert len(output_data) == 2
    assert output_data["name"].tolist() == ["Alice Bright", "Tom Brady"]
