import pytest
from unredactor import load_data

def test_load_data(tmp_path):
    mock_file = tmp_path / "example.tsv"
    mock_file.write_text("training\tAlice Bright\tThis is a test.\nvalidation\tTom Brady\tAnother test.")

    data = load_data(mock_file)

    assert len(data) == 2
    assert data['split'].tolist() == ["training", "validation"]
    assert data['name'].tolist() == ["Alice Bright", "Tom Brady"]
    assert data['context'].tolist() == ["This is a test.", "Another test."]
