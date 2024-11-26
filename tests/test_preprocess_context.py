import pytest
from unredactor import preprocess_context

def test_preprocess_context():
    input_text = "This is ████████ a test ████."
    expected_output = "This is [REDACTED] a test [REDACTED]."
    assert preprocess_context(input_text) == expected_output
