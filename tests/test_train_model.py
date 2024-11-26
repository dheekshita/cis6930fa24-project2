import numpy as np
from unredactor import train_model

def test_train_model():
    X = np.random.rand(10, 2)
    y = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] 

    model = train_model(X, y)

    assert model is not None
    assert hasattr(model, "predict")
