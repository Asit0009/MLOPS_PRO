import os
import pytest
from model import train_model

def test_train_model():
    accuracy = train_model()
    assert accuracy > 0.8
    assert os.path.exists('model.pkl')
