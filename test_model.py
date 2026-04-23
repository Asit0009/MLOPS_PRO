import os
from model import train_model

def test_train_model():
    accuracy = train_model()
    assert accuracy > 0.8
    assert os.path.exists('model.pkl')
    print("Tests passed successfully!")

if __name__ == "__main__":
    test_train_model()
