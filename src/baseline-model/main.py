from preprocessing import preprocess_train
from preprocessing import preprocess_test

from train import train_model
from test_model import test_model

if __name__ == "__main__":
    train_loader, val_loader = preprocess_train(timesteps=4)
    test_loader = preprocess_test(timesteps=4)
    train_model(train_loader, val_loader)

    test_model(test_loader)