from preprocessing import preprocess
from train import train_model

if __name__ == "__main__":
    train_loader = preprocess(timesteps=4)
    train_model(train_loader)
